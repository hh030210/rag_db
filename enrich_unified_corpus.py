# -*- coding: utf-8 -*-
"""
enrich_unified_corpus.py
========================
对 unified_corpus 已有 852 条 chunk 调 LLM 做维度抽取 + 打标签，写回 payload。

字段（payload 多列，无向量变更）：
    summary              str     1-2 句内容摘要
    keywords             str[]   关键词
    topic                str     主题标签
    spot_entities        str[]   景点名称
    person_entities      str[]   历史人物
    dynasty_entities     str[]   朝代
    era_entities         str[]   年代时间
    building_entities    str[]   建筑类型
    place_entities       str[]   地理位置
    event_entities       str[]   历史事件
    work_entities        str[]   作品/文献
    org_entities         str[]   机构/家族
    ritual_entities      str[]   祭祀/礼仪
    identity_entities    str[]   人物身份
    culture_entities     str[]   文化价值
    exhibit_entities     str[]   展出内容
    quality_score        float   内容质量自评 1-5

断点续跑 / 缓存：
    进度文件  cache/enrich_progress.jsonl 每行 {id, payload_fields, ts}
    完成后   cache/enrich_summary.json

用法：
    python enrich_unified_corpus.py                  # 全量
    python enrich_unified_corpus.py --limit 50       # 只跑前 50 条做试跑
    python enrich_unified_corpus.py --dry-run        # 不入库，仅生成缓存（不入 Qdrant）
    python enrich_unified_corpus.py --resume         # 跳过已缓存的
"""
import argparse
import json
import os
import sys
import time
import httpx
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent

# ── 配置 ───────────────────────────────────────────────────────────
COLLECTION = "unified_corpus"
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333

DS_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DS_MODEL = "qwen-plus"
DS_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DS_TIMEOUT = 60.0

CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)
PROGRESS_FILE = CACHE_DIR / "enrich_progress.jsonl"
SUMMARY_FILE = CACHE_DIR / "enrich_summary.json"

DIM_LIST = [
    "spot_entities", "person_entities", "dynasty_entities", "era_entities",
    "building_entities", "place_entities", "event_entities", "work_entities",
    "org_entities", "ritual_entities", "identity_entities",
    "culture_entities", "exhibit_entities",
]

EXTRACTION_SYSTEM_PROMPT = """你是中文知识库的"文本结构化助手"。你的任务是从一段文字中精准抽取指定的结构化字段。

抽取规则：
1. 严格按照 JSON 格式输出，不要包含额外说明。
2. 实体抽取要求：
   - 严格使用文本中出现的原文，不要改写/不要补全/不要臆造。
   - 人名用全称（带姓氏），如"孔子"、"孔端友"、"宋高宗"。
   - 朝代用精确朝代名（宋/明/清/元/唐/宋/汉/魏晋南北朝/隋/五代十国/辽/金/西夏……）。
   - 年代时间用具体年号+公元年（"建炎二年(1128)"）或单独年份（"1128"）。
   - 地点精确到文末明确写出的"市/县/区/省/景区名"。
   - 建筑/园林等称谓保留专有名（"佛香阁"、"祭孔大典"、"耕织图石刻"）。
3. 找不到的字段输出空数组或空字符串，不要乱填。
4. 关键词：从文本中选 5-8 个最能代表内容的名词/术语。
5. 主题 topic 从给定候选主题中选最接近的一个：景点、景点知识、历史人物、历史事件、礼仪制度、文献著作、机构家族、运营信息、其他。
6. 摘要：1-2 句话、不超过 80 字、事实准确。
7. 内容质量 self_score 给 1-5 整数：5=信息密度高/事实明确，1=几乎无信息。"""

EXTRACTION_USER_TEMPLATE = """[JSON 抽取任务]

请阅读以下文本 chunk，抽取为严格 JSON：

文本（genre={genre}, doc_title={doc_title}）：
\"\"\"
{chunk_text}
\"\"\"

输出 schema（键必须齐全）：
{{
  "summary": "<=80字 摘要",
  "keywords": ["...","..."],
  "topic": "景点|景点知识|历史人物|历史事件|礼仪制度|文献著作|机构家族|运营信息|其他",
  "spot_entities": [...],
  "person_entities": [...],
  "dynasty_entities": [...],
  "era_entities": [...],
  "building_entities": [...],
  "place_entities": [...],
  "event_entities": [...],
  "work_entities": [...],
  "org_entities": [...],
  "ritual_entities": [...],
  "identity_entities": [...],
  "culture_entities": [...],
  "exhibit_entities": [...],
  "self_score": 1
}}

只要 JSON，不要解释。"""


# ══════════════════════════════════════════════════════════════════
# Qdrant 客户端
# ══════════════════════════════════════════════════════════════════

def qdrant_get_all_points(limit=200):
    """Paginate scroll 拉所有点"""
    client = httpx.Client(base_url=f"http://{QDRANT_HOST}:{QDRANT_PORT}", timeout=60)
    points = []
    offset = None
    while True:
        body = {"limit": limit, "with_payload": True, "with_vector": False}
        if offset:
            body["offset"] = offset
        r = client.post(f"/collections/{COLLECTION}/points/scroll", json=body)
        r.raise_for_status()
        pts = r.json().get("result", {}).get("points", [])
        points.extend(pts)
        offset = r.json().get("result", {}).get("next_page_offset")
        if not offset:
            break
    return points


def qdrant_set_payload(point_id, payload_fields):
    client = httpx.Client(base_url=f"http://{QDRANT_HOST}:{QDRANT_PORT}", timeout=60)
    r = client.post(
        f"/collections/{COLLECTION}/points/payload",
        params={"wait": "false"},
        json={"payload": payload_fields, "points": [point_id]},
    )
    r.raise_for_status()
    return r.json()


# ══════════════════════════════════════════════════════════════════
# DashScope (OpenAI 兼容) 客户端
# ══════════════════════════════════════════════════════════════════

class RateLimiter:
    """简单令牌桶：qwen-plus 默认 RPM=60。控制每分钟调用次数。"""
    def __init__(self, max_per_min=40):
        self.max = max_per_min
        self.ts = []

    def wait(self):
        now = time.time()
        # 清理 60s 之外的
        self.ts = [t for t in self.ts if now - t < 60]
        if len(self.ts) >= self.max:
            sleep_for = 60 - (now - self.ts[0]) + 0.1
            print(f"  [限频] 等待 {sleep_for:.1f}s ...")
            time.sleep(sleep_for)
        self.ts.append(time.time())


def call_llm(chunk_text: str, genre: str, doc_title: str, max_retries=3):
    """调 DashScope qwen-plus 抽取。返回 dict / None（失败）"""
    headers = {
        "Authorization": f"Bearer {DS_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": DS_MODEL,
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": EXTRACTION_USER_TEMPLATE.format(
                genre=genre, doc_title=doc_title, chunk_text=chunk_text[:2200]
            )},
        ],
        "temperature": 0.1,
        "top_p": 0.9,
        "response_format": {"type": "json_object"},
    }
    url = f"{DS_BASE_URL}/chat/completions"
    for attempt in range(max_retries):
        try:
            r = httpx.post(url, headers=headers, json=body, timeout=DS_TIMEOUT)
            if r.status_code == 429 or r.status_code >= 500:
                wait = (attempt + 1) * 5
                print(f"  [LLM] 状态 {r.status_code}，sleep {wait}s 后重试")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, str):
                result = json.loads(content)
            else:
                result = content
            return result
        except httpx.HTTPError as e:
            print(f"  [LLM] 异常: {type(e).__name__} {e} (attempt {attempt+1})")
            time.sleep((attempt + 1) * 3)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [LLM] JSON 解码失败: {e}")
            return None
    return None


def _clean_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(s).strip() for s in x if s and str(s).strip() and str(s).strip() != "无"]
    if isinstance(x, str):
        s = x.strip()
        return [s] if s and s != "无" else []
    return []


def normalize_extraction(raw: dict) -> dict:
    """把 LLM 输出规范化、补齐字段"""
    out = {}
    out["summary"] = (raw.get("summary") or "").strip()[:200]
    out["keywords"] = _clean_list(raw.get("keywords"))[:12]
    topic = (raw.get("topic") or "其他").strip()
    valid_topics = {"景点", "景点知识", "历史人物", "历史事件", "礼仪制度",
                    "文献著作", "机构家族", "运营信息", "其他"}
    out["topic"] = topic if topic in valid_topics else "其他"

    for dim in DIM_LIST:
        out[dim] = _clean_list(raw.get(dim))[:30]

    try:
        score = float(raw.get("self_score", 3))
        out["quality_score"] = max(1.0, min(5.0, score))
    except (TypeError, ValueError):
        out["quality_score"] = 3.0

    return out


# ══════════════════════════════════════════════════════════════════
# 进度持久化
# ══════════════════════════════════════════════════════════════════

def load_progress():
    """返回 {id_int: payload_fields} 缓存；缺失/损坏返空字典"""
    cache = {}
    if not PROGRESS_FILE.exists():
        return cache
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cache[rec["id"]] = rec["payload"]
    except Exception as e:
        print(f"  [进度] 读取失败（忽略）：{e}")
    return cache


def append_progress(point_id, payload_fields):
    """追加一行到 jsonl"""
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "id": point_id,
            "payload": payload_fields,
            "ts": time.time(),
        }, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════
# 单条处理 + 并发
# ══════════════════════════════════════════════════════════════════

def process_one(point, write_to_qdrant=True, limiter=None):
    pid = point["id"]
    payload = point.get("payload") or {}
    chunk_text = (payload.get("chunk_text") or payload.get("chunk_text_full") or "").strip()
    if not chunk_text:
        return pid, None, "empty_text"

    if limiter:
        limiter.wait()

    raw = call_llm(
        chunk_text=chunk_text,
        genre=payload.get("genre", "其他"),
        doc_title=payload.get("doc_title", "?"),
    )
    if raw is None:
        return pid, None, "llm_failed"

    normed = normalize_extraction(raw)

    if write_to_qdrant:
        try:
            qdrant_set_payload(pid, normed)
        except Exception as e:
            print(f"  [Qdrant] 写 id={pid} 失败：{e}")
            return pid, normed, "qdrant_failed"

    append_progress(pid, normed)
    return pid, normed, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="限制处理条数（0 = 全量）")
    parser.add_argument("--workers", type=int, default=4, help="并发数（默认 4）")
    parser.add_argument("--dry-run", action="store_true", help="不写 Qdrant，只跑 LLM 并缓存")
    parser.add_argument("--resume", action="store_true", help="跳过已缓存的 id")
    args = parser.parse_args()

    print(f"\n{'='*60}\n  unified_corpus 富化（维度抽取 + 打标签）\n  Collection: {COLLECTION}\n  模型: {DS_MODEL}\n{'='*60}\n")

    print("[1/4] 拉取所有 points ...")
    all_points = qdrant_get_all_points()
    print(f"      共 {len(all_points)} 条")

    progress = load_progress()
    print(f"[2/4] 已缓存 {len(progress)} 条")

    targets = [p for p in all_points if not (args.resume and p["id"] in progress)]
    if args.limit > 0:
        targets = targets[: args.limit]
    print(f"[3/4] 待处理 {len(targets)} 条（resume={args.resume}, limit={args.limit}, dry_run={args.dry_run}）")

    if not targets:
        print("\n[DONE] 无需处理。")
        return

    limiter = RateLimiter(max_per_min=35)
    ok_count = fail_count = 0

    print(f"[4/4] 开始处理（并发={args.workers}）...\n")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, p, write_to_qdrant=not args.dry_run, limiter=limiter): p for p in targets}
        for i, fut in enumerate(as_completed(futures), 1):
            pid, normed, status = fut.result()
            if status == "ok":
                ok_count += 1
            else:
                fail_count += 1
            if i % 20 == 0 or i == len(targets):
                print(f"    进度 {i}/{len(targets)}  ok={ok_count} fail={fail_count}")

    print(f"\n完成。 ok={ok_count}  fail={fail_count}  缓存文件={PROGRESS_FILE}")

    # 统计摘要
    summary = {
        "collection": COLLECTION,
        "model": DS_MODEL,
        "total_points": len(all_points),
        "processed": ok_count,
        "failed": fail_count,
        "fields_written": [
            "summary", "keywords", "topic", "quality_score",
            "spot_entities", "person_entities", "dynasty_entities", "era_entities",
            "building_entities", "place_entities", "event_entities", "work_entities",
            "org_entities", "ritual_entities", "identity_entities",
            "culture_entities", "exhibit_entities",
        ],
        "cache_file": str(PROGRESS_FILE),
        "dry_run": args.dry_run,
        "ts": time.time(),
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"摘要写入: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
