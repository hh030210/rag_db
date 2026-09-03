"""
test_improvements.py
====================

单元测试：对每个增强模块做最小化 sanity check。

用法：
  python test_improvements.py

不依赖 GPU / 大模型，只验证逻辑正确性。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# 添加工作目录到 path
sys.path.insert(0, str(Path(__file__).parent))

from chunking_enhancements import (
    AdaptiveSplitter,
    LengthAwareDenoiser,
    JiebaFingerprintDedup,
    EmbeddingIGCalculator,
    BatchedLM,
    build_enhancements,
)


def test_adaptive_splitter():
    print("\n=== Test 1: AdaptiveSplitter ===")
    splitter = AdaptiveSplitter(l_min=300, l_max=600, target_ratio=0.75)

    # 阶段 A: 累积 100 字符（< l_min=300），应禁止切分
    for i in range(5):
        splitter.add(20)
        result = splitter.should_split(100, ppl=200.0, base_threshold=100.0)
        assert not result, f"l_min 保护失败 at chunk_len=100"
    print(f"  [PASS] l_min 保护：累积 100 字符时不切分")

    # 阶段 B: 累积到 500 字符，ppl 高于阈值，应切分
    splitter2 = AdaptiveSplitter(l_min=300, l_max=600)
    for i in range(25):
        splitter2.add(20)  # 累积 500
    result = splitter2.should_split(500, ppl=200.0, base_threshold=100.0)
    assert result, f"应该切分但没切: chunk_len=500, ppl=200, thresh=100"
    print(f"  [PASS] 超过 target 长度后阈值降低，PPL 高时正确切分")

    # 阶段 C: l_max 强制切分
    splitter3 = AdaptiveSplitter(l_min=300, l_max=600)
    for i in range(31):
        splitter3.add(20)  # 累积 620
    result = splitter3.should_split(620, ppl=10.0, base_threshold=100.0)
    assert result, f"l_max 强制切分失败"
    print(f"  [PASS] l_max 强制切分：超过 600 字符无条件切分")


def test_length_aware_denoiser():
    print("\n=== Test 2: LengthAwareDenoiser ===")
    denoiser = LengthAwareDenoiser(l_min=300)

    # 模拟 5 句：4 句正常，1 句 PPL 异常高（短句）
    sents = [
        "新华社北京8月11日电，",                # 短，含数字时间戳
        "据中央气象台预计，11日至13日，",
        "东北及京津冀地区将有强降雨天气，",
        "松辽和海河流域汛情将发展。",
        "受台风卡努外围云系影响，" * 30,         # 长句
    ]
    ppls = [500.0, 480.0, 50.0, 55.0, 60.0]  # 前两句 PPL 异常高

    kept_sents, kept_ppls = denoiser.denoise(sents, ppls)

    # 前两句 PPL > mean+3sigma 但长度 < l_min，应该被保留
    assert kept_sents == sents, f"短句应被保留，实际删除 {len(sents) - len(kept_sents)} 句"
    print(f"  [PASS] 短句保护：PPL 异常但长度 < l_min 的短新闻被保留 ({len(kept_sents)}/{len(sents)})")


def test_jieba_dedup():
    print("\n=== Test 3: JiebaFingerprintDedup ===")
    dedup = JiebaFingerprintDedup(n=4)

    if not dedup.available:
        print(f"  [SKIP] jieba 未安装")
        return

    chunks = [
        {"chunk_text": "今天美联储宣布加息25个基点以应对通胀压力，市场反应积极，今天美联储宣布加息25个基点以应对通胀压力，市场反应积极。", "chunk_len": 50},
        {"chunk_text": "今天美联储宣布加息25个基点以应对通胀压力，市场反应积极，今天美联储宣布加息25个基点以应对通胀压力，市场反应积极。", "chunk_len": 50},  # 重复
        {"chunk_text": "今天美联储宣布加息25个基点以应对通胀压力，市场反应积极。" * 5, "chunk_len": 250},  # 同 fp，更长
        {"chunk_text": "据中央气象台预计近期将出现大范围强降雨天气过程，请各部门做好防汛准备。", "chunk_len": 30},
        {"chunk_text": "短文本", "chunk_len": 4},  # 短，无 fingerprint
    ]

    deduped = dedup.dedup(chunks)
    print(f"  原始: {len(chunks)} chunks")
    print(f"  去重后: {len(deduped)} chunks")

    # 期望：第 1、2、3 合并为 1（保留最长的 #3），第 4 单独，第 5 单独
    assert len(deduped) == 3, f"期望 3 个 unique chunks，实际 {len(deduped)}"
    kept_texts = [c["chunk_text"] for c in deduped]
    longest = max(deduped, key=lambda c: c["chunk_len"])
    assert longest["chunk_len"] == 250, f"应保留最长的 250 字符 chunk，实际保留 {longest['chunk_len']}"
    print(f"  [PASS] 去重成功，保留最长版本 (chunk_len={longest['chunk_len']})")


def test_embedding_ig_skipped():
    print("\n=== Test 4: EmbeddingIGCalculator (可选) ===")
    import os
    # 如果环境变量没设置，假定离线/无网络，直接跳过
    if os.environ.get("RUN_NETWORK_TESTS") != "1":
        print(f"  [SKIP] 未设置 RUN_NETWORK_TESTS=1，跳过网络依赖测试")
        return
    ig = EmbeddingIGCalculator()
    if not ig.available:
        print(f"  [SKIP] sentence-transformers 未安装或加载失败")
        return
    sim_high = ig.ig("美联储宣布加息 25 个基点", "美联储再次宣布加息 25 个基点")
    sim_low = ig.ig("美联储宣布加息 25 个基点", "今天天气晴朗适合出行")
    print(f"  相近文本 IG: {sim_high:.4f}")
    print(f"  不相关文本 IG: {sim_low:.4f}")
    assert sim_high > sim_low, f"语义相似度判断错误: {sim_high} <= {sim_low}"
    print(f"  [PASS] Embedding IG 区分能力正常")


def test_batched_lm_skipped():
    print("\n=== Test 5: BatchedLM (可选) ===")
    import os
    if os.environ.get("RUN_NETWORK_TESTS") != "1":
        print(f"  [SKIP] 未设置 RUN_NETWORK_TESTS=1，跳过网络依赖测试")
        return
    try:
        scorer = BatchedLM("Qwen/Qwen2.5-0.5B-Instruct", batch_size=4)
        if not scorer.available:
            print(f"  [SKIP] 模型加载失败（可能未安装 transformers/torch）")
            return
        contexts = ["", "今天天气", "美联储宣布加息"]
        sentences = ["你好世界", "晴朗", "25 个基点"]
        ppls = scorer.score_batch(contexts, sentences)
        print(f"  推理 PPL: {ppls}")
        assert all(p > 0 for p in ppls), "PPL 应为正数"
        print(f"  [PASS] BatchedLM 批处理推理正常")
    except Exception as e:
        print(f"  [SKIP] {e}")


def test_end_to_end_smoke():
    print("\n=== Test 6: 端到端 smoke test (EnhancedChunker) ===")
    import tempfile
    from enhanced_chunker import EnhancedChunker, EnhancedConfig

    # 构造测试输入
    test_text = (
        "新华社北京8月11日电（记者黄垚）据中央气象台预计，11日至13日，"
        "东北及京津冀地区将有强降雨天气，松辽和海河流域汛情将发展。"
        "受台风卡努外围云系影响，10日白天至11日上午，吉林和黑龙江部分地区出现大到暴雨，"
        "黑龙江佳木斯、七台河、牡丹江和吉林延边局地现大暴雨。"
        "辽东半岛、山东半岛东部、吉林东部出现6至8级阵风。"
    ) * 3  # 重复 3 次以触发体裁检测

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(test_text)
        tmp_path = f.name

    with tempfile.TemporaryDirectory() as out_dir:
        # 模拟 args
        class Args:
            window_w = 3
            beta_small = 0.8
            beta = 1.1
            denoise = True
            line_mode = True
            llm_api_key = ""
            llm_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            llm_model = "qwen3-8b"
            llm_sample_interval = 0
            ppl_model_name = ""

        config = EnhancedConfig(use_dedup=True, use_embedding_ig=False)
        chunker = EnhancedChunker(Args(), config)
        t0 = time.time()
        chunker.run(Path(tmp_path), Path(out_dir))
        elapsed = time.time() - t0

        out_json = Path(out_dir) / "all_chunks_chunks.json"
        chunks = json_load(out_json)
        assert len(chunks) > 0, "应至少切出 1 个 chunk"
        print(f"  [PASS] 端到端运行成功: {len(chunks)} chunks, {elapsed:.2f}s")
        print(f"  第一个 chunk 前 60 字: {chunks[0]['chunk_text'][:60]}")


def json_load(path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    print("=" * 60)
    print("Chunking 增强模块单元测试")
    print("=" * 60)
    test_adaptive_splitter()
    test_length_aware_denoiser()
    test_jieba_dedup()
    test_embedding_ig_skipped()
    test_batched_lm_skipped()
    test_end_to_end_smoke()
    print("\n" + "=" * 60)
    print("所有测试完成")


if __name__ == "__main__":
    main()
