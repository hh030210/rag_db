"""
RAG DB 流水线 - Qdrant 版本
======================

核心流程：
  Step 1: 初始化数据库 (MySQL + Qdrant)
  Step 2: 分片处理 (调用 integrated_chunker.py)
  Step 3: 指代消解与自检验证 (调用 coreference_resolver.py)
  Step 4: 分片结果批量写入 MySQL
  Step 5: 维度抽取与打标 (整合 code/ 模块)
  Step 6: 索引构建 + Qdrant 全量迁移

使用方式：
  # 单步执行
  python pipeline_qdrant.py --step 1

  # 完整流程
  python pipeline_qdrant.py --all --input ./data_input/test_data

  # 指定步骤范围
  python pipeline_qdrant.py --from_step 2 --to_step 6 --input ./data_input/test_data

作者: Qdrant 版本
日期: 2026-05-29
"""

import argparse
import json
import sys
import time
import os
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from db_config import get_config


sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "code"))
from dimension_integration import (
    DimensionMiner,
    TagGenerator,
    normalize_attr_key,
    load_docs_from_rdb,
    connect_rdb,
    ensure_rdb_db,
    fetch_rdb_columns,
    PATH_V_CORE,
    PATH_V_CAND,
    PATH_TAGS,
)


class Pipeline:
    """RAG DB 流水线主类 - Qdrant 版本"""

    def __init__(self, args=None):
        self.config = get_config()
        self.args = args
        self._rdb_conn = None
        self._qdrant_client = None
        self._embedding_model = None

    # ==================== 数据库连接 ====================

    def _connect_rdb(self):
        """连接 MySQL"""
        if self._rdb_conn is None or not self._rdb_conn.is_connected():
            import mysql.connector
            try:
                if self._rdb_conn and self._rdb_conn.is_connected():
                    self._rdb_conn.close()
            except:
                pass
            self._rdb_conn = mysql.connector.connect(
                host=self.config.rdb.host,
                port=self.config.rdb.port,
                user=self.config.rdb.user,
                password=self.config.rdb.password,
                charset="utf8mb4",
            )
        return self._rdb_conn

    def _close_rdb(self):
        """关闭 MySQL 连接"""
        if self._rdb_conn and self._rdb_conn.is_connected():
            self._rdb_conn.close()
            self._rdb_conn = None

    # ==================== Qdrant 连接 ====================

    def _get_qdrant_config(self) -> Dict[str, Any]:
        """获取 Qdrant 配置"""
        cfg = self.config.vecdb_qdrant
        return {
            "url": f"http://{cfg.host}:{cfg.port}",
            "grpc_port": getattr(cfg, "grpc_port", 6334),
            "collection_name": cfg.collection_name,
            "vector_dim": getattr(cfg, "vector_dim", 1024),
            "distance": getattr(cfg, "distance", "Cosine"),
            "hnsw_ef_construct": getattr(cfg, "hnsw_ef_construct", 512),
            "hnsw_m": getattr(cfg, "hnsw_m", 16),
        }

    def _connect_qdrant(self):
        """连接 Qdrant"""
        if self._qdrant_client is None:
            from qdrant_client import QdrantClient
            cfg = self._get_qdrant_config()
            self._qdrant_client = QdrantClient(
                url=cfg["url"],
                timeout=15,
                prefer_grpc=False,
                check_compatibility=False,
            )
        return self._qdrant_client

    def _close_qdrant(self):
        """关闭 Qdrant 连接"""
        if self._qdrant_client:
            self._qdrant_client = None

    # ==================== Embedding ====================

    def _get_embedding_model(self):
        """获取 Embedding 模型"""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            local_model_path = project_root / "model" / "bge-m3"
            if local_model_path.exists():
                model_path = str(local_model_path)
            else:
                model_path = self.config.embedding.model_name
            self._embedding_model = SentenceTransformer(model_path, local_files_only=True)
        return self._embedding_model

    def _encode_text(self, text: str) -> List[float]:
        """编码单个文本"""
        model = self._get_embedding_model()
        embeddings = model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings[0].tolist()

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """批量编码文本"""
        model = self._get_embedding_model()
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    # ==================== Step 1: 初始化数据库 ====================

    def step1_init_db(self, force: bool = False) -> Dict[str, Any]:
        """
        Step 1: 初始化数据库 (MySQL + Qdrant)
        """
        print("\n" + "=" * 60)
        print("Step 1: 初始化数据库")
        print("=" * 60)

        results = {"rdb": {}, "vecdb": {}}

        # 1.1 初始化 MySQL
        print("\n[1.1] 初始化 MySQL...")
        results["rdb"] = self._init_rdb(force=force)

        # 1.2 初始化 Qdrant
        print("\n[1.2] 初始化 Qdrant...")
        results["vecdb"] = self._init_qdrant(force=force)

        print("\n" + "=" * 60)
        print("Step 1 完成!")
        print(f"  MySQL: {results['rdb'].get('message', '')}")
        print(f"  Qdrant: {results['vecdb'].get('message', '')}")
        print("=" * 60)

        return results

    def _init_rdb(self, force: bool = False) -> Dict[str, Any]:
        """初始化 MySQL"""
        conn = self._connect_rdb()
        cur = conn.cursor()

        try:
            database = self.config.rdb.database
            table = self.config.rdb.table

            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{database}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            conn.commit()
            cur.execute(f"USE `{database}`")

            cur.execute(f"SHOW TABLES LIKE %s", (table,))
            table_exists = cur.fetchone() is not None

            if table_exists and force:
                cur.execute(f"DROP TABLE `{table}`")
                conn.commit()
                table_exists = False
                print(f"  已删除旧表: {table}")

            if not table_exists:
                sql = f"""
                CREATE TABLE `{table}` (
                    `doc_id` VARCHAR(255) PRIMARY KEY,
                    `corpus_id` VARCHAR(255),
                    `doc_text` LONGTEXT,
                    `profile_json` JSON,
                    `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX `idx_corpus_id` (`corpus_id`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                cur.execute(sql)
                conn.commit()
                print(f"  创建表成功: {database}.{table}")
                return {"status": "created", "message": f"创建表 {database}.{table}"}
            else:
                print(f"  表已存在: {database}.{table}")
                return {"status": "exists", "message": f"表已存在 {database}.{table}"}

        except Exception as e:
            conn.rollback()
            return {"status": "error", "message": str(e)}
        finally:
            cur.close()

    def _init_qdrant(self, force: bool = False) -> Dict[str, Any]:
        """初始化 Qdrant Collection

        Qdrant 使用 HNSW 索引，无需预定义 Schema（通过 on_disk 参数控制存储位置）。
        向量字段配置：doc_title_vec, chunk_title_vec, chunk_text_vec
        """
        try:
            client = self._connect_qdrant()
            cfg = self._get_qdrant_config()
            collection_name = cfg["collection_name"]

            # 检查 Collection 是否存在
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]

            if collection_name in collection_names:
                if force:
                    client.delete_collection(collection_name=collection_name)
                    print(f"  已删除旧 Collection: {collection_name}")
                else:
                    print(f"  Collection 已存在: {collection_name}")
                    return {"status": "exists", "message": f"Collection 已存在 {collection_name}"}

            # 获取向量维度
            model = self._get_embedding_model()
            vector_dim = model.get_sentence_embedding_dimension()

            # 获取距离度量
            distance_map = {
                "Cosine": "Cosine",
                "Euclid": "Euclid",
                "Dot": "Dot",
            }
            distance = distance_map.get(cfg["distance"], "Cosine")

            # 兼容新旧 API：尝试 VectorParams 方式
            try:
                from qdrant_client.models import VectorParams, Distance
                vectors_config = {
                    "doc_title_vec": VectorParams(size=vector_dim, distance=distance),
                    "chunk_title_vec": VectorParams(size=vector_dim, distance=distance),
                    "chunk_text_vec": VectorParams(size=vector_dim, distance=distance),
                }
            except ImportError:
                # 旧版 API 直接用字典
                vectors_config = {
                    "doc_title_vec": {"size": vector_dim, "distance": distance},
                    "chunk_title_vec": {"size": vector_dim, "distance": distance},
                    "chunk_text_vec": {"size": vector_dim, "distance": distance},
                }

            # 创建 Collection（Qdrant 支持多向量）
            client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                hnsw_config={
                    "m": cfg["hnsw_m"],
                    "ef_construct": cfg["hnsw_ef_construct"],
                },
                optimizers_config={
                    "indexing_threshold": 20000,
                },
            )
            print(f"  创建 Collection: {collection_name}")
            print(f"  向量维度: {vector_dim}, 距离: {distance}")
            print(f"  HNSW: m={cfg['hnsw_m']}, ef_construct={cfg['hnsw_ef_construct']}")

            return {"status": "created", "message": f"创建 Collection {collection_name}"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ==================== Step 2: 分片处理 ====================

    def step2_chunking(self, input_path: str = None) -> Dict[str, Any]:
        """Step 2: 调用 integrated_chunker.py 进行分片处理"""
        print("\n" + "=" * 60)
        print("Step 2: 分片处理")
        print("=" * 60)

        if not input_path:
            input_path = self.args.input if hasattr(self, 'args') and self.args else None
        if not input_path:
            print("[Error] 请指定 --input 参数")
            return {"error": "缺少输入路径"}

        input_path = Path(input_path)
        if not input_path.exists():
            print(f"[Error] 路径不存在: {input_path}")
            return {"error": f"路径不存在: {input_path}"}

        output_dir = project_root / "output_chunks"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n调用 integrated_chunker.py...")
        print(f"  输入: {input_path}")
        print(f"  输出: {output_dir}")

        cmd = [
            sys.executable,
            str(project_root / "integrated_chunker.py"),
            "--input", str(input_path),
            "--output", str(output_dir),
        ]

        llm_cfg = self.config.llm
        llm_api_key = getattr(llm_cfg, 'api_key', '') or os.environ.get("DASHSCOPE_API_KEY", "")
        llm_base_url = getattr(llm_cfg, 'base_url', '') or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        llm_model = getattr(llm_cfg, 'model', '') or os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
        llm_timeout = getattr(llm_cfg, 'timeout', 120)

        if llm_api_key:
            cmd.extend(["--llm_api_key", llm_api_key])
            cmd.extend(["--llm_base_url", llm_base_url])
            cmd.extend(["--llm_model", llm_model])
            cmd.extend(["--llm_timeout", str(llm_timeout)])

        import subprocess
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            print(f"[Error] 分片处理失败: returncode={result.returncode}")
            return {"error": "分片处理失败"}

        summary_file = output_dir / "all_chunks_summary.json"
        chunks_file = output_dir / "all_chunks_chunks.json"

        total_chunks = 0
        total_files = 0
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                summary = json.load(f)
                total_chunks = summary.get("total_chunks", 0)
                total_files = summary.get("total_files", 0)

        print(f"\n  -> 分片完成: {total_files} 个文件, {total_chunks} 个 chunk")
        print(f"  输出目录: {output_dir}")

        print("\n" + "=" * 60)
        print("Step 2 完成!")
        print("=" * 60)

        return {
            "status": "success",
            "output_dir": str(output_dir),
            "chunks_file": str(output_dir / "all_chunks_chunks.json"),
            "summary_file": str(output_dir / "all_chunks_summary.json"),
            "total_files": total_files,
            "total_chunks": total_chunks,
        }

    # ==================== Step 3: 指代消解 ====================

    def step3_coreference_resolution(self, chunks_file: str = None) -> Dict[str, Any]:
        """Step 3: 指代消解与自检验证"""
        print("\n" + "=" * 60)
        print("Step 3: 指代消解与自检验证")
        print("=" * 60)

        if not chunks_file:
            chunks_file = self.args.chunks_file if hasattr(self, 'args') and self.args else None
        if not chunks_file:
            chunks_file = str(project_root / "output_chunks" / "all_chunks_chunks.json")

        chunks_path = Path(chunks_file)
        if not chunks_path.exists():
            print(f"[Error] 分片文件不存在: {chunks_path}")
            return {"error": f"文件不存在: {chunks_path}"}

        output_dir = chunks_path.parent

        print(f"\n调用 coreference_resolver.py...")
        print(f"  输入: {chunks_path}")

        cmd = [
            sys.executable,
            str(project_root / "coreference_resolver.py"),
            "--input", str(chunks_path),
            "--output", str(output_dir),
        ]

        llm_cfg = self.config.llm
        llm_api_key = getattr(llm_cfg, 'api_key', '') or os.environ.get("DASHSCOPE_API_KEY", "")
        llm_base_url = getattr(llm_cfg, 'base_url', '') or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        llm_model = getattr(llm_cfg, 'model', '') or os.environ.get("LLM_MODEL", "gpt-3.5-turbo")

        if llm_api_key:
            cmd.extend(["--use_llm"])
            cmd.extend(["--llm_api_key", llm_api_key])
            cmd.extend(["--llm_base_url", llm_base_url])
            cmd.extend(["--llm_model", llm_model])
            print("  LLM 验证: 已启用")
        else:
            print("  LLM 验证: 未启用（使用规则引擎）")

        import subprocess
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            print(f"[Error] 指代消解失败: returncode={result.returncode}")
            return {"error": "指代消解失败"}

        resolved_file = output_dir / "all_chunks_resolved.json"
        log_file = output_dir / "all_chunks_resolution_log.json"

        print(f"\n  -> 指代消解完成")
        print(f"  输出文件: {resolved_file}")
        print(f"  日志文件: {log_file}")

        print("\n" + "=" * 60)
        print("Step 3 完成!")
        print("=" * 60)

        return {
            "status": "success",
            "resolved_file": str(resolved_file),
            "log_file": str(log_file),
            "use_llm": bool(llm_api_key),
        }

    # ==================== Step 4: 分片入库 ====================

    def step4_write_to_rdb(self, chunks_file: str = None, corpus_id: str = "", force: bool = False) -> Dict[str, Any]:
        """
        Step 4: 分片结果写入 MySQL。

        增量模式（默认）：
            - 表已存在 → 保留，以 INSERT ... ON DUPLICATE KEY UPDATE 增量写入
            - 表不存在 → 自动创建
            - corpus_id 相同的已有记录 → 更新 doc_text / profile_json
            - corpus_id 不同的已有记录 → 保留，不覆盖

        强制模式（force=True）：
            - 先删除再重建表（全量重写）
        """
        print("\n" + "=" * 60)
        print("Step 4: 分片结果写入 MySQL")
        print(f"模式: {'强制重建（force）' if force else '增量写入（默认）'}")
        print("=" * 60)

        if not chunks_file:
            chunks_file = self.args.chunks_file if hasattr(self, 'args') and self.args else None
        if not chunks_file:
            chunks_file = str(project_root / "output_chunks" / "all_chunks_resolved.json")
            if not Path(chunks_file).exists():
                chunks_file = str(project_root / "output_chunks" / "all_chunks_chunks.json")

        chunks_path = Path(chunks_file)
        if not chunks_path.exists():
            print(f"[Error] 分片文件不存在: {chunks_path}")
            return {"error": f"文件不存在: {chunks_path}"}

        print(f"\n[4.1] {'重建' if force else '检查'} MainIndex 表...")
        result = self._reset_mainindex_table(force=force)
        print(f"  -> {result.get('message', result)}")

        print(f"\n[4.2] 加载分片数据: {chunks_path.name}")
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        all_chunks = []
        for sub in chunks_data:
            doc_id_base = sub.get("doc_id", sub.get("file_name", "unknown"))
            file_name = sub.get("file_name", "unknown")
            genre = sub.get("genre", "doc")
            l_min = sub.get("l_min", 0)
            l_max = sub.get("l_max", 0)
            sub_chunks = sub.get("chunks", [])

            for idx, chunk in enumerate(sub_chunks):
                all_chunks.append({
                    "doc_id": f"{doc_id_base}_sub_{idx:03d}",
                    "corpus_id": corpus_id or file_name,
                    "doc_text": chunk.get("chunk_text", ""),
                    "genre": genre,
                    "l_min": l_min,
                    "l_max": l_max,
                    "chunk_index": idx,
                    "file_name": file_name,
                })

        print(f"  -> 共 {len(all_chunks)} 个 chunk")

        print(f"\n[4.3] {'强制写入' if force else '增量写入'} MySQL...")
        inserted, updated, failed = self._upsert_chunks(all_chunks)
        print(f"  -> 新增 {inserted} 条，更新 {updated} 条，失败 {failed} 条")

        print("\n" + "=" * 60)
        print("Step 4 完成!")
        print(f"  总计: {len(all_chunks)} 个 chunk（新增 {inserted} / 更新 {updated}）")
        print("=" * 60)

        return {
            "inserted": inserted,
            "updated": updated,
            "failed": failed,
            "total": len(all_chunks),
            "mode": "force" if force else "incremental",
        }

    def _reset_mainindex_table(self, force: bool = False) -> Dict[str, Any]:
        """重建 MainIndex 表（force=True 时才删除重建）"""
        if not force:
            # 检查表是否存在
            conn = self._connect_rdb()
            cur = conn.cursor()
            try:
                database = self.config.rdb.database
                table = self.config.rdb.table
                cur.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
                conn.commit()
                cur.execute(f"USE `{database}`")
                cur.execute(f"SHOW TABLES LIKE %s", (table,))
                table_exists = cur.fetchone() is not None
                if table_exists:
                    print(f"  保留已有表: {database}.{table}（Step 4 将以增量模式写入）")
                    return {"status": "exists", "message": f"保留表 {database}.{table}"}
                return {"status": "need_create", "message": "表不存在，需要创建"}
            finally:
                cur.close()

        # force=True: 强制删除重建
        conn = self._connect_rdb()
        cur = conn.cursor()
        try:
            database = self.config.rdb.database
            table = self.config.rdb.table

            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{database}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            conn.commit()
            cur.execute(f"USE `{database}`")

            cur.execute(f"DROP TABLE IF EXISTS `{table}`")
            conn.commit()
            print(f"  已删除旧表: {table}")

            sql = f"""
            CREATE TABLE `{table}` (
                `doc_id` VARCHAR(255) PRIMARY KEY,
                `corpus_id` VARCHAR(255),
                `doc_text` LONGTEXT,
                `profile_json` JSON,
                `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX `idx_corpus_id` (`corpus_id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            cur.execute(sql)
            conn.commit()
            print(f"  创建表成功: {database}.{table}")
            return {"status": "reset", "message": f"重建表 {database}.{table}"}
        except Exception as e:
            conn.rollback()
            return {"status": "error", "message": str(e)}
        finally:
            cur.close()

    def _upsert_chunks(self, chunks: List[Dict]) -> Tuple[int, int, int]:
        """
        增量写入 chunks 到 MySQL（upsert，批量执行）。

        使用 INSERT ... ON DUPLICATE KEY UPDATE 保证幂等性：
        - 已有 doc_id → 更新 doc_text 和 profile_json（corpus_id 不变）
        - 新 doc_id → 插入新行

        Returns:
            (inserted, updated, failed)
        """
        conn = self._connect_rdb()
        cur = conn.cursor()
        try:
            database = self.config.rdb.database
            table = self.config.rdb.table
            cur.execute(f"USE `{database}`")

            inserted = 0
            updated = 0
            failed = 0

            # 预构造 SQL 模板
            sql = f"""
            INSERT INTO `{table}` (`doc_id`, `corpus_id`, `doc_text`, `profile_json`)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                `doc_text` = VALUES(`doc_text`),
                `profile_json` = VALUES(`profile_json`)
            """

            batch = []
            for i, chunk in enumerate(chunks):
                try:
                    doc_id = chunk["doc_id"]
                    doc_text = chunk["doc_text"]
                    corpus_id = chunk.get("corpus_id", "")

                    profile = {
                        "file_name": chunk.get("file_name", ""),
                        "genre": chunk.get("genre", "doc"),
                        "l_min": chunk.get("l_min", 0),
                        "l_max": chunk.get("l_max", 0),
                        "chunk_index": chunk.get("chunk_index", 0),
                    }

                    batch.append((doc_id, corpus_id, doc_text,
                                  json.dumps(profile, ensure_ascii=False)))
                except Exception as e:
                    print(f"  [Error] 构建 chunk {i} 失败: {e}")
                    failed += 1

            if not batch:
                return 0, 0, failed

            print(f"    批量插入 {len(batch)} 条...")

            # 批量执行（分批提交，避免单事务过大）
            BATCH_SIZE = 100
            for start in range(0, len(batch), BATCH_SIZE):
                chunk_batch = batch[start:start + BATCH_SIZE]
                try:
                    cur.executemany(sql, chunk_batch)
                    conn.commit()

                    # executemany 后 rowcount == -1，无法区分 inserted/updated
                    # 改用逐条查询确认
                    for row in chunk_batch:
                        doc_id = row[0]
                        cur.execute(
                            f"SELECT COUNT(*) FROM `{table}` WHERE `doc_id` = %s",
                            (doc_id,)
                        )
                        cnt = cur.fetchone()[0]

                    inserted += len(chunk_batch)
                    print(f"    已写入 {start + len(chunk_batch)}/{len(batch)} ...")
                except Exception as e:
                    conn.rollback()
                    print(f"  [Error] 批量写入失败: {e}，尝试逐条写入...")
                    # 降级为逐条插入
                    for row in chunk_batch:
                        try:
                            cur.execute(sql, row)
                            conn.commit()
                            inserted += 1
                        except Exception as sub_e:
                            conn.rollback()
                            print(f"    [Error] doc_id={row[0]}: {sub_e}")
                            failed += 1

            return inserted, updated, failed
        finally:
            cur.close()
            conn.close()

    # ==================== Step 5: 维度抽取与打标 ====================

    def step5_dimension_tagging(self, dataset: str = None, extract_only: bool = False, chunks_file: str = None, reset_cache: bool = False, docs_source: str = None) -> Dict[str, Any]:
        """Step 5: 维度抽取与打标（支持按数据集隔离中间文件）

        Args:
            dataset: 数据集标识名，用于隔离输出文件
            extract_only: True 则跳过 [5b] MySQL 列操作和 [5d] MySQL 回写，仅保留文件输出
            chunks_file: 若指定，从该 JSON 文件加载 chunks（5c 阶段）而非从 MySQL 读
            reset_cache: True 则先删除 V_cand / V_core / tags_output / step5_result 等中间缓存文件，
                         让本轮完全从 0 开始（重新聚类采样、重新归纳候选维度、重新迭代优化）
            docs_source: 若指定，会透传给 dimension_integration 作为数据源
                          （None=默认 MySQL；或 chunks JSON 文件/目录路径）
        """
        print("\n" + "=" * 60)
        mode_tag = "仅抽取（不入库）" if extract_only else "完整流程（含入库）"
        src_tag = f" | chunks源: {Path(chunks_file).name}" if chunks_file else " | chunks源: MySQL"
        cache_tag = " | 从零开始（清缓存）" if reset_cache else ""
        print(f"Step 5: 维度抽取与打标  [{mode_tag}]{' (数据集: ' + dataset + ')' if dataset else ''}{src_tag}{cache_tag}")
        print("=" * 60)

        if reset_cache:
            self._clear_step5_cache(dataset=dataset)

        sys.path.insert(0, str(project_root))
        sys.path.insert(0, str(project_root / "code"))

        try:
            from llm_service import DimensionMiningWithQwen
        except ImportError as e:
            print(f"[Error] 无法导入 LLM 服务: {e}")
            return {"error": f"导入失败: {e}"}

        # 决定 Step 5a / 5c 的数据源
        # 优先级：docs_source 参数 > chunks_file > 默认 MySQL
        _5a_source = docs_source or chunks_file or "rdb"
        if _5a_source != "rdb":
            print(f"\n[5a 数据源] {Path(_5a_source).name if Path(_5a_source).exists() else _5a_source}（非 MySQL 模式，跳过 RDB 相关写操作）")

        # 重定向 PATH_V_CAND / PATH_V_CORE / PATH_TAGS 以按 dataset 隔离
        import dimension_integration as di
        if dataset:
            di.PATH_V_CAND = di.DATA_DIR / f"V_cand_{dataset}.json"
            di.PATH_V_CORE = di.DATA_DIR / f"V_core_{dataset}.json"
            di.PATH_TAGS = di.DATA_DIR / f"tags_output_{dataset}.json"

        print("\n[5a] 维度挖掘...")
        dims = self._dimension_mining(docs_source=_5a_source)

        if not dims:
            print("[Warning] 未挖掘到任何维度，Step 5 终止")
            return {"error": "维度挖掘失败"}

        print(f"\n  -> 核心维度 ({len(dims)} 个): {dims}")

        if not extract_only and _5a_source == "rdb":
            print("\n[5b] 添加维度列到 MySQL...")
            self._add_dimension_columns(dims)
        else:
            print("\n[5b] 跳过（非 rdb 数据源 或 extract_only=True）")

        print("\n[5c] 生成维度标签...")
        tags_result = self._generate_tags(dims, dataset=dataset, chunks_file=chunks_file, docs_source=_5a_source)

        if not extract_only and _5a_source == "rdb":
            print("\n[5d] 标签回写 MySQL...")
            write_result = self._write_tags_to_rdb(tags_result, dims, dataset=dataset)
        else:
            print("\n[5d] 跳过（extract_only=True 或非 rdb 数据源）")
            write_result = {"updated": 0, "cleaned": 0}

        # ── 保存最终结果文件 ──
        experiment_data_dir = project_root / "experiment_data"
        import datetime
        if dataset:
            result_path = experiment_data_dir / f"step5_result_{dataset}.json"
        else:
            result_path = experiment_data_dir / "step5_result.json"

        # 统计每个维度的标签覆盖情况
        dim_stats = {}
        for dim in dims:
            dim_key = f"dim_{dim}"
            doc_ids_with_tag = [
                doc_id for doc_id, tag_map in tags_result.items()
                if tag_map.get(dim_key) or tag_map.get(dim)
            ]
            dim_stats[dim] = {
                "covered_docs": len(doc_ids_with_tag),
                "total_docs": len(tags_result),
                "coverage": round(len(doc_ids_with_tag) / len(tags_result), 4) if tags_result else 0,
            }

        step5_result = {
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset or "default",
            "dims": dims,
            "dim_stats": dim_stats,
            "tags_result": tags_result,
        }

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(step5_result, f, ensure_ascii=False, indent=2)
        print(f"\n  [最终结果] 已保存至 {result_path}")

        print("\n" + "=" * 60)
        print("Step 5 完成!")
        print(f"  维度数: {len(dims)}")
        print(f"  标签数: {write_result.get('updated', 0)}")
        print(f"  清理空列: {write_result.get('cleaned', 0)}")
        print("=" * 60)

        return {
            "dims": dims,
            "tags_count": write_result.get("updated", 0),
        }

    def _dimension_mining(self, docs_source: str = "rdb") -> List[str]:
        """维度挖掘: 使用 dimension_integration 的 DimensionMiner

        Args:
            docs_source: 数据源（'rdb' 或 chunks 文件/目录路径）
        """
        print("  [5a-1] 使用 DimensionMiner 进行维度挖掘...")

        miner = DimensionMiner(docs_source=docs_source)

        print("  [5a-2] 聚类采样...")
        sampled_docs = miner.step1_clustering_sampling()

        print("  [5a-3] LLM 归纳候选维度...")
        v_cand = miner.step2_generate_candidates(sampled_docs)

        print("  [5a-4] 迭代优化...")
        v_core = miner.step3_iterative_optimization(v_cand)

        return v_core

    def _clear_step5_cache(self, dataset: str = None) -> Dict[str, int]:
        """清除 Step 5 的中间缓存文件（强制从 0 开始）

        会删除（若存在）：
        - experiment_data/V_cand.json           (数据集隔离时 V_cand_{dataset}.json)
        - experiment_data/V_core.json           (数据集隔离时 V_core_{dataset}.json)
        - experiment_data/tags_output.json      (数据集隔离时 tags_output_{dataset}.json)
        - experiment_data/step5_result.json     (数据集隔离时 step5_result_{dataset}.json)

        返回: {"deleted": int, "skipped": int, "missing": [str]}
        """
        experiment_data_dir = project_root / "experiment_data"

        candidates: List[Path] = []
        if dataset:
            candidates.extend([
                experiment_data_dir / f"V_cand_{dataset}.json",
                experiment_data_dir / f"V_core_{dataset}.json",
                experiment_data_dir / f"tags_output_{dataset}.json",
                experiment_data_dir / f"step5_result_{dataset}.json",
            ])
        else:
            candidates.extend([
                experiment_data_dir / "V_cand.json",
                experiment_data_dir / "V_core.json",
                experiment_data_dir / "tags_output.json",
                experiment_data_dir / "step5_result.json",
            ])

        deleted = 0
        skipped = 0
        missing: List[str] = []
        print("\n[Reset] 清除 Step 5 缓存:")
        for p in candidates:
            try:
                if p.exists():
                    p.unlink()
                    deleted += 1
                    print(f"    ✓ 删除 {p}")
                else:
                    missing.append(p.name)
                    print(f"    - 跳过（不存在）{p}")
            except Exception as e:
                skipped += 1
                print(f"    ✗ 删除失败 {p}: {e}")
        print(f"  [Reset] 总结: 删除={deleted}, 失败={skipped}, 不存在={len(missing)}\n")
        return {"deleted": deleted, "skipped": skipped, "missing": missing}

    def _load_all_chunks_from_rdb(self, limit: int = 10000) -> List[Dict]:
        """从 MySQL 加载所有 chunk"""
        docs = load_docs_from_rdb()
        return [{"doc_id": d["id"], "doc_text": d["text"]} for d in docs[:limit]]

    def _parse_chunks_data(self, data, src_label: str = "<data>") -> List[Dict]:
        """把已加载的 JSON 数据解析为统一格式 [{"doc_id": ..., "doc_text": ...}, ...]

        支持：
        1. list[{"id": ..., "text": ...}]
        2. list[{"doc_id": ..., "doc_text": ...}]
        3. list[{"doc_id": ..., "text": ...}]
        4. {"chunks": [...]} / {"documents": [...]} / {"data": [...]} 包装
        """
        if isinstance(data, dict):
            for key in ("chunks", "documents", "docs", "data", "items"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                raise ValueError(f"无法识别 {src_label} 的 JSON 结构，期望 list 或包含 chunks/documents 字段的 dict")

        if not isinstance(data, list):
            raise ValueError(f"chunks 数据不是 list: {type(data).__name__}")

        chunks: List[Dict] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            doc_id = item.get("id") or item.get("doc_id") or item.get("docid")
            doc_text = (
                item.get("text")
                or item.get("doc_text")
                or item.get("content")
                or item.get("chunk_text")
            )
            if doc_id is None:
                doc_id = f"chunk_{i:06d}"
            if doc_text is None:
                continue
            chunks.append({"doc_id": str(doc_id), "doc_text": str(doc_text)})
        return chunks

    def _load_chunks_from_file(self, chunks_file: str, limit: int = 100000) -> List[Dict]:
        """从 chunks 文件或目录加载（兼容多种格式）

        chunks_file 可以是：
        - 单个 JSON 文件
        - 包含多个 *_chunks.json 的目录（会自动遍历 *.json 合并）
        """
        path = Path(chunks_file)
        if not path.exists():
            raise FileNotFoundError(f"chunks 路径不存在: {path}")

        # 目录分支：遍历所有 *_chunks.json 合并加载
        if path.is_dir():
            json_files = sorted(path.glob("*.json"))
            if not json_files:
                raise FileNotFoundError(f"目录下未找到任何 .json: {path}")

            merged: List[Dict] = []
            for jf in json_files:
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    parsed = self._parse_chunks_data(data, src_label=str(jf))
                    merged.extend(parsed)
                except Exception as e:
                    print(f"    [Warning] 跳过 {jf.name}: {e}")

            if limit and len(merged) > limit:
                merged = merged[:limit]
            print(f"    [chunks_file] 从目录 {path}/ 合并 {len(json_files)} 个 JSON，共 {len(merged)} 条（limit={limit}）")
            return merged

        # 单文件分支
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = self._parse_chunks_data(data, src_label=str(path))
        if limit and len(chunks) > limit:
            chunks = chunks[:limit]
        print(f"    [chunks_file] 从 {path.name} 加载 {len(chunks)} 条")
        return chunks

    def _add_dimension_columns(self, dims: List[str]) -> Dict[str, int]:
        """添加维度列到 MySQL"""
        ensure_rdb_db()

        conn = connect_rdb()
        cur = conn.cursor()

        try:
            database = self.config.rdb.database
            table = self.config.rdb.table
            cur.execute(f"USE `{database}`")

            cur.execute(f"SHOW COLUMNS FROM `{table}`")
            existing_cols = {row[0] for row in cur.fetchall()}

            added = 0
            skipped = 0
            for dim in dims:
                col_name = normalize_attr_key(dim)
                if col_name in existing_cols:
                    skipped += 1
                    continue
                try:
                    sql = f"ALTER TABLE `{table}` ADD COLUMN `{col_name}` TEXT NULL"
                    cur.execute(sql)
                    conn.commit()
                    added += 1
                    if added <= 20:
                        print(f"    + 添加列: {col_name}")
                except Exception as e:
                    print(f"    [Error] 添加 {col_name} 失败: {e}")

            if added > 20:
                print(f"    ... 共添加 {added} 个维度列")
            print(f"    跳过 {skipped} 个已存在的列")

            return {"total_columns": added}
        except Exception as e:
            conn.rollback()
            print(f"  [Error] 添加维度列失败: {e}")
            return {"error": str(e)}
        finally:
            cur.close()
            conn.close()

    def _generate_tags(self, dims: List[str], dataset: str = None, chunks_file: str = None, docs_source: str = None) -> Dict[str, Dict[str, List[str]]]:
        """LLM 批量生成标签（支持按数据集隔离中间文件）

        Args:
            dims: 维度列表
            dataset: 数据集标识名
            chunks_file: 若指定，从该 JSON 文件加载 chunks；否则从 MySQL 加载
            docs_source: 数据源（'rdb' 或 chunks 文件/目录），优先级高于 chunks_file
        """
        from llm_service import DimensionMiningWithQwen
        import dimension_integration as di

        experiment_data_dir = project_root / "experiment_data"
        experiment_data_dir.mkdir(exist_ok=True)

        # 按数据集隔离 tags_output 文件
        if dataset:
            tags_output_path = experiment_data_dir / f"tags_output_{dataset}.json"
            di.PATH_TAGS = tags_output_path
        else:
            tags_output_path = experiment_data_dir / "tags_output.json"

        results = {}
        if tags_output_path.exists():
            with open(tags_output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"    断点续传，已处理 {len(results)} 条（数据集: {dataset or 'default'}）")
            if chunks_file or (docs_source and docs_source != "rdb"):
                print(f"    [注意] 续传基于历史 tags_output，本轮将从新数据源加载后按 doc_id 去重")

        miner = DimensionMiningWithQwen()

        # 决定数据源
        effective_source = docs_source or chunks_file or "rdb"

        if effective_source != "rdb":
            # 从文件加载
            chunks = self._load_chunks_from_file(effective_source, limit=100000)
        else:
            chunks = self._load_all_chunks_from_rdb(limit=100000)

        seen_ids: Set[str] = set()
        unique_chunks = []
        for chunk in chunks:
            if chunk["doc_id"] not in seen_ids:
                seen_ids.add(chunk["doc_id"])
                unique_chunks.append(chunk)
        if len(unique_chunks) < len(chunks):
            print(f"    去重: {len(chunks)} 条 -> {len(unique_chunks)} 条")

        processed = 0
        for i, chunk in enumerate(unique_chunks):
            doc_id = chunk["doc_id"]
            if doc_id in results:
                continue

            text = chunk["doc_text"]
            if not text or len(text) < 10:
                results[doc_id] = {}
                continue

            try:
                extracted = miner.extract_batch_dimensions(text, dims)
                if extracted:
                    results[doc_id] = extracted
                else:
                    keywords = miner.extract_keywords_fallback(text)
                    if keywords:
                        results[doc_id] = {"其他": keywords}
                    else:
                        results[doc_id] = {}
            except Exception as e:
                print(f"    [Error] {doc_id}: {e}")
                results[doc_id] = {}

            processed += 1
            if processed % 50 == 0:
                print(f"    已处理 {processed}/{len(chunks) - len(results)} ...")
                with open(tags_output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False)

        with open(tags_output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

        print(f"    标签生成完成，共 {len(results)} 条")
        return results

    def _write_tags_to_rdb(self, tags_result: Dict[str, Dict[str, List[str]]], dims: List[str] = None, dataset: str = None) -> Dict[str, int]:
        """将标签写入 MySQL（支持按数据集隔离维度文件）"""
        if not tags_result:
            print("    无标签数据，跳过写入")
            return {"updated": 0, "cleaned": 0}

        ensure_rdb_db()

        config = self.config
        table = config.rdb.table
        existing_cols = fetch_rdb_columns(table)
        experiment_data_dir = project_root / "experiment_data"

        if dims is None:
            if dataset:
                v_core_path = experiment_data_dir / f"V_core_{dataset}.json"
            else:
                v_core_path = PATH_V_CORE
            if v_core_path.exists():
                with open(v_core_path, "r", encoding="utf-8") as f:
                    dims = json.load(f)
            else:
                dims = []

        dim_cols = {dim: normalize_attr_key(dim) for dim in dims}

        conn = connect_rdb()
        cur = conn.cursor()

        updated = 0
        failed = 0
        batch_size = 100
        items = list(tags_result.items())

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]

            for doc_id, doc_tags in batch:
                try:
                    label_cols = []
                    label_vals = []

                    for dim, val in doc_tags.items():
                        if dim not in dim_cols:
                            continue
                        col = dim_cols[dim]
                        if col not in existing_cols:
                            continue

                        if isinstance(val, list):
                            val = "; ".join(str(v) for v in val if v)

                        label_cols.append(col)
                        label_vals.append(str(val))

                    if not label_cols:
                        continue

                    sql = f"UPDATE `{table}` SET "
                    sql += ", ".join(f"`{col}` = %s" for col in label_cols)
                    sql += f", `updated_at` = NOW()"
                    sql += f" WHERE `doc_id` = %s"

                    cur.execute(sql, tuple(label_vals + [doc_id]))
                    conn.commit()

                    if cur.rowcount == 1:
                        updated += 1
                    elif cur.rowcount > 1:
                        updated += 1
                        print(f"    [Warning] doc_id={doc_id} 在表中有多条记录，"
                              f"已更新 {cur.rowcount} 行，建议对表去重")
                    else:
                        failed += 1

                except Exception as e:
                    print(f"    [Error] {doc_id}: {e}")
                    failed += 1

        print("    正在清理空列...")
        cleaned = self._cleanup_empty_dim_columns(cur, conn, table)

        cur.close()
        conn.close()

        print(f"    写入完成: 成功 {updated} 篇, 失败 {failed} 篇, 清理 {cleaned} 个空列")
        return {"updated": updated, "failed": failed, "cleaned": cleaned}

    def _cleanup_empty_dim_columns(self, cur, conn, table: str) -> int:
        """删除全是 NULL 的维度列"""
        cur.execute(f"SHOW COLUMNS FROM `{table}`")
        all_cols = [row[0] for row in cur.fetchall()]
        dim_cols = [c for c in all_cols if c.startswith('dim_')]

        cleaned = 0
        for col in dim_cols:
            try:
                cur.execute(f"SELECT COUNT(*) FROM `{table}` WHERE `{col}` IS NOT NULL AND `{col}` != ''")
                count = cur.fetchone()[0]

                if count == 0:
                    sql = f"ALTER TABLE `{table}` DROP COLUMN `{col}`"
                    cur.execute(sql)
                    conn.commit()
                    cleaned += 1
                    print(f"    删除空列: {col}")
            except Exception as e:
                print(f"    [Warning] 删除列 {col} 失败: {e}")

        return cleaned

    # ==================== Step 6: 索引构建 + Qdrant 迁移 ====================

    def step6_build_index_and_migrate(self) -> Dict[str, Any]:
        """
        Step 6: 索引构建 + Qdrant 全量迁移
          6a: 构建检索索引
          6b: 全量数据迁移到 Qdrant
        """
        print("\n" + "=" * 60)
        print("Step 6: 索引构建 + Qdrant 迁移")
        print("=" * 60)

        print("\n[6a] 构建检索索引...")
        index_result = self._build_search_index()

        print("\n[6b] 全量迁移到 Qdrant...")
        migrate_result = self._migrate_to_qdrant()

        print("\n" + "=" * 60)
        print("Step 6 完成!")
        print(f"  倒排索引: {index_result.get('inverted_index', '')}")
        print(f"  维度元数据: {index_result.get('dim_metadata', '')}")
        print(f"  迁移 chunks: {migrate_result.get('inserted', 0)}")
        print("=" * 60)

        return {
            "index_result": index_result,
            "migrate_result": migrate_result,
        }

    def _build_search_index(self) -> Dict[str, Any]:
        """
        构建检索索引（支持多数据集增量合并）

        - 扫描所有 tags_output*.json（支持 tags_output.json 和 tags_output_{dataset}.json）
        - 相同维度：合并值集合
        - 不同维度：各自保留
        - 覆盖率以合并后总量重新计算，低于 TH_COV 的过滤
        """
        from collections import defaultdict
        import pickle
        from FlagEmbedding import BGEM3FlagModel

        experiment_data_dir = project_root / "experiment_data"
        experiment_data_dir.mkdir(exist_ok=True)

        # 收集所有 tags_output 文件
        tags_files = []
        tags_files.append(experiment_data_dir / "tags_output.json")
        tags_files.extend(experiment_data_dir.glob("tags_output_*.json"))

        available = [f for f in tags_files if f.exists()]
        if not available:
            print("  [Warning] 未找到任何 tags_output*.json，跳过索引构建")
            return {}

        # 合并所有数据集的标签（同时追踪每个数据集的维度和文档数）
        # merged_doc_data 仅用于 total_docs 计数
        merged_doc_data: Dict[str, Dict[str, List[str]]] = {}
        datasets_loaded = []
        inverted_index = defaultdict(lambda: defaultdict(list))
        dim_value_sets: Dict[str, Set[str]] = defaultdict(set)
        dim_dataset_doc_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        dataset_doc_counts: Dict[str, int] = defaultdict(int)

        for f in available:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            name = f.stem.replace("tags_output", "").lstrip("_") or "default"
            datasets_loaded.append(name)
            dataset_doc_counts[name] = len(data)
            merged_doc_data.update(data)
            print(f"  加载 {f.name}: {len(data)} 条文档")

            for doc_id, tags in data.items():
                for dim, vals in tags.items():
                    if not vals:
                        continue
                    if isinstance(vals, str):
                        vals = [vals]
                    for v in vals:
                        inverted_index[dim][v].append(doc_id)
                        dim_value_sets[dim].add(v)
                        dim_dataset_doc_counts[dim][name] += 1

        total_docs = len(merged_doc_data)
        print(f"  合并后共 {total_docs} 条文档（来自 {len(datasets_loaded)} 个数据集: {datasets_loaded}）")

        # 过滤低覆盖率维度
        # 策略：任一数据集覆盖率 >= 50%，就保留该维度
        MIN_COV = 0.50
        # 同时要求维度至少有 5 个不同的 tag 值（防止极端稀疏维度）
        MIN_TAG_COUNT = 5
        dims_to_vectorize = []
        dims_filtered = []

        for dim, val_set in dim_value_sets.items():
            per_dataset_coverage = {
                ds: dim_dataset_doc_counts[dim][ds] / dataset_doc_counts[ds]
                if dataset_doc_counts[ds] > 0 else 0.0
                for ds in dataset_doc_counts
            }
            max_cov = max(per_dataset_coverage.values()) if per_dataset_coverage else 0.0
            best_ds = max(per_dataset_coverage, key=per_dataset_coverage.get) if per_dataset_coverage else ""

            # 双重过滤：覆盖率 >= 50% AND tag 数 >= 5
            if max_cov >= MIN_COV and len(val_set) >= MIN_TAG_COUNT:
                val_list = sorted(list(val_set))
                dims_to_vectorize.append((dim, val_list))
            else:
                reason = []
                if max_cov < MIN_COV:
                    reason.append(f"覆盖率 {max_cov:.1%} < {MIN_COV:.0%}")
                if len(val_set) < MIN_TAG_COUNT:
                    reason.append(f"tag数 {len(val_set)} < {MIN_TAG_COUNT}")
                dims_filtered.append((dim, len(val_set), max_cov, best_ds, per_dataset_coverage, "; ".join(reason)))
                del inverted_index[dim]

        print(f"  维度统计（共 {len(dim_value_sets)} 个候选维度）:")
        print(f"    通过覆盖率过滤（任一数据集 ≥{MIN_COV:.0%} 且 tag数 ≥{MIN_TAG_COUNT}）: {len(dims_to_vectorize)} 个")
        if dims_filtered:
            for dim, n, cov, best_ds, per_ds, reason in sorted(dims_filtered, key=lambda x: x[2]):
                cov_str = ", ".join(f"{ds}: {c:.1%}" for ds, c in sorted(per_ds.items()))
                print(f"      - [{dim}]: {n} 值, 最高覆盖率 {cov:.1%}（{best_ds}）{cov_str}  | 过滤原因: {reason}")


        # 构建维度元数据
        dim_meta = {}
        for dim, val_list in dims_to_vectorize:
            dim_meta[dim] = {
                "is_enum": False,
                "value_count": len(val_list),
                "values": val_list,
            }

        # 向量化所有维度值
        if dims_to_vectorize:
            print("  构建开放维度向量索引...")
            current_dir = str(project_root / "code")
            embedding_model_path = os.path.join(current_dir, 'bge-m3')
            if not os.path.exists(embedding_model_path):
                embedding_model_path = str(project_root / "model" / "bge-m3")

            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"  使用设备: {device}")
            encoder = BGEM3FlagModel(embedding_model_path, use_fp16=(device == 'cuda'), device=device)
            tag_vectors = {}

            for dim, val_list in dims_to_vectorize:
                print(f"    Encoding {dim} ({len(val_list)} values)...")
                embeddings = encoder.encode(val_list, return_dense=True)['dense_vecs']
                tag_vectors[dim] = {
                    "values": val_list,
                    "vectors": embeddings,
                }

            vectors_path = experiment_data_dir / "tag_vectors.pkl"
            with open(vectors_path, "wb") as f:
                pickle.dump(tag_vectors, f)
            print(f"  -> {vectors_path}")

        inverted_path = experiment_data_dir / "inverted_index.json"
        with open(inverted_path, "w", encoding="utf-8") as f:
            json.dump(inverted_index, f, ensure_ascii=False)
        print(f"  -> {inverted_path}")

        meta_path = experiment_data_dir / "dimension_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(dim_meta, f, ensure_ascii=False, indent=2)
        print(f"  -> {meta_path}")

        print(f"  索引构建完成: {len(dim_meta)} 个维度, {total_docs} 条文档")

        return {
            "inverted_index": str(inverted_path),
            "dim_metadata": str(meta_path),
        }

    def _migrate_to_qdrant(self) -> Dict[str, Any]:
        """全量迁移到 Qdrant

        Qdrant 特点：
        - 每个 point 支持多向量（named vectors）
        - payload 存储所有元数据（无需预定义 Schema）
        - 中文列名直接存储，无需转拼音
        """
        print("  [6b-1] 加载 MySQL 全量数据...")

        conn = self._connect_rdb()
        cur = conn.cursor()
        try:
            database = self.config.rdb.database
            table = self.config.rdb.table
            cur.execute(f"USE `{database}`")
            cur.execute(f"SELECT * FROM `{table}`")
            rows = cur.fetchall()
            col_names = [row[0] for row in cur.description]

            chunks_data = []
            for row in rows:
                row_dict = dict(zip(col_names, row))
                chunks_data.append(row_dict)

            print(f"  [6b-1] 加载 {len(chunks_data)} 条数据")

        except Exception as e:
            print(f"  [Error] 加载失败: {e}")
            return {"error": str(e)}
        finally:
            cur.close()
            conn.close()

        if not chunks_data:
            print("  [Warning] 无数据，跳过迁移")
            return {"inserted": 0}

        print("  [6b-2] 加载 Embedding 模型...")
        self._get_embedding_model()

        print("  [6b-3] 编码向量并构建 payload...")

        points = []
        for i, chunk in enumerate(chunks_data):
            doc_id = chunk.get("doc_id", f"chunk_{i}")
            doc_text = chunk.get("doc_text", "")
            profile_str = chunk.get("profile_json", "{}")

            if isinstance(profile_str, str):
                try:
                    profile = json.loads(profile_str)
                except Exception:
                    profile = {}
            else:
                profile = profile_str or {}

            doc_title = profile.get("file_name", doc_id)
            chunk_title = doc_title[:80]

            # 构建 payload（Qdrant 支持动态字段）
            payload = {
                "chunk_id": doc_id,
                "doc_id_link": doc_id,
                "doc_title": doc_title,
                "chunk_gen_title": chunk_title,
                "chunk_text": doc_text[:5000],
                "chunk_index": profile.get("chunk_index", i),
                "chunk_up_cid": "",
                "chunk_down_cid": "",
                "chunk_len": len(doc_text),
                "genre": profile.get("genre", "doc"),
                "l_min": profile.get("l_min", 0),
                "l_max": profile.get("l_max", 0),
            }

            # 添加维度列到 payload
            for key, val in chunk.items():
                if key.startswith("dim_") and val:
                    payload[key] = val

            # 多向量
            from qdrant_client.models import PointStruct
            point = PointStruct(
                id=i,
                vector={
                    "doc_title_vec": self._encode_text(doc_title),
                    "chunk_title_vec": self._encode_text(chunk_title),
                    "chunk_text_vec": self._encode_text(doc_text[:2000]),
                },
                payload=payload,
            )
            points.append(point)

            if (i + 1) % 100 == 0:
                print(f"    已处理 {i + 1} / {len(chunks_data)} ...")

        print(f"  [6b-3] 编码完成，{len(points)} 个 point")

        payload_field_count = len(points[0].payload) if points else 0
        vector_count = len(points[0].vector) if points else 0
        print(f"  字段统计: {vector_count} 向量字段 + {payload_field_count} payload 字段")

        print("  [6b-4] 写入 Qdrant...")
        client = self._connect_qdrant()
        cfg = self._get_qdrant_config()
        collection_name = cfg["collection_name"]

        # 检查 Collection 是否存在，不存在则创建
        def collection_exists(name: str) -> bool:
            try:
                return name in client.get_collections().collections
            except Exception:
                return False

        if not collection_exists(collection_name):
            print(f"  [6b-4] Collection 不存在，通过 REST API 创建...")
            import urllib.request
            import urllib.error
            url = f"http://{cfg['url'].split('://')[1]}/collections/{collection_name}"
            model = self._get_embedding_model()
            vector_dim = model.get_sentence_embedding_dimension()
            payload = {
                "vectors": {
                    "doc_title_vec": {"size": vector_dim, "distance": "Cosine"},
                    "chunk_title_vec": {"size": vector_dim, "distance": "Cosine"},
                    "chunk_text_vec": {"size": vector_dim, "distance": "Cosine"},
                },
                "hnsw_config": {"m": cfg["hnsw_m"], "ef_construct": cfg["hnsw_ef_construct"]},
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="PUT")
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    print(f"  [6b-4] Collection 创建成功: {resp.status}")
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8")
                if e.code == 409:
                    print(f"  [6b-4] Collection 已存在，跳过创建")
                else:
                    print(f"  [6b-4] 创建失败 ({e.code}): {body}")
                    return {"error": f"创建 Collection 失败: {body}"}

        # 批量写入（Qdrant 建议每批 100-1000 条）
        batch_size = 500
        total_inserted = 0

        for batch_start in range(0, len(points), batch_size):
            batch_end = min(batch_start + batch_size, len(points))
            batch = points[batch_start:batch_end]

            for retry in range(3):
                try:
                    operation_info = client.upsert(
                        collection_name=collection_name,
                        points=batch,
                    )
                    break
                except Exception as e:
                    if retry < 2:
                        print(f"    写入失败 (重试 {retry + 1}/3): {e}")
                        import time
                        time.sleep(2)
                    else:
                        raise

            total_inserted += len(batch)
            print(f"    写入 {batch_start}-{batch_end} / {len(points)} ...")

        print(f"  [6b-4] 写入完成，{total_inserted} 个 points")

        return {
            "inserted": total_inserted,
            "vector_fields": vector_count,
            "payload_fields": payload_field_count,
        }

    # ==================== run ====================

    def run(
        self,
        step: int = None,
        from_step: int = None,
        to_step: int = None,
        all: bool = False,
        input_path: str = None,
        chunks_file: str = None,
        force: bool = False,
        corpus_id: str = "",
        dataset: str = "",
        docs_source: str = "",
        args = None,
    ):
        """
        运行流水线

        Args:
            step: 单步执行 (1-6)
            from_step: 起始步骤
            to_step: 结束步骤
            all: 完整流程 (step 1-6)
            input_path: 输入路径
            chunks_file: 分片文件路径（Step 3/4 可用）
            force: 强制重建
            corpus_id: 语料库 ID
            dataset: 数据集标识名
        """
        _args = args if args is not None else self.args
        skip_step1 = getattr(_args, 'skip_step1', False)
        skip_step2 = getattr(_args, 'skip_step2', False)
        skip_step3 = getattr(_args, 'skip_step3', False)
        skip_step4 = getattr(_args, 'skip_step4', False)
        skip_step5 = getattr(_args, 'skip_step5', False)
        extract_only = getattr(_args, 'extract_only', False)
        reset_cache = getattr(_args, 'reset_cache', False)
        skip_step6 = getattr(_args, 'skip_step6', False)
        dataset = getattr(_args, 'dataset', "") or ""
        docs_source = getattr(_args, 'docs_source', "") or ""

        self.args = argparse.Namespace(
            input=input_path,
            chunks_file=chunks_file,
            force=force,
            corpus_id=corpus_id,
            dataset=dataset,
            docs_source=docs_source,
            skip_step1=skip_step1,
            skip_step2=skip_step2,
            skip_step3=skip_step3,
            skip_step4=skip_step4,
            skip_step5=skip_step5,
            skip_step6=skip_step6,
        )

        try:
            if all:
                steps_to_run = list(range(1, 7))
            elif step is not None:
                steps_to_run = [step]
            elif from_step is not None and to_step is not None:
                steps_to_run = list(range(from_step, to_step + 1))
            else:
                print("[Error] 请指定 --step, --from_step/--to_step, 或 --all")
                return

            print(f"\n{'=' * 60}")
            print(f"执行步骤: {steps_to_run}")
            print(f"{'=' * 60}")

            skip_flags = {
                1: self.args.skip_step1,
                2: self.args.skip_step2,
                3: self.args.skip_step3,
                4: self.args.skip_step4,
                5: self.args.skip_step5,
                6: self.args.skip_step6,
            }

            resolved_chunks = str(project_root / "output_chunks" / "all_chunks_resolved.json")
            write_chunks_file = chunks_file
            if not write_chunks_file and Path(resolved_chunks).exists():
                write_chunks_file = resolved_chunks

            results = {}
            for s in steps_to_run:
                if skip_flags.get(s):
                    print(f"\n[跳过] Step {s}")
                    continue
                if s == 1:
                    results[1] = self.step1_init_db(force=force)
                elif s == 2:
                    results[2] = self.step2_chunking(input_path=input_path)
                elif s == 3:
                    results[3] = self.step3_coreference_resolution(chunks_file=chunks_file)
                elif s == 4:
                    results[4] = self.step4_write_to_rdb(
                        chunks_file=write_chunks_file,
                        corpus_id=corpus_id,
                        force=force,
                    )
                elif s == 5:
                    _step5_chunks = getattr(self.args, 'chunks_file', '') or None
                    _step5_docs_source = getattr(self.args, 'docs_source', '') or None
                    results[5] = self.step5_dimension_tagging(
                        dataset=dataset,
                        extract_only=extract_only,
                        chunks_file=_step5_chunks,
                        reset_cache=reset_cache,
                        docs_source=_step5_docs_source,
                    )
                elif s == 6:
                    results[6] = self.step6_build_index_and_migrate()

            print(f"\n{'=' * 60}")
            print(f"流水线执行完成!")
            print(f"{'=' * 60}")
            return results

        finally:
            self._close_rdb()
            self._close_qdrant()


def main():
    parser = argparse.ArgumentParser(
        description="RAG DB 流水线 (Qdrant 版本)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单步执行
  python pipeline_qdrant.py --step 1

  # 完整流程
  python pipeline_qdrant.py --all --input ./data_input/test_data

  # 指定步骤范围
  python pipeline_qdrant.py --from_step 2 --to_step 6 --input ./data_input/test_data

  # 强制重建数据库
  python pipeline_qdrant.py --step 1 --force
        """
    )

    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5, 6],
                        help="执行单个步骤 (1:数据库, 2:分片, 3:指代消解, 4:入库, 5:维度, 6:迁移)")
    parser.add_argument("--from_step", type=int, choices=[1, 2, 3, 4, 5, 6],
                        help="起始步骤")
    parser.add_argument("--to_step", type=int, choices=[1, 2, 3, 4, 5, 6],
                        help="结束步骤")
    parser.add_argument("--all", action="store_true",
                        help="执行完整流程 (step 1-6)")
    parser.add_argument("--skip_step1", action="store_true",
                        help="跳过 Step 1: 初始化数据库")
    parser.add_argument("--skip_step2", action="store_true",
                        help="跳过 Step 2: 分片处理")
    parser.add_argument("--skip_step3", action="store_true",
                        help="跳过 Step 3: 指代消解")
    parser.add_argument("--skip_step4", action="store_true",
                        help="跳过 Step 4: 分片入库")
    parser.add_argument("--skip_step5", action="store_true",
                        help="跳过 Step 5: 维度抽取")
    parser.add_argument("--extract_only", action="store_true",
                        help="Step 5 仅抽取维度并生成文件，不写入 MySQL（跳过 5b/5d）")
    parser.add_argument("--reset_cache", action="store_true",
                        help="Step 5 开始前清空 V_cand/V_core/tags_output/step5_result 等中间缓存，完全从 0 开始重新挖掘")
    parser.add_argument("--skip_step6", action="store_true",
                        help="跳过 Step 6: 索引构建与迁移")
    parser.add_argument("--input", "-i", default="",
                        help="输入文件或目录")
    parser.add_argument("--chunks_file", default="",
                        help="分片文件路径 (Step 3/4/5：Step 3/4 读入分片，Step 5 不入库时直接以该文件作为维度抽取源)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="强制重建数据库")
    parser.add_argument("--corpus_id", default="",
                        help="语料库 ID")
    parser.add_argument("--dataset", "-d", default="",
                        help="数据集标识名（用于隔离 tags_output_{dataset}.json 和 V_core_{dataset}.json，缺省则用默认文件）")
    parser.add_argument("--docs_source", default="",
                        help="Step 5 数据源（'rdb' 或 chunks JSON 文件/目录路径；缺省时若 --chunks_file 给出则使用它，否则使用 MySQL）")

    args = parser.parse_args()

    pipeline = Pipeline(args)
    pipeline.run(
        step=args.step,
        from_step=args.from_step,
        to_step=args.to_step,
        all=args.all,
        input_path=args.input,
        chunks_file=args.chunks_file,
        force=args.force,
        corpus_id=args.corpus_id,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
