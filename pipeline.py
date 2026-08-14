"""
RAG DB 流水线 - 重构版 完整版
======================

核心流程：
  Step 1: 初始化数据库 (MySQL + Milvus)
  Step 2: 分片处理 (调用 integrated_chunker.py)
  Step 3: 指代消解与自检验证 (调用 coreference_resolver.py)
  Step 4: 分片结果批量写入 MySQL
  Step 5: 维度抽取与打标 (整合 code/ 模块)
  Step 6: 索引构建 + Milvus 全量迁移

使用方式：
  # 单步执行
  python pipeline.py --step 1

  # 完整流程
  python pipeline.py --all --input ./data_input/test_data

  # 指定步骤范围
  python pipeline.py --from_step 2 --to_step 6 --input ./data_input/test_data

作者: 重构版本
日期: 2026-05-25
"""

import argparse
import json
import sys
import time
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from db_config import get_config


# 从 dimension_integration 导入维度挖掘和标签生成
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
    """RAG DB 流水线主类"""

    def __init__(self, args=None):
        self.config = get_config()
        self.args = args
        self._rdb_conn = None
        self._collection = None
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

    def _connect_vecdb(self, timeout: int = 60):
        """连接 Milvus"""
        from pymilvus import Collection, connections, utility

        if self._collection is None:
            connections.connect(
                "default",
                host=self.config.vecdb.host,
                port=str(self.config.vecdb.port)
            )
            collection_name = self.config.vecdb.collection_name
            if not utility.has_collection(collection_name):
                raise RuntimeError(f"VecDB Collection 不存在: {collection_name}")
            self._collection = Collection(collection_name)
            for attempt in range(3):
                try:
                    self._collection.load()
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"  Collection 加载失败，等待 {timeout}s 后重试...")
                        time.sleep(timeout)
                    else:
                        print(f"  Warning: Collection 加载失败: {e}")
        return self._collection

    def _close_vecdb(self):
        """关闭 Milvus 连接"""
        if self._collection:
            try:
                self._collection.release()
            except:
                pass
            self._collection = None
        try:
            from pymilvus import connections
            connections.disconnect()
        except:
            pass

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
        Step 1: 初始化数据库 (MySQL + Milvus)
        """
        print("\n" + "=" * 60)
        print("Step 1: 初始化数据库")
        print("=" * 60)

        results = {"rdb": {}, "vecdb": {}}

        # 1.1 初始化 MySQL
        print("\n[1.1] 初始化 MySQL...")
        results["rdb"] = self._init_rdb(force=force)

        # 1.2 初始化 Milvus
        print("\n[1.2] 初始化 Milvus...")
        results["vecdb"] = self._init_vecdb(force=force)

        print("\n" + "=" * 60)
        print("Step 1 完成!")
        print(f"  MySQL: {results['rdb'].get('message', '')}")
        print(f"  Milvus: {results['vecdb'].get('message', '')}")
        print("=" * 60)

        return results

    def _init_rdb(self, force: bool = False) -> Dict[str, Any]:
        """初始化 MySQL"""
        conn = self._connect_rdb()
        cur = conn.cursor()

        try:
            database = self.config.rdb.database
            table = self.config.rdb.table

            # 创建数据库
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{database}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            conn.commit()
            cur.execute(f"USE `{database}`")

            # 检查表是否存在
            cur.execute(f"SHOW TABLES LIKE %s", (table,))
            table_exists = cur.fetchone() is not None

            if table_exists and force:
                cur.execute(f"DROP TABLE `{table}`")
                conn.commit()
                table_exists = False
                print(f"  已删除旧表: {table}")

            if not table_exists:
                # 创建标准表结构
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

    def _init_vecdb(self, force: bool = False, timeout: int = 60, dim_field_names: List[str] = None) -> Dict[str, Any]:
        """初始化 Milvus Collection
        Args:
            force: 是否强制重建
            timeout: 加载超时时间
            dim_field_names: 预定义的维度字段名列表
        """
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

        try:
            connections.connect(
                "default",
                host=self.config.vecdb.host,
                port=str(self.config.vecdb.port)
            )

            collection_name = self.config.vecdb.collection_name

            # 获取向量维度
            model = self._get_embedding_model()
            vector_dim = model.get_sentence_embedding_dimension()

            # 检查 Collection 是否存在
            if utility.has_collection(collection_name):
                if force:
                    Collection(collection_name).drop()
                    print(f"  已删除旧 Collection: {collection_name}")
                else:
                    print(f"  Collection 已存在: {collection_name}")
                    return {"status": "exists", "message": f"Collection 已存在 {collection_name}"}

            # 定义 Schema
            fields = [
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
                FieldSchema(name="doc_id_link", dtype=DataType.VARCHAR, max_length=256, is_partition_key=True),
                FieldSchema(name="doc_title", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="chunk_gen_title", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="chunk_up_cid", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="chunk_down_cid", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="chunk_len", dtype=DataType.INT64),
                # 向量字段
                FieldSchema(name="doc_title_vec", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
                FieldSchema(name="chunk_title_vec", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
                FieldSchema(name="chunk_text_vec", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
                # 元数据
                FieldSchema(name="genre", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="l_min", dtype=DataType.INT64),
                FieldSchema(name="l_max", dtype=DataType.INT64),
            ]

            # 添加预定义的维度字段
            if dim_field_names:
                for field_name in dim_field_names:
                    fields.append(FieldSchema(name=field_name, dtype=DataType.VARCHAR, max_length=4096))
                print(f"  添加 {len(dim_field_names)} 个维度字段")

            # 维度列作为预定义字段存储
            schema = CollectionSchema(
                fields=fields,
                description="RAG Corpus Chunks",
                enable_dynamic_field=True
            )
            print(f"  Schema: {len(fields)} 个字段 ({15} 固定 + {len(dim_field_names) if dim_field_names else 0} 维度)")

            # 创建 Collection
            collection = Collection(name=collection_name, schema=schema)
            print(f"  创建 Collection: {collection_name}")

            # 创建索引
            index_params = {
                "metric_type": self.config.vecdb.metric_type,
                "index_type": self.config.vecdb.index_type,
                "params": {"nlist": self.config.vecdb.nlist}
            }

            for vec_field in ["doc_title_vec", "chunk_title_vec", "chunk_text_vec"]:
                collection.create_index(vec_field, index_params)
                print(f"    索引: {vec_field}")

            # 加载
            print(f"  正在加载 Collection (超时: {timeout}s)...")
            for attempt in range(3):
                try:
                    collection.load()
                    print(f"  Collection 加载完成")
                    return {"status": "created", "message": f"创建 Collection {collection_name}"}
                except Exception as e:
                    if attempt < 2:
                        print(f"  加载超时，等待 {timeout}s 后重试...")
                        time.sleep(timeout)
                    else:
                        print(f"  Collection 创建成功但加载超时")
                        return {"status": "created_not_loaded", "message": f"Collection 创建成功但加载超时"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ==================== Step 2: 分片处理 ====================

    def step2_chunking(self, input_path: str = None) -> Dict[str, Any]:
        """
        Step 2: 调用 integrated_chunker.py 进行分片处理
        """
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

        # 确定输出目录
        output_dir = project_root / "output_chunks"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 调用 integrated_chunker.py
        print(f"\n调用 integrated_chunker.py...")
        print(f"  输入: {input_path}")
        print(f"  输出: {output_dir}")

        # 构建命令
        cmd = [
            sys.executable,
            str(project_root / "integrated_chunker.py"),
            "--input", str(input_path),
            "--output", str(output_dir),
        ]

        # 如果有 LLM 配置，从 db_config.yaml 读取并传给 integrated_chunker.py
        llm_cfg = self.config.llm
        llm_api_key = getattr(llm_cfg, 'api_key', '') or os.environ.get("DASHSCOPE_API_KEY", "")
        llm_base_url = getattr(llm_cfg, 'base_url', '') or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        llm_model = getattr(llm_cfg, 'model', '') or os.environ.get("LLM_MODEL", "gpt-3.5-turbo")

        if llm_api_key:
            cmd.extend(["--llm_api_key", llm_api_key])
            cmd.extend(["--llm_base_url", llm_base_url])
            cmd.extend(["--llm_model", llm_model])

        # 执行
        import subprocess
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            print(f"[Error] 分片处理失败: returncode={result.returncode}")
            return {"error": "分片处理失败"}

        # 查找输出文件
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
        print(f"  输出文件: {output_dir / 'all_chunks_chunks.json'}")
        print(f"  汇总文件: {output_dir / 'all_chunks_summary.json'}")
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
        """
        Step 3: 指代消解与自检验证
          - 对分片结果进行代词/指示词消解
          - 自检验证：检测消解错误并回滚
          - 一致性检查：确保同一文档内指代消解一致
        """
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

        # 确定输出目录
        output_dir = chunks_path.parent

        # 调用 coreference_resolver.py
        print(f"\n调用 coreference_resolver.py...")
        print(f"  输入: {chunks_path}")

        # 构建命令
        cmd = [
            sys.executable,
            str(project_root / "coreference_resolver.py"),
            "--input", str(chunks_path),
            "--output", str(output_dir),
        ]

        # 如果有 LLM 配置，启用 LLM 验证
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

        # 执行
        import subprocess
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            print(f"[Error] 指代消解失败: returncode={result.returncode}")
            return {"error": "指代消解失败"}

        # 查找输出文件
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

    def step4_write_to_rdb(self, chunks_file: str = None, corpus_id: str = "") -> Dict[str, Any]:
        """
        Step 4: 分片结果批量写入 MySQL
        """
        print("\n" + "=" * 60)
        print("Step 4: 分片结果写入 MySQL")
        print("=" * 60)

        if not chunks_file:
            chunks_file = self.args.chunks_file if hasattr(self, 'args') and self.args else None
        if not chunks_file:
            # 默认使用指代消解后的文件
            chunks_file = str(project_root / "output_chunks" / "all_chunks_resolved.json")
            if not Path(chunks_file).exists():
                chunks_file = str(project_root / "output_chunks" / "all_chunks_chunks.json")

        chunks_path = Path(chunks_file)
        if not chunks_path.exists():
            print(f"[Error] 分片文件不存在: {chunks_path}")
            return {"error": f"文件不存在: {chunks_path}"}

        # 4.1 重建表
        print("\n[4.1] 重建 MainIndex 表...")
        result = self._reset_mainindex_table()
        print(f"  -> {result.get('message', result)}")

        # 4.2 加载分片数据
        print(f"\n[4.2] 加载分片数据: {chunks_path.name}")
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        # 展平为单个 chunk 列表
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

        # 4.3 批量写入
        print(f"\n[4.3] 批量写入 MySQL...")
        inserted, failed = self._batch_insert_chunks(all_chunks)
        print(f"  -> 成功 {inserted} 条，失败 {failed} 条")

        print("\n" + "=" * 60)
        print("Step 4 完成!")
        print(f"  写入: {inserted} / {len(all_chunks)} 个 chunk")
        print("=" * 60)

        return {"inserted": inserted, "failed": failed, "total": len(all_chunks)}

    def _reset_mainindex_table(self) -> Dict[str, Any]:
        """强制删除并重建 MainIndex 表"""
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

    def _batch_insert_chunks(self, chunks: List[Dict]) -> Tuple[int, int]:
        """批量写入 chunks 到 MySQL"""
        conn = self._connect_rdb()
        cur = conn.cursor()
        try:
            database = self.config.rdb.database
            table = self.config.rdb.table
            cur.execute(f"USE `{database}`")

            inserted = 0
            failed = 0

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

                    sql = f"""
                    INSERT INTO `{table}` (`doc_id`, `corpus_id`, `doc_text`, `profile_json`)
                    VALUES (%s, %s, %s, %s)
                    """
                    cur.execute(sql, (doc_id, corpus_id, doc_text, json.dumps(profile, ensure_ascii=False)))
                    conn.commit()
                    inserted += 1

                    if (i + 1) % 100 == 0:
                        print(f"    已写入 {i + 1}/{len(chunks)} ...")

                except Exception as e:
                    conn.rollback()
                    print(f"  [Error] {chunk.get('doc_id', '?')}: {e}")
                    failed += 1

            return inserted, failed
        finally:
            cur.close()
            conn.close()

    # ==================== Step 5: 维度抽取与打标 ====================

    def step5_dimension_tagging(self) -> Dict[str, Any]:
        """
        Step 5: 维度抽取与打标
          5a: 维度挖掘 (聚类 + LLM 归纳 + 迭代优化)
          5b: 添加维度列到 MySQL
          5c: LLM 批量生成标签
          5d: 标签回写 MySQL
        """
        print("\n" + "=" * 60)
        print("Step 5: 维度抽取与打标")
        print("=" * 60)

        sys.path.insert(0, str(project_root))
        sys.path.insert(0, str(project_root / "code"))

        try:
            from llm_service import DimensionMiningWithQwen
        except ImportError as e:
            print(f"[Error] 无法导入 LLM 服务: {e}")
            return {"error": f"导入失败: {e}"}

        # 5a: 维度挖掘
        print("\n[5a] 维度挖掘...")
        dims = self._dimension_mining()

        if not dims:
            print("[Warning] 未挖掘到任何维度，Step 5 终止")
            return {"error": "维度挖掘失败"}

        print(f"\n  -> 核心维度 ({len(dims)} 个): {dims}")

        # 5b: 添加维度列
        print("\n[5b] 添加维度列到 MySQL...")
        self._add_dimension_columns(dims)

        # 5c: 生成标签
        print("\n[5c] 生成维度标签...")
        tags_result = self._generate_tags(dims)

        # 5d: 回写标签（传入 dims 确保列名映射一致，避免磁盘缓存的 V_core.json 与当前 dims 不匹配）
        print("\n[5d] 标签回写 MySQL...")
        write_result = self._write_tags_to_rdb(tags_result, dims)

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

    def _dimension_mining(self) -> List[str]:
        """维度挖掘: 使用 dimension_integration 的 DimensionMiner"""
        print("  [4a-1] 使用 DimensionMiner 进行维度挖掘...")

        miner = DimensionMiner()

        # Step 1: 聚类采样
        print("  [4a-2] 聚类采样...")
        sampled_docs = miner.step1_clustering_sampling()

        # Step 2: 生成候选维度
        print("  [4a-3] LLM 归纳候选维度...")
        v_cand = miner.step2_generate_candidates(sampled_docs)

        # Step 3: 迭代优化
        print("  [4a-4] 迭代优化...")
        v_core = miner.step3_iterative_optimization(v_cand)

        return v_core

    def _load_all_chunks_from_rdb(self, limit: int = 10000) -> List[Dict]:
        """从 MySQL 加载所有 chunk"""
        docs = load_docs_from_rdb()
        return [{"doc_id": d["id"], "doc_text": d["text"]} for d in docs[:limit]]

    def _add_dimension_columns(self, dims: List[str]) -> Dict[str, int]:
        """添加维度列到 MySQL - 单列模式（每个维度只创建一列）

        Args:
            dims: 维度列表

        Returns:
            创建的列数统计
        """
        # 确保数据库存在
        ensure_rdb_db()

        conn = connect_rdb()
        cur = conn.cursor()

        try:
            database = self.config.rdb.database
            table = self.config.rdb.table
            cur.execute(f"USE `{database}`")

            # 检查现有列
            cur.execute(f"SHOW COLUMNS FROM `{table}`")
            existing_cols = {row[0] for row in cur.fetchall()}

            added = 0
            skipped = 0
            for dim in dims:
                # 使用 normalize_attr_key 生成列名，与 _write_tags_to_rdb 保持一致
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

    def _generate_tags(self, dims: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """LLM 批量生成标签"""
        from llm_service import DimensionMiningWithQwen

        experiment_data_dir = project_root / "experiment_data"
        experiment_data_dir.mkdir(exist_ok=True)
        tags_output_path = experiment_data_dir / "tags_output.json"

        # 断点续传
        results = {}
        if tags_output_path.exists():
            with open(tags_output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"    断点续传，已处理 {len(results)} 条")

        miner = DimensionMiningWithQwen()
        chunks = self._load_all_chunks_from_rdb(limit=100000)

        # ----------------------------------------------------------
        # Bug 修复：同一 doc_id 可能对应多条 chunk，直接按 doc_id
        # 去重（保留第一条）。否则同一 doc_id 会触发多次 LLM 调用，
        # 导致同一维度被重复提取出不同标签。
        # ----------------------------------------------------------
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

        # 最后保存
        with open(tags_output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

        print(f"    标签生成完成，共 {len(results)} 条")
        return results

    def _write_tags_to_rdb(self, tags_result: Dict[str, Dict[str, List[str]]], dims: List[str] = None) -> Dict[str, int]:
        """将标签写入 MySQL - 单列模式

        Args:
            tags_result: {doc_id: {dim_name: [val1, val2, ...]}}
            dims: 维度列表，用于构建列名映射。若为 None 则从 PATH_V_CORE 读取（保留兼容）。

        每个维度的多个值用英文分号连接，写入同一列。
        写入完成后删除全是 NULL 的列。
        """
        if not tags_result:
            print("    无标签数据，跳过写入")
            return {"updated": 0, "cleaned": 0}

        # 确保数据库存在
        ensure_rdb_db()

        config = self.config
        table = config.rdb.table
        existing_cols = fetch_rdb_columns(table)

        # 优先使用传入的 dims，必要时从磁盘读取（兼容外部调用）
        if dims is None:
            if PATH_V_CORE.exists():
                with open(PATH_V_CORE, "r", encoding="utf-8") as f:
                    dims = json.load(f)
            else:
                dims = []

        # 构建列名映射: {dim_name: col_name}，与 _add_dimension_columns 严格一致
        dim_cols = {dim: normalize_attr_key(dim) for dim in dims}

        # 连接 RDB
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

                        # 多值用分号连接
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

        # 写入完成后删除空列
        print("    正在清理空列...")
        cleaned = self._cleanup_empty_dim_columns(cur, conn, table)

        cur.close()
        conn.close()

        print(f"    写入完成: 成功 {updated} 篇, 失败 {failed} 篇, 清理 {cleaned} 个空列")
        return {"updated": updated, "failed": failed, "cleaned": cleaned}


    def _cleanup_empty_dim_columns(self, cur, conn, table: str) -> int:
        """删除全是 NULL 的维度列"""
        # 获取所有维度相关列
        cur.execute(f"SHOW COLUMNS FROM `{table}`")
        all_cols = [row[0] for row in cur.fetchall()]
        dim_cols = [c for c in all_cols if c.startswith('dim_')]
        
        cleaned = 0
        for col in dim_cols:
            try:
                # 检查该列是否全是 NULL
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


    def step6_build_index_and_migrate(self) -> Dict[str, Any]:
        """
        Step 6: 索引构建 + Milvus 全量迁移
          6a: 构建检索索引
          6b: 全量数据迁移到 Milvus
        """
        print("\n" + "=" * 60)
        print("Step 6: 索引构建 + Milvus 迁移")
        print("=" * 60)

        # 6a: 构建检索索引
        print("\n[6a] 构建检索索引...")
        index_result = self._build_search_index()

        # 6b: 迁移到 Milvus
        print("\n[6b] 全量迁移到 Milvus...")
        migrate_result = self._migrate_to_vecdb()

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
        """构建检索索引"""
        from collections import defaultdict
        import pickle
        from FlagEmbedding import BGEM3FlagModel

        experiment_data_dir = project_root / "experiment_data"
        experiment_data_dir.mkdir(exist_ok=True)

        tags_path = experiment_data_dir / "tags_output.json"
        v_core_path = experiment_data_dir / "V_core.json"

        if not tags_path.exists():
            print("  [Warning] 未找到 tags_output.json，跳过索引构建")
            return {}

        print("  加载标签数据...")
        with open(tags_path, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        # 构建倒排索引
        print("  构建倒排索引...")
        inverted_index = defaultdict(lambda: defaultdict(list))
        dim_value_sets = defaultdict(set)

        for doc_id, tags in doc_data.items():
            for dim, vals in tags.items():
                if not vals:
                    continue
                if isinstance(vals, str):
                    vals = [vals]
                for v in vals:
                    inverted_index[dim][v].append(doc_id)
                    dim_value_sets[dim].add(v)

        # 维度元数据
        dim_meta = {}
        dims_to_vectorize = []

        print("  分析维度属性...")
        for dim, val_set in dim_value_sets.items():
            val_list = sorted(list(set(val_set)))
            dim_meta[dim] = {
                "is_enum": False,
                "value_count": len(val_list),
                "values": val_list,
            }
            print(f"    维度 [{dim}]: {len(val_list)} 个值")
            if len(val_list) > 0:
                dims_to_vectorize.append((dim, val_list))

        # 构建开放维度向量
        if dims_to_vectorize:
            print("  构建开放维度向量索引...")
            current_dir = str(project_root / "code")
            embedding_model_path = os.path.join(current_dir, 'bge-m3')
            if not os.path.exists(embedding_model_path):
                embedding_model_path = str(project_root / "model" / "bge-m3")

            encoder = BGEM3FlagModel(embedding_model_path, use_fp16=True, device='cuda')
            tag_vectors = {}

            for dim, val_list in dims_to_vectorize:
                print(f"    Encoding {dim} ({len(val_list)} values)...")
                embeddings = encoder.encode(val_list, return_dense=True)['dense_vecs']
                tag_vectors[dim] = {
                    "values": val_list,
                    "vectors": embeddings,
                }

            # 保存 tag_vectors
            vectors_path = experiment_data_dir / "tag_vectors.pkl"
            with open(vectors_path, "wb") as f:
                pickle.dump(tag_vectors, f)
            print(f"  -> {vectors_path}")

        # 保存倒排索引
        inverted_path = experiment_data_dir / "inverted_index.json"
        with open(inverted_path, "w", encoding="utf-8") as f:
            json.dump(inverted_index, f, ensure_ascii=False)
        print(f"  -> {inverted_path}")

        # 保存维度元数据
        meta_path = experiment_data_dir / "dimension_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(dim_meta, f, ensure_ascii=False, indent=2)
        print(f"  -> {meta_path}")

        return {
            "inverted_index": str(inverted_path),
            "dim_metadata": str(meta_path),
        }

    def _migrate_to_vecdb(self) -> Dict[str, Any]:
        """全量迁移到 Milvus - 单列模式（与 MySQL dim_xxx 列完全对齐）

        维度数据直接从 MySQL 的 dim_xxx 列读取（分号连接的多值），
        Milvus 也每个维度只创建一列。Milvus 只支持 ASCII 字段名，
        因此 MySQL 的 dim_中文 列名会被映射为 dim_拼音缩写。
        """
        import hashlib

        def _dim_col_to_milvus_field(mysql_col: str) -> str:
            if not mysql_col.startswith("dim_"):
                return mysql_col
            cn_part = mysql_col[4:]
            _cn_map = {
                "人物身份": "renwu_shenfen",
                "传承方式": "chengchuan_fangshi",
                "历史时期": "lishi_shiqi",
                "圣物名称": "shengwu_mingcheng",
                "地理位置": "dili_weizhi",
                "重要事件": "zhongyao_shijian",
                "文化事件": "wenhua_shijian",
                "建筑名称": "jianzhu_mingcheng",
                "官职爵位": "guanzhi_juewei",
                "非物质文化遗产": "feiwuzhi_wenhua_yichan",
                "物质文化遗产": "wuzhi_wenhua_yichan",
                "文化传统": "wenhua_chuantong",
                "文献名称": "wenxian_mingcheng",
                "自然景观/文化景观": "ziran_jingguan",
                "时间范围": "shijian_fanwei",
                "书法艺术风格": "shufa_yishu_fengge",
                "封号": "fenghao",
                "教育职能": "jiaoyu_zhineng",
                "家族世系": "jiazu_shixi",
                "祭祀仪式": "jisi_yishi",
                "官职": "guanzhi",
                "其他": "qita",
                "宗教": "zongjiao",
                "供奉载体": "gongfeng_zaiti",
                "家族传承理念": "jiazu_chuancheng_linian",
                "儒家价值信仰": "rujia_jiazhi_xinyang",
                "职官封赠": "zhiguan_fengzeng",
                "封号职官": "fenghao_zhiguan",
                "仪式场所": "yishi_changsuo",
                "非遗行政级别": "feiyi_xingzheng_jibie",
            }
            if cn_part in _cn_map:
                return f"dim_{_cn_map[cn_part]}"
            suffix = hashlib.md5(cn_part.encode("utf-8")).hexdigest()[:6]
            return f"dim_cn{suffix}"

        print("  [5b-1] 加载 MySQL 全量数据...")

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

            print(f"  [5b-1] 加载 {len(chunks_data)} 条数据")

        except Exception as e:
            print(f"  [Error] 加载失败: {e}")
            return {"error": str(e)}
        finally:
            cur.close()
            conn.close()

        if not chunks_data:
            print("  [Warning] 无数据，跳过迁移")
            return {"inserted": 0}

        # MySQL 列名 -> Milvus 字段名 映射
        dim_col_names = [c for c in chunks_data[0].keys() if c.startswith('dim_')]
        dim_field_map = {c: _dim_col_to_milvus_field(c) for c in dim_col_names}
        milvus_dim_fields = sorted(dim_field_map.values())
        print(f"  [5b-1] 维度列: {len(milvus_dim_fields)} 个 -> {milvus_dim_fields[:5]}...")

        print("  [5b-2] 加载 Embedding 模型...")
        self._get_embedding_model()

        print("  [5b-3] 编码向量...")

        data_rows = []
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

            row = {
                "chunk_id": doc_id,
                "doc_id_link": doc_id,
                "doc_title": doc_title,
                "chunk_gen_title": chunk_title,
                "chunk_text": doc_text[:5000],
                "chunk_index": profile.get("chunk_index", i),
                "chunk_up_cid": "",
                "chunk_down_cid": "",
                "chunk_len": len(doc_text),
                "doc_title_vec": self._encode_text(doc_title),
                "chunk_title_vec": self._encode_text(chunk_title),
                "chunk_text_vec": self._encode_text(doc_text[:2000]),
                "genre": profile.get("genre", "doc"),
                "l_min": profile.get("l_min", 0),
                "l_max": profile.get("l_max", 0),
            }

            for mysql_col, milvus_field in dim_field_map.items():
                val = chunk.get(mysql_col, "")
                row[milvus_field] = val if val else ""

            data_rows.append(row)

            if (i + 1) % 100 == 0:
                print(f"    已处理 {i + 1} / {len(chunks_data)} ...")

        print(f"  [5b-3] 编码完成，{len(data_rows)} 个 chunk")

        fixed_field_names = [
            "chunk_id", "doc_id_link", "doc_title", "chunk_gen_title", "chunk_text",
            "chunk_index", "chunk_up_cid", "chunk_down_cid", "chunk_len",
            "doc_title_vec", "chunk_title_vec", "chunk_text_vec",
            "genre", "l_min", "l_max"
        ]
        all_field_names = fixed_field_names + milvus_dim_fields
        print(f"  字段统计: {len(fixed_field_names)} 固定字段 + {len(milvus_dim_fields)} 维度字段 = {len(all_field_names)} 总字段")

        print("  [5b-4] 写入 Milvus...")
        from pymilvus import Collection, connections, utility
        collection_name = self.config.vecdb.collection_name

        connections.connect("default", host=self.config.vecdb.host, port=str(self.config.vecdb.port))

        if utility.has_collection(collection_name):
            existing_schema = Collection(collection_name).schema
            existing_field_count = len(existing_schema.fields)
            expected_field_count = len(all_field_names)

            if existing_field_count != expected_field_count:
                print(f"  [Warning] Schema 不匹配: 期望 {expected_field_count} 个字段，当前 {existing_field_count} 个，重建 Collection...")
                Collection(collection_name).drop()
                self._collection = None

        if not utility.has_collection(collection_name):
            print(f"  [5b-4] 创建 Collection...")
            self._init_vecdb(force=False, dim_field_names=milvus_dim_fields)

        collection = self._connect_vecdb()
        collection.insert(data_rows)
        collection.flush()

        print(f"  [5b-4] 写入完成，{len(data_rows)} 条数据")
        return {
            "inserted": len(data_rows),
            "fixed_fields": len(fixed_field_names),
            "dim_fields": len(milvus_dim_fields),
            "total_cols": len(all_field_names)
        }


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
        """
        _args = args if args is not None else self.args
        skip_step1 = getattr(_args, 'skip_step1', False)
        skip_step2 = getattr(_args, 'skip_step2', False)
        skip_step3 = getattr(_args, 'skip_step3', False)
        skip_step4 = getattr(_args, 'skip_step4', False)
        skip_step5 = getattr(_args, 'skip_step5', False)
        skip_step6 = getattr(_args, 'skip_step6', False)

        self.args = argparse.Namespace(
            input=input_path,
            chunks_file=chunks_file,
            force=force,
            corpus_id=corpus_id,
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

            # Step 4 默认读取 Step 3 消解后的分片文件
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
                    )
                elif s == 5:
                    results[5] = self.step5_dimension_tagging()
                elif s == 6:
                    results[6] = self.step6_build_index_and_migrate()

            print(f"\n{'=' * 60}")
            print(f"流水线执行完成!")
            print(f"{'=' * 60}")
            return results

        finally:
            self._close_rdb()
            self._close_vecdb()


def main():
    parser = argparse.ArgumentParser(
        description="RAG DB 流水线 (重构版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单步执行
  python pipeline.py --step 1

  # 完整流程
  python pipeline.py --all --input ./data_input/test_data

  # 指定步骤范围
  python pipeline.py --from_step 2 --to_step 6 --input ./data_input/test_data

  # 强制重建数据库
  python pipeline.py --step 1 --force
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
    parser.add_argument("--skip_step6", action="store_true",
                        help="跳过 Step 6: 索引构建与迁移")
    parser.add_argument("--input", "-i", default="",
                        help="输入文件或目录")
    parser.add_argument("--chunks_file", default="",
                        help="分片文件路径 (Step 3/4，默认优先使用消解后文件)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="强制重建数据库")
    parser.add_argument("--corpus_id", default="",
                        help="语料库 ID")

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
    )


if __name__ == "__main__":
    main()
