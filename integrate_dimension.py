"""
将 dimension_integration.py 的维度抽取和打标逻辑集成到 pipeline.py
"""
import re

with open(r'd:\RAG_DB_slim\pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

# ========================================
# 1. 添加 import 语句
# ========================================
# 在现有的 import 语句后添加 dimension_integration 的导入

# 找到 import 语句的位置（在类定义之前）
import_section_end = content.find('\nclass Pipeline')
if import_section_end == -1:
    print("ERROR: Cannot find class Pipeline")
    exit(1)

# 添加新的 import
new_imports = '''
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

'''

# 在 class Pipeline 前插入新 import
content = content[:import_section_end] + new_imports + content[import_section_end:]

# ========================================
# 2. 替换 _dimension_mining 方法
# ========================================
old_dimension_mining = '''    def _dimension_mining(self) -> List[str]:
        """维度挖掘: 聚类采样 + LLM 归纳 + 迭代优化"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances_argmin_min
        from collections import Counter
        from scipy.stats import entropy
        from FlagEmbedding import BGEM3FlagModel
        import numpy as np

        print("  [4a-1] 加载分片数据...")
        chunks = self._load_all_chunks_from_rdb(limit=10000)
        if not chunks:
            print("  [Error] 无法从 RDB 加载数据")
            return []

        texts = [c["doc_text"] for c in chunks]
        doc_ids = [c["doc_id"] for c in chunks]

        print(f"  [4a-1] 已加载 {len(texts)} 个 chunk")

        # 编码
        print("  [4a-2] 编码文本...")
        current_dir = str(project_root / "code")
        embedding_model_path = os.path.join(current_dir, 'bge-m3')
        if not os.path.exists(embedding_model_path):
            embedding_model_path = str(project_root / "model" / "bge-m3")

        encoder = BGEM3FlagModel(embedding_model_path, use_fp16=True, device='cuda')
        vectors = encoder.encode(texts, return_dense=True)['dense_vecs']
        vectors = np.array(vectors, dtype=np.float32)

        # 聚类采样
        print("  [4a-3] 聚类采样...")
        K_CLUSTERS = 50
        N_CORE_SAMPLES = 5
        N_BOUND_SAMPLES = 5

        if len(vectors) < K_CLUSTERS:
            sampled_texts = texts
        else:
            kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors)

            selected_indices = set()
            for i in range(K_CLUSTERS):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) == 0:
                    continue
                cluster_vectors = vectors[cluster_indices]
                center = kmeans.cluster_centers_[i].reshape(1, -1)
                dists = pairwise_distances_argmin_min(cluster_vectors, center, metric='euclidean')[1]
                core_local_idx = np.argsort(dists)[:N_CORE_SAMPLES]
                bound_local_idx = np.argsort(dists)[-N_BOUND_SAMPLES:]
                for idx in np.concatenate([core_local_idx, bound_local_idx]):
                    selected_indices.add(cluster_indices[idx])

            sampled_texts = [texts[idx] for idx in selected_indices]

        print(f"  [4a-3] 采样 {len(sampled_texts)} 篇文档")

        # LLM 归纳候选维度
        print("  [4a-4] LLM 归纳候选维度...")
        from llm_service import DimensionMiningWithQwen
        print("  [DEBUG] 导入 DimensionMiningWithQwen 成功")
        miner = DimensionMiningWithQwen()
        print("  [DEBUG] 创建 DimensionMiningWithQwen 实例成功")

        experiment_data_dir = project_root / "experiment_data"
        experiment_data_dir.mkdir(exist_ok=True)
        v_cand_path = experiment_data_dir / "V_cand.json"
        v_core_path = experiment_data_dir / "V_core.json"

        if v_cand_path.exists():
            print("  [4a-4] 检测到已有候选维度，跳过生成")
            with open(v_cand_path, "r", encoding="utf-8") as f:
                dims = json.load(f)
        else:
            print("  [DEBUG] 开始调用 LLM 生成候选维度...")
            dims = miner.generate_candidate_dimensions(sampled_texts)
            print(f"  [DEBUG] LLM 返回了 {len(dims)} 个维度")
            with open(v_cand_path, "w", encoding="utf-8") as f:
                json.dump(dims, f, ensure_ascii=False, indent=2)
            print(f"  [4a-4] 生成候选维度 {len(dims)} 个")

        if not dims:
            return []

        # 迭代优化
        if v_core_path.exists():
            print("  [4a-5] 检测到已有核心维度，跳过优化")
            with open(v_core_path, "r", encoding="utf-8") as f:
                core_dims = json.load(f)
            return core_dims

        print("  [4a-5] 迭代优化维度...")
        verified_dims = set()

        # 减少迭代次数和采样数量以加快速度
        MAX_ITERATIONS = 2
        VALIDATION_SAMPLE_SIZE = 30

        for iteration in range(MAX_ITERATIONS):
            print(f"    Iteration {iteration + 1}/{MAX_ITERATIONS}...")

            extraction_results = {dim: [] for dim in dims if dim not in verified_dims}
            if not extraction_results:
                break

            for text in texts[:VALIDATION_SAMPLE_SIZE]:
                for dim in extraction_results:
                    res = miner.extract_dimension_value(text, dim)
                    if res:
                        extraction_results[dim].append(res)

            dims_to_remove = set()
            new_dims = []

            for dim in list(extraction_results.keys()):
                values = extraction_results[dim]
                cov = len(values) / max(VALIDATION_SAMPLE_SIZE, 1)

                if len(values) > 0:
                    value_counts = Counter(values)
                    probs = [c / len(values) for c in value_counts.values()]
                    dis = entropy(probs)
                else:
                    dis = 0.0

                if cov < 0.20 or dis < 1.0:
                    decision = miner.optimize_dimension(dim, "低覆盖率" if cov < 0.20 else "低辨识度",
                                                        f"覆盖率{cov:.2%}, 熵{dis:.2f}", values[:5])
                    action = decision.get("action", "").upper()
                    if action == "DELETE":
                        dims_to_remove.add(dim)
                    elif action == "KEEP":
                        verified_dims.add(dim)
                    elif action in ("RENAME", "SPLIT", "MERGE"):
                        dims_to_remove.add(dim)
                        new_dims.extend(decision.get("new_dimensions", []))

            dims = [d for d in dims if d not in dims_to_remove]
            for nd in new_dims:
                if nd and nd not in dims:
                    dims.append(nd)

        core_dims = list(dims)
        with open(v_core_path, "w", encoding="utf-8") as f:
            json.dump(core_dims, f, ensure_ascii=False, indent=2)

        return core_dims

    def _load_all_chunks_from_rdb(self, limit: int = 10000) -> List[Dict]:
        """从 MySQL 加载所有 chunk"""
        conn = self._connect_rdb()
        cur = conn.cursor()
        try:
            database = self.config.rdb.database
            table = self.config.rdb.table
            cur.execute(f"USE `{database}`")

            cur.execute(f"SELECT doc_id, doc_text FROM `{table}` LIMIT %s", (limit,))
            rows = cur.fetchall()

            return [{"doc_id": row[0], "doc_text": row[1]} for row in rows]
        except Exception as e:
            print(f"  [Error] 加载失败: {e}")
            return []
        finally:
            cur.close()
            conn.close()'''

new_dimension_mining = '''    def _dimension_mining(self) -> List[str]:
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
        return [{"doc_id": d["id"], "doc_text": d["text"]} for d in docs[:limit]]'''

content = content.replace(old_dimension_mining, new_dimension_mining)

# ========================================
# 3. 替换 _add_dimension_columns 方法
# ========================================
old_add_columns = '''    def _add_dimension_columns(self, dims: List[str], max_values_per_dim: int = 10) -> Dict[str, int]:
        """添加维度列到 MySQL - 每个维度值单独一列
        
        Args:
            dims: 维度列表
            max_values_per_dim: 每个维度最多支持的值数量（列数）
        
        Returns:
            创建的列数统计
        """
        conn = self._connect_rdb()
        cur = conn.cursor()
        try:
            database = self.config.rdb.database
            table = self.config.rdb.table
            cur.execute(f"USE `{database}`")

            # 检查现有列
            cur.execute(f"SHOW COLUMNS FROM `{table}`")
            existing_cols = {row[0] for row in cur.fetchall()}

            total_cols = 0
            for dim in dims:
                # 直接使用维度名（保留中文），只替换空格为下划线
                safe_dim = dim.replace(" ", "_")
                # 每个维度创建多列：dim_维度名_1, dim_维度名_2, ...
                for i in range(max_values_per_dim):
                    col_name = f"dim_{safe_dim}_{i + 1}"
                    if col_name not in existing_cols:
                        sql = f"ALTER TABLE `{table}` ADD COLUMN `{col_name}` TEXT NULL"
                        cur.execute(sql)
                        total_cols += 1
                        if total_cols <= 20:
                            print(f"    添加列: {col_name}")

            if total_cols > 20:
                print(f"    ... 共添加 {total_cols} 个维度列")

            conn.commit()
            return {"total_columns": total_cols}
        except Exception as e:
            conn.rollback()
            print(f"  [Error] 添加维度列失败: {e}")
            return {"error": str(e)}
        finally:
            cur.close()
            conn.close()'''

new_add_columns = '''    def _add_dimension_columns(self, dims: List[str], max_values_per_dim: int = 10) -> Dict[str, int]:
        """添加维度列到 MySQL - 使用 dimension_integration 的逻辑
        注意: dimension_integration 使用单列模式 (dim_维度名)，而不是多列模式
        """
        # 确保数据库存在
        ensure_rdb_db()

        # 获取现有列
        table = self.config.rdb.table
        existing_cols = fetch_rdb_columns(table)

        conn = connect_rdb()
        cur = conn.cursor()

        added = []
        skipped = []

        for dim in dims:
            col_name = normalize_attr_key(dim)
            if col_name in existing_cols:
                skipped.append(col_name)
                continue

            try:
                sql = f'ALTER TABLE `{table}` ADD COLUMN `{col_name}` TEXT NULL'
                cur.execute(sql)
                conn.commit()
                added.append(col_name)
                if len(added) <= 20:
                    print(f"    + 添加列: {col_name}")
            except Exception as e:
                print(f"    [Error] 添加 {col_name} 失败: {e}")

        cur.close()
        conn.close()

        if len(added) > 20:
            print(f"    ... 共添加 {len(added)} 个维度列")
        print(f"    跳过 {len(skipped)} 个已存在的列")

        return {"total_columns": len(added)}'''

content = content.replace(old_add_columns, new_add_columns)

# ========================================
# 4. 替换 _generate_tags 方法
# ========================================
old_generate_tags = '''    def _generate_tags(self, dims: List[str]) -> Dict[str, Dict[str, List[str]]]:
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

        processed = 0
        for i, chunk in enumerate(chunks):
            doc_id = chunk["doc_id"]
            if doc_id in results:
                continue

            text = chunk["doc_text"]
            if len(text) < 10:
                results[doc_id] = {}
                continue

            try:
                extracted = miner.extract_batch_dimensions(text, dims)
                doc_tags = {}
                if extracted:
                    for dim, val_list in extracted.items():
                        if val_list and len(val_list) > 0:
                            doc_tags[dim] = val_list

                if not doc_tags:
                    keywords = miner.extract_keywords_fallback(text)
                    if keywords:
                        doc_tags["关键词"] = keywords

                results[doc_id] = doc_tags
            except Exception as e:
                print(f"    [Error] {doc_id}: {e}")
                results[doc_id] = {}

            processed += 1
            if processed % 10 == 0:
                print(f"    进度: {processed}/{len(chunks)}")

            if processed % 50 == 0:
                with open(tags_output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        with open(tags_output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"    生成完成，共 {len(results)} 条标签")
        return results'''

new_generate_tags = '''    def _generate_tags(self, dims: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """LLM 批量生成标签 - 使用 dimension_integration 的 TagGenerator"""
        print("    初始化 TagGenerator...")
        generator = TagGenerator(dims)

        print("    开始生成标签...")
        tags = generator.run()

        print(f"    生成完成，共 {len(tags)} 条标签")
        return tags'''

content = content.replace(old_generate_tags, new_generate_tags)

# ========================================
# 5. 替换 _write_tags_to_rdb 方法
# ========================================
# 找到 _write_tags_to_rdb 方法
write_rdb_start = content.find('    def _write_tags_to_rdb(self')
if write_rdb_start == -1:
    print("WARNING: _write_tags_to_rdb not found")
else:
    # 找到下一个方法或类定义
    next_def = content.find('\n    def ', write_rdb_start + 10)
    next_class = content.find('\nclass ', write_rdb_start + 10)
    method_end = min(x for x in [next_def, next_class, len(content)] if x > write_rdb_start)

    old_write_rdb = content[write_rdb_start:method_end]

    new_write_rdb = '''    def _write_tags_to_rdb(self, tags_result: Dict[str, Dict[str, List[str]]]) -> Dict[str, int]:
        """将标签写入 MySQL - 使用 dimension_integration 的逻辑"""
        if not tags_result:
            print("    无标签数据，跳过写入")
            return {"updated": 0}

        # 直接使用 run_step4 的逻辑
        import mysql.connector

        # 确保数据库存在
        ensure_rdb_db()

        config = self.config
        table = config.rdb.table
        existing_cols = fetch_rdb_columns(table)

        # 读取维度
        if PATH_V_CORE.exists():
            with open(PATH_V_CORE, "r", encoding="utf-8") as f:
                dims = json.load(f)
        else:
            dims = []

        # 构建列名映射
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

                    for dim, col in dim_cols.items():
                        if col not in existing_cols:
                            continue
                        val = doc_tags.get(dim)
                        if val is not None:
                            if isinstance(val, list):
                                val = ",".join(str(v) for v in val)
                            elif not isinstance(val, str):
                                val = str(val)
                            label_cols.append(col)
                            label_vals.append(val)

                    if not label_cols:
                        continue

                    sql = f"UPDATE `{table}` SET "
                    sql += ", ".join(f"`{col}` = %s" for col in label_cols)
                    sql += f", `updated_at` = NOW()"
                    sql += f" WHERE `doc_id` = %s"

                    cur.execute(sql, tuple(label_vals + [doc_id]))
                    conn.commit()

                    if cur.rowcount > 0:
                        updated += 1
                    else:
                        failed += 1

                except Exception as e:
                    print(f"    [Error] {doc_id}: {e}")
                    failed += 1

        cur.close()
        conn.close()

        print(f"    写入完成: 成功 {updated} 篇, 失败 {failed} 篇")
        return {"updated": updated, "failed": failed}

'''

    content = content[:write_rdb_start] + new_write_rdb + content[method_end:]

# 保存修改后的文件
with open(r'd:\RAG_DB_slim\pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done: 已将 dimension_integration.py 的逻辑集成到 pipeline.py")
print("\n主要变更:")
print("1. _dimension_mining() -> 使用 DimensionMiner 类")
print("2. _add_dimension_columns() -> 使用 normalize_attr_key 和 ALTER TABLE")
print("3. _generate_tags() -> 使用 TagGenerator 类")
print("4. _write_tags_to_rdb() -> 使用 fetch_rdb_columns, normalize_attr_key 等")
