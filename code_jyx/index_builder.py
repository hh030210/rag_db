import os
import json
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from collections import defaultdict

class IndexConfig:
    DATA_DIR = "./experiment_data"
    PATH_TAGS_OPT = os.path.join(DATA_DIR, "tags_debug_top10.json")
    
    # 输出文件
    PATH_INVERTED_INDEX = os.path.join(DATA_DIR, "inverted_index_med_D.json") # 标签->文档索引
    PATH_DIM_META = os.path.join(DATA_DIR, "dimension_metadata_med_D.json")   # 维度的元数据(是否枚举, 值域列表)
    PATH_TAG_VECTORS = os.path.join(DATA_DIR, "tag_vectors_med_D.pkl")        # 开放维度的向量索引
    
    # 判定阈值：如果某维度的唯一值数量 <= 50，视为“伪枚举”
    ENUM_THRESHOLD = 0 #现在不考虑枚举类型，都按描述算
    
    # 原始定义的枚举类 (强制枚举)
    FORCE_ENUMS = []

class IndexBuilder:
    def __init__(self):
        print("初始化索引构建器...")
        self.encoder = None # 懒加载
        
    def build(self):
        # 1. 加载优化后的标签数据
        # 格式: {doc_id: {dim: [val1, val2]}}
        with open(IndexConfig.PATH_TAGS_OPT, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
            
        print(f"加载了 {len(doc_data)} 条文档标签。")
        
        # 2. 构建倒排索引 & 统计值域
        # inverted_index: {dim: {val: [doc_id1, doc_id2]}}
        inverted_index = defaultdict(lambda: defaultdict(list))
        dim_value_sets = defaultdict(set)
        
        for doc_id, tags in doc_data.items():
            for dim, vals in tags.items():
                if not vals: continue
                # 兼容处理
                if isinstance(vals, str): vals = [vals]
                
                for v in vals:
                    inverted_index[dim][v].append(doc_id)
                    dim_value_sets[dim].add(v)
                    
        # 3. 维度元数据分析 (区分 Enum vs Open)
        dim_meta = {}
        dims_to_vectorize = [] # 需要做向量索引的维度
        
        print("\n=== 维度属性分析 ===")
        for dim, val_set in dim_value_sets.items():
            val_list = sorted(list(set(val_set)))
            count = len(val_list)
            
            # 判断逻辑：强制枚举 OR 值域较窄
            is_enum = (dim in IndexConfig.FORCE_ENUMS) or (count <= IndexConfig.ENUM_THRESHOLD)
            
            dim_meta[dim] = {
                "is_enum": is_enum,
                "value_count": count,
                "values": val_list if is_enum else [] # 如果是枚举，直接存值域
            }
            
            tag_type = "枚举 (Enum)" if is_enum else "开放 (Open)"
            print(f"维度 [{dim}]: {count} 个值 -> {tag_type}")
            
            if not is_enum:
                dims_to_vectorize.append((dim, val_list))
                
        # 4. 构建开放维度的向量索引 (用于后续 Matcher)
        if dims_to_vectorize:
            print("\n正在构建开放维度的向量索引...")
            current_dir = os.path.dirname(__file__)
            embedding_model_path = os.path.join(current_dir, 'bge-m3')
            self.encoder = BGEM3FlagModel(embedding_model_path, use_fp16=True, device='cuda')
            
            tag_vectors = {} # {dim: {'vals': [], 'vecs': np.array}}
            
            for dim, val_list in dims_to_vectorize:
                print(f"  Encoding {dim} ({len(val_list)} values)...")
                embeddings = self.encoder.encode(val_list, return_dense=True)['dense_vecs']
                tag_vectors[dim] = {
                    'values': val_list,
                    'vectors': embeddings
                }
            
            import pickle
            with open(IndexConfig.PATH_TAG_VECTORS, 'wb') as f:
                pickle.dump(tag_vectors, f)
        
        # 5. 保存结果
        with open(IndexConfig.PATH_INVERTED_INDEX, 'w', encoding='utf-8') as f:
            json.dump(inverted_index, f, ensure_ascii=False)
            
        with open(IndexConfig.PATH_DIM_META, 'w', encoding='utf-8') as f:
            json.dump(dim_meta, f, ensure_ascii=False, indent=2)
            
        print("\n索引构建完成！")

if __name__ == "__main__":
    builder = IndexBuilder()
    builder.build()