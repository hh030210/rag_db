import os
import json
from llm_service import DimensionMiningWithQwen

class QueryParser:
    def __init__(self, cache_file="./experiment_data/query_cache.json"):
        self.miner = DimensionMiningWithQwen()
        self.cache_file = cache_file
        self.cache = {}
        self._load_cache()
        
    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)
    
    def _save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def parse(self, qid, query_text, dim_meta, schema_dim_fields=None):
        """
        解析 Query，优先读缓存。

        :param dim_meta: IndexBuilder 生成的维度元数据
        :param schema_dim_fields: Milvus schema 中的实际 dim_xxx 字段名列表
                               用于约束 LLM 输出使用正确的维度名
        """
        qid = str(qid)

        # 1. 读缓存
        if qid in self.cache:
            return self.cache[qid]

        # 2. 构造动态配置 (区分 Enum 和 Open)
        enum_map = {}
        open_dims = []

        for dim, meta in dim_meta.items():
            if meta['is_enum']:
                enum_map[dim] = meta['values']
            else:
                open_dims.append(dim)

        # 3. 调用 LLM
        all_dims = list(enum_map.keys()) + open_dims

        parsed_result = self.miner.parse_query_intent(
            query_text, all_dims,
            enum_values_map=enum_map,
            schema_dim_fields=schema_dim_fields
        )

        # 4. 写入缓存
        self.cache[qid] = parsed_result
        self._save_cache()

        return parsed_result