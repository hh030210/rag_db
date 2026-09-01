# -*- coding: utf-8 -*-
import sys, io, json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

data = json.load(open('D:/RAG_DB_slim/experiment_data/dimension_metadata.json', encoding='utf-8'))

print(f"==== 共 {len(data)} 个维度 ====\n")

for name, info in sorted(data.items(), key=lambda x: x[1]['value_count'], reverse=True):
    enum_mark = " [枚举]" if info.get("is_enum", False) else ""
    print(f"【{name}】{info['value_count']} 个值{enum_mark}")
    values = info['values']
    # 只显示前 30 个 + 后 5 个，中间用 ... 省略
    if len(values) <= 35:
        for v in values:
            print(f"  - {v}")
    else:
        for v in values[:30]:
            print(f"  - {v}")
        print(f"  ... (中间省略 {len(values) - 35} 个值) ...")
        for v in values[-5:]:
            print(f"  - {v}")
    print()
