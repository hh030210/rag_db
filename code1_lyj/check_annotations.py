import pandas as pd

# 读取parquet文件查看annotations结构
df = pd.read_parquet(r'i:\毕业论文最新版\Code\chapter3\datasets\wikipedia\natural_questions\validation-00000-of-00007.parquet')

# 查看第一行的annotations
first_row = df.iloc[0]
print("=" * 80)
print("第一行的Annotations结构:")
print("=" * 80)
annotations = first_row['annotations']
print(f"类型: {type(annotations)}")
print(f"内容: {annotations}")

print("\n" + "=" * 80)
print("第一行的Long Answer Candidates结构:")
print("=" * 80)
candidates = first_row['long_answer_candidates']
print(f"类型: {type(candidates)}")
if isinstance(candidates, dict):
    for key, value in candidates.items():
        print(f"  {key}: {type(value)}, 长度: {len(value) if hasattr(value, '__len__') else 'N/A'}")
