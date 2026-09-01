import pandas as pd

# 读取TSV文件前200行
df = pd.read_csv('i:\\毕业论文最新版\\Code\\chapter3\\datasets\\wikipedia\\psgs_w100.tsv', sep='\t', nrows=200)

# 显示前200行
print(df.to_string())
print(f"\n总行数: {len(df)}")
print(f"\n列名: {df.columns.tolist()}")
