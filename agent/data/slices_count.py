import pandas as pd

# 1. 读取 CSV 文件
df = pd.read_csv('./agent/data/sar_slices.csv')

# 2. 定义正则表达式 (加上 \b 单词边界，确保严格匹配 1-3 位数字)
pattern = r'\bcluster_\d{1,3}_slice_\d{1,3}\b'

# 3. 使用 str.contains() 进行正则匹配查询
# na=False 表示如果该列有空值(NaN)，则不将其视为匹配
matched_df = df[df['slice_id'].astype(str).str.contains(pattern, regex=True, na=False)]

# 4. 获取并打印相关的数量
count = len(matched_df)
print(f"符合 'cluster_[1-3位数字]_slice_[1-3位数字]' 规则的数量为: {count}")

# （可选）如果你想查看具体匹配到的数据前几行，可以取消下面这行的注释：
# print(matched_df.head())