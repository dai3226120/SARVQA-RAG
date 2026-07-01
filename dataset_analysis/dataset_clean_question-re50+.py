import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.path_tool import get_abs_path

# 1. 读取问题统计文件，找出需要删除的问题（出现次数 > 50）
df_stats = pd.read_csv(get_abs_path('utils/dataset_analysis_result/问题-出现的数据集.csv'))
# 获取出现总次数大于50的问题集合
questions_to_remove = set(df_stats[df_stats['出现总次数'] > 50]['问题'])

# 2. 读取主数据集
df_main = pd.read_csv(get_abs_path('SAR-VQA_ALL-246980_utf8.csv'))

# 3. 数据清洗：保留不在删除列表中的行
# ~ 表示取反，isin() 检查是否在集合中
df_filtered = df_main[~df_main['question'].isin(questions_to_remove)]

# 4. 保存结果到新文件
df_filtered.to_csv(get_abs_path('SAR-VQA_ALL_filtered_question-re50+.csv'), index=False, encoding='utf-8')

# 输出统计信息
print(f"处理前总行数: {len(df_main)}")
print(f"删除的问题种类数: {len(questions_to_remove)}")
print(f"处理后剩余行数: {len(df_filtered)}")