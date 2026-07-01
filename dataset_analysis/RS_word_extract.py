import pandas as pd
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set

# ==============================================
# 所有可配置参数（集中管理，按需修改）
# ==============================================
# 文件与接口配置
CSV_PATH = "./SAR-VQA-180375.csv"          # CSV文件路径
API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
API_KEY = "e9f2dacd-d0a2-4c9a-ba2a-805ee0b40dcd"  # 请替换为真实API密钥
MODEL_NAME = "doubao-1-5-lite-32k-250115"  # 模型名称
MAX_WORKERS = 50                          # 并行请求数量
BATCH_SIZE = 100                          # 每批合并的answer数量

# 模型生成参数（全部可配置）
MAX_TOKENS = 2048                          # 批量提取需更大输出长度
TEMPERATURE = 0.1                          # 提取任务用低温度保证稳定
TOP_P = 0.9
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0
TIMEOUT = 60                               # 批量请求超时适当延长
MAX_RETRY = 2                              # 单批请求失败重试次数

# 停用词集合
STOP_WORDS = {
    'the','a','an','and','or','is','are','was','were','in','on','at','to','for','of','with','by','this','that','these','those','it','its',
    'as','from','into','about','if','then','but','also','be','being','been','can','could','would','should','may','might','will','shall',
    'has','have','had','do','does','did','not','no','so','such','than','too','very','more','most','some','any','all','their','there',
    'when','where','who','what','which','while','during','among','were','each','other','per','through','over','under','between','without',
    'within','对于','和','与','也','有','是','在','的','了','不','或','及','且','这','那','其','被','为'
}

# 系统提示词：批量提取规则
SYSTEM_PROMPT = """你是遥感领域专业术语提取专家。用户会输入多条用分隔符隔开的遥感问答答案文本，请你从中提取所有与遥感、合成孔径雷达(SAR)、对地观测、图像处理相关的专业词汇、名词短语和技术术语。
严格遵守以下规则：
1. 只提取遥感领域专业术语，剔除普通通用词汇、虚词、口语化表达
2. 对所有文本中的术语做全局去重，不要重复输出
3. 自动剔除冠词、介词、连词等停用词与无意义短词
4. 输出格式：严格返回标准JSON数组，例如 ["backscatter", "speckle patterns", "ship detection"]
5. 不允许输出任何解释文字、前缀、后缀或markdown格式，只输出纯JSON
"""


# ==============================================
# 工具函数
# ==============================================
def split_batches(data_list: List, batch_size: int) -> List[List]:
    """将列表按指定大小切分为多个批次"""
    return [
        data_list[i:i + batch_size]
        for i in range(0, len(data_list), batch_size)
    ]


def extract_terms_batch(answer_batch: List[str]) -> List[str]:
    """
    批量调用大模型：将一批answer合并后一次性提取遥感术语
    """
    # 合并批次文本，用分隔符隔开，避免语义粘连
    merged_text = "\n---\n".join(answer_batch)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": merged_text}
        ],
        # "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "frequency_penalty": FREQUENCY_PENALTY,
        "presence_penalty": PRESENCE_PENALTY,
        "stream": False
    }

    for retry in range(MAX_RETRY + 1):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()

            # 优先按JSON解析，解析失败则按逗号/换行兜底分割
            try:
                terms = json.loads(content)
                if not isinstance(terms, list):
                    raise ValueError("返回格式非数组")
            except (json.JSONDecodeError, ValueError):
                terms = [
                    t.strip().strip('"').strip("'")
                    for t in content.replace('\n', ',').split(',')
                    if t.strip()
                ]

            # 本批次内过滤停用词与空值
            filtered = [
                term.strip()
                for term in terms
                if term.strip() and term.strip().lower() not in STOP_WORDS and len(term.strip()) > 1
            ]
            return filtered

        except Exception as e:
            if retry < MAX_RETRY:
                time.sleep(1 * (retry + 1))  # 指数退避重试
                continue
            print(f"[批次失败] 批次大小: {len(answer_batch)} | 错误: {str(e)}")
            return []


def main():
    # 1. 读取CSV并预处理
    print("正在读取CSV文件...")
    df = pd.read_csv(CSV_PATH)
    # 先对answer去重，避免重复内容浪费token；如需保留原始全部可注释.unique()
    answer_list = df["answer"].dropna().unique().tolist()
    total_answers = len(answer_list)
    print(f"共加载 {len(df)} 条原始数据，去重后剩余 {total_answers} 条独立answer")

    # 2. 切分批次
    batches = split_batches(answer_list, BATCH_SIZE)
    print(f"按每批 {BATCH_SIZE} 条切分，共 {len(batches)} 个批次")
    print(f"并发请求数: {MAX_WORKERS}，开始批量提取...\n")

    all_terms: Set[str] = set()
    start_time = time.time()

    # 3. 并发处理所有批次
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(extract_terms_batch, batch): idx
            for idx, batch in enumerate(batches)
        }

        processed = 0
        for future in as_completed(future_map):
            processed += 1
            batch_terms = future.result()
            all_terms.update(batch_terms)

            # 进度打印
            if processed % 5 == 0 or processed == len(batches):
                print(f"进度: {processed}/{len(batches)} 批 | 累计不重复术语: {len(all_terms)}")

    # 4. 全局兜底二次过滤停用词
    final_terms = {
        term for term in all_terms
        if term.lower() not in STOP_WORDS and len(term) > 1
    }

    # 5. 输出与保存
    cost_time = time.time() - start_time
    print(f"\n===== 处理完成 =====")
    print(f"总耗时: {cost_time:.2f} 秒")
    print(f"最终提取不重复遥感术语数: {len(final_terms)}")

    # 保存到文件（按字典序排序）
    output_path = "./sar_remote_sensing_terms_batch.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for term in sorted(final_terms):
            f.write(f"{term}\n")
    print(f"结果已保存至: {output_path}")

    # 打印前30个示例
    print("\n提取结果示例（前30个）：")
    for i, term in enumerate(sorted(final_terms)[:30], 1):
        print(f"{i:2d}. {term}")


if __name__ == "__main__":
    main()