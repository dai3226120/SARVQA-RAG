"""
指标计算模块
包含信息增益度、信息密度等指标的计算
"""

import os
import re
from agent.utils.text_utils import tokenize_text
from agent.utils.thread_lock import lock


# ====================== 领域词汇表加载 ======================
def load_domain_terms(terms_file_path):
    """加载领域词汇表，返回去重后的词汇集合"""
    terms = set()
    try:
        with open(terms_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                term = line.strip()
                if term and not term.startswith('['):
                    terms.add(term.lower())
    except Exception as e:
        print(f"加载领域词汇表失败: {e}")
    return terms


def build_domain_regex(domain_terms):
    """构建领域词汇匹配正则表达式，长短语优先匹配"""
    if not domain_terms:
        return None
    sorted_terms = sorted(domain_terms, key=len, reverse=True)
    escaped_terms = [re.escape(term) for term in sorted_terms]
    pattern = r'\b(' + '|'.join(escaped_terms) + r')\b'
    return re.compile(pattern, re.IGNORECASE)


def count_domain_terms(text, domain_regex):
    """统计文本中匹配的领域词汇数量（去重）"""
    if not text or not domain_regex:
        return 0
    matches = domain_regex.findall(text.lower())
    return len(set(match.lower() for match in matches))


# 领域词汇表路径（相对于项目根目录）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DOMAIN_TERMS_FILE = os.path.join(_PROJECT_ROOT, 'dataset_analysis', 'sar_remote_sensing_terms_batch.txt')
_DOMAIN_TERMS = load_domain_terms(_DOMAIN_TERMS_FILE)
_DOMAIN_REGEX = build_domain_regex(_DOMAIN_TERMS)


# ====================== 停用词列表（保留用于兼容）======================
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
    'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'it', 'its',
    'as', 'from', 'into', 'about', 'if', 'then', 'but', 'also', 'be', 'being', 'been',
    'can', 'could', 'would', 'should', 'may', 'might', 'will', 'shall',
    'has', 'have', 'had', 'do', 'does', 'did', 'not', 'no', 'so', 'such', 'than',
    'too', 'very', 'more', 'most', 'some', 'any', 'all', 'their', 'there',
    'when', 'where', 'who', 'what', 'which', 'while', 'during', 'among', 'were',
    'each', 'other', 'per', 'through', 'over', 'under', 'between', 'without',
    'within', '对于', '和', '与', '也', '有', '是', '在', '的', '了', '不', '或',
    '及', '且', '这', '那', '其', '被', '为'
}


# ====================== 信息指标计算 ======================
def compute_information_metrics(question, prompt_text):
    """
    计算信息增益度(IG)和信息密度(ID)
    基于遥感领域专业词汇计算，衡量增强型Prompt对原始问题的领域知识补充效果

    参数:
        question: 问题文本
        prompt_text: 提示词文本

    返回:
        tuple: (IG值, ID值)
    """
    q_domain_count = count_domain_terms(question, _DOMAIN_REGEX)
    p_domain_count = count_domain_terms(prompt_text, _DOMAIN_REGEX)
    raw_prompt_tokens = tokenize_text(prompt_text)

    ig = 0.0
    if q_domain_count > 0:
        ig = (p_domain_count - q_domain_count) / q_domain_count

    id_value = 0.0
    if len(raw_prompt_tokens) > 0:
        id_value = p_domain_count / len(raw_prompt_tokens)

    return round(ig, 4), round(id_value, 4)


# ====================== 检索指标管理 ======================
class RetrievalMetrics:
    """检索指标管理器"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有指标"""
        self.information_gain_values = []
        self.information_density_values = []

    def add_ig_id(self, ig_value, id_value):
        """添加IG和ID值"""
        with lock:
            self.information_gain_values.append(ig_value)
            self.information_density_values.append(id_value)

    def update_from_response(self, raw_text):
        """从响应文本中提取并更新指标"""
        pass

    def format_summary(self, agent_client=None):
        """格式化指标摘要"""
        summary = []

        # 隶属度命中率（从 agent_client 获取真正的隶属度命中率）
        membership_hit_rate = None
        if agent_client and hasattr(agent_client, 'get_rag_rscsv_membership_hit_rate'):
            membership_hit_rate = agent_client.get_rag_rscsv_membership_hit_rate()
        
        if membership_hit_rate is not None:
            summary.append(f"rag_rscsv 隶属度命中率: {membership_hit_rate:.2%}")
        else:
            summary.append("隶属度命中率: 未检测到隶属度计算")

        # 检索耗时统计
        if agent_client and hasattr(agent_client, 'get_retrieval_latency_stats'):
            latency_stats = agent_client.get_retrieval_latency_stats()
            summary.append(f"检索工具调用次数: {latency_stats['call_count']}")
            summary.append(f"检索总耗时: {latency_stats['total_latency']:.4f}秒")
            summary.append(f"平均检索耗时: {latency_stats['avg_latency']:.4f}秒/次")

        # 信息增益度
        ig_values = self.information_gain_values
        avg_ig = sum(ig_values) / len(ig_values) if ig_values else None
        summary.append(f"信息增益度 IG: {avg_ig:.4f}" if avg_ig is not None else "信息增益度 IG: N/A")

        # 信息密度
        id_values = self.information_density_values
        avg_id = sum(id_values) / len(id_values) if id_values else None
        summary.append(f"信息密度 ID: {avg_id:.4f}" if avg_id is not None else "信息密度 ID: N/A")

        return summary


# 全局检索指标实例
retrieval_metrics = RetrievalMetrics()


# ====================== 评估指标计算（用于 benchmark）======================
def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """计算两个文本的余弦相似度"""
    if not text1 or not text2:
        return 0.0

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words='english',
            min_df=1
        )

        tfidf_matrix = vectorizer.fit_transform([text1.lower(), text2.lower()])

        if tfidf_matrix.nnz == 0:
            return 0.0

        return round(float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]), 4)

    except Exception as e:
        print(f"计算余弦相似度错误: {str(e)}")
        return 0.0


def calculate_rouge_l(text1: str, text2: str) -> float:
    """计算ROUGE-L分数"""
    if not text1 or not text2:
        return 0.0

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(text1, text2)
        return scores['rougeL'].fmeasure
    except Exception as e:
        print(f"计算ROUGE-L错误: {str(e)}")
        return 0.0


def calculate_bleu_scores(reference: str, candidate: str) -> dict:
    """计算BLEU-1到BLEU-4分数"""
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    results = {'BLEU1': 0.0, 'BLEU2': 0.0, 'BLEU3': 0.0, 'BLEU4': 0.0}

    if not reference or not candidate:
        return results

    try:
        pred_tokens = candidate.lower().split()
        ans_tokens = reference.lower().split()
        ans_tokens_list = [ans_tokens]

        smoothie = SmoothingFunction().method1
        weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]

        for i, w in enumerate(weights):
            results[f'BLEU{i + 1}'] = round(sentence_bleu(ans_tokens_list, pred_tokens, weights=w, smoothing_function=smoothie), 4)

        return results

    except Exception as e:
        print(f"计算BLEU错误: {str(e)}")
        return results


def calculate_meteor_score(reference: str, candidate: str) -> float:
    """计算METEOR分数"""
    import nltk
    from nltk.translate.meteor_score import meteor_score

    if not reference or not candidate:
        return 0.0

    try:
        pred_tokens = candidate.lower().split()
        ans_tokens = reference.lower().split()
        return round(meteor_score([ans_tokens], pred_tokens), 4)
    except Exception as e:
        print(f"计算METEOR错误: {str(e)}")
        return 0.0


def calculate_all_metrics(predicted: str, answer: str) -> dict:
    """计算所有评估指标"""
    results = {
        'cosine': calculate_cosine_similarity(predicted, answer),
        'ROUGEL': calculate_rouge_l(predicted, answer),
    }
    results.update(calculate_bleu_scores(answer, predicted))
    results['METEOR'] = calculate_meteor_score(answer, predicted)
    return results
