"""
vLLM 部署模型客户端（tinygptv / model-2 / model-3）
- tinygptv: TinyGPT-V（vLLM 部署；公网端点与 InternVL 相同 i-2.gpushare.com:31867，
  服务端内部为 localhost:8080，发布时由服务端切换映射）
- model-2 / model-3: 占位新模型（与 InternVL 同服务器同端口，模型在服务端手动切换）

⚠️ 用法 / 改名（发布正式模型时）：
    1. 把 model-2/3（或 tinygptv）替换为正式模型名（客户端实例名 + label）
    2. 同步修改（搜索对应模型名即可定位）：
       - config/model.yml             模型名/端点等配置
       - config/eval.yml              model_types / file_tags
       - config/__init__.py           ModelType 常量
       - model/factory.py             工厂实例名与配置前缀
       - benchmark/models/__init__.py 导出
       - benchmark/main_eval.py       _MODEL_REGISTRY 条目与 MODEL_KEY 注释
"""
from model.factory import tinygptv_model, model_2_model, model_3_model
from .base_model import BaseAPIClient


class VLLMAPIClient(BaseAPIClient):
    """vLLM 部署模型通用客户端（tinygptv / model-2/3）"""

    def __init__(self, model_instance, model_label="vllm"):
        super().__init__(model_instance, model_label=model_label)


class TinyGPTVAPIClient(VLLMAPIClient):
    """TinyGPT-V 专用客户端：视觉轮直接发送裸问题

    Stage4 训练/推理格式为 `###<s>[INST] <Img><ImageHere></Img> {q}### [/INST]`：
    - [INST] 包装与图片占位符由服务端 chat template 完成（实测裸问题+图片
      prompt_tokens = 35+34，包装 ~17 token、图片 34 token 均已注入），客户端无需自包；
    - 因此不再套用 DEFAULT_PROMPT 的英文指令文本，避免 [INST] 内出现训练时未见的内容
    """
    def _build_formatted_prompt(self, question, prompt_template=None):
        return question


tinygptv_client = TinyGPTVAPIClient(tinygptv_model, model_label="tinygptv")
model_2_client = VLLMAPIClient(model_2_model, model_label="model-2")
model_3_client = VLLMAPIClient(model_3_model, model_label="model-3")
