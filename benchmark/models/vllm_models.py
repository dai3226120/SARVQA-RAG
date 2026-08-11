"""
vLLM / OpenAI 兼容部署模型客户端（tinygptv / tinygptv-stage4 / geochat / skyeyegpt）
- tinygptv: TinyGPT-V/SAR-GPT（vLLM 部署；公网端点与 InternVL 相同 i-2.gpushare.com:31867，
  服务端内部为 localhost:8080，发布时由服务端切换映射）
- tinygptv-stage4: 官方 Stage4（[INST] 模板，同服务端）
- geochat: GeoChat-7B（LLaVA-1.5 架构，同服务端；部署见《GeoChat部署指南》）
- skyeyegpt: SkyEyeGPT（MiniGPT-v2 架构，OpenAI 兼容 API 服务端，同端点；
  部署见《SkyEyeGPT部署指南》）

⚠️ 用法 / 改名（发布正式模型时）：
    1. 把占位名替换为正式模型名（客户端实例名 + label）
    2. 同步修改（搜索对应模型名即可定位）：
       - config/model.yml             模型名/端点等配置
       - config/eval.yml              model_types / file_tags
       - config/__init__.py           ModelType 常量
       - model/factory.py             工厂实例名与配置前缀
       - benchmark/models/__init__.py 导出
       - benchmark/main_eval.py       _MODEL_REGISTRY 条目与 MODEL_KEY 注释
"""
from model.factory import tinygptv_model, tinygptv_stage4_model, geochat_model, skyeyegpt_model
from .base_model import BaseAPIClient


class VLLMAPIClient(BaseAPIClient):
    """vLLM/OpenAI 兼容部署模型通用客户端（tinygptv / tinygptv-stage4 / geochat / skyeyegpt）"""

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


class GeoChatAPIClient(VLLMAPIClient):
    """GeoChat-7B 专用客户端：视觉轮直接发送裸问题

    GeoChat（LLaVA-1.5 架构）服务端 chat template 渲染 USER: <image>\n{q} ASSISTANT:，
    {q} 位置为裸问题本身（部署指南示例均为裸问题），不套 DEFAULT_PROMPT 英文指令。
    """

    def _build_formatted_prompt(self, question, prompt_template=None):
        return question


class SkyEyeGPTAPIClient(VLLMAPIClient):
    """SkyEyeGPT 专用客户端：视觉轮直接发送裸问题

    SkyEyeGPT（MiniGPT-v2 架构）经 OpenAI 兼容 API 服务端推理（非 vLLM，但接口契约一致）；
    服务端负责 [INST] 等模板包装，messages 中 text 为裸问题（部署指南示例即裸问题），
    不套 DEFAULT_PROMPT 英文指令。⚠️ 服务端为串行推理（单锁），并发评测会排队。
    """

    def _build_formatted_prompt(self, question, prompt_template=None):
        return question


tinygptv_client = TinyGPTVAPIClient(tinygptv_model, model_label="tinygptv")
# Stage4（官方检查点，[INST] 模板）：与 tinygptv 同服务端，模型在服务端手动切换；
# 普通模式同样发裸问题（服务端模板负责 [INST] 包装），仅结果目录/文件标签区分
tinygptv_stage4_client = TinyGPTVAPIClient(tinygptv_stage4_model, model_label="tinygptv-stage4")
geochat_client = GeoChatAPIClient(geochat_model, model_label="geochat")
skyeyegpt_client = SkyEyeGPTAPIClient(skyeyegpt_model, model_label="skyeyegpt")
