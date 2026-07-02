"""yaml做配置k:v"""
#用于加载配置参数
import yaml
from dotenv import load_dotenv
from utils.path_tool import get_abs_path

# 在读取任何配置前加载 .env 文件，确保环境变量（DASHSCOPE_API_KEY、DEEPSEEK_API_KEY、DOUBAO_SEED_API_KEY、LANGSMITH_* 等）已就绪
load_dotenv(get_abs_path(".env"))

def laod_model_config(config_path:str = get_abs_path("config/model.yml"),encoding:str = "utf-8"):
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)
    
def laod_chroma_config(config_path:str = get_abs_path("config/chroma.yml"),encoding:str = "utf-8"):
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

def laod_prompts_config(config_path:str = get_abs_path("config/prompts.yml"),encoding:str = "utf-8"):
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

def laod_agent_config(config_path:str = get_abs_path("config/agent.yml"),encoding:str = "utf-8"):
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

def laod_eval_config(config_path:str = get_abs_path("config/eval.yml"),encoding:str = "utf-8"):
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

model_conf = laod_model_config()
chroma_conf = laod_chroma_config()
prompts_conf = laod_prompts_config()
agent_conf = laod_agent_config()
eval_conf = laod_eval_config()

if __name__ == "__main__":
    print(chroma_conf["md5_hex_store"])
