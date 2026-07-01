"""yaml做配置k:v"""
#用于加载配置参数
import yaml
from utils.path_tool import get_abs_path

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
