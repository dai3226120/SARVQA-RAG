# agent-text-internvl.py 修改说明

## 修改概述
已将代码修改为直接调用 OpenGVLab/InternVL2-8B 模型，通过 vLLM OpenAI 兼容 API 进行推理。

## 主要修改内容

### 1. **导入修改**
- ❌ 移除: `MainAgent` 相关导入（dotenv, langchain, agent 导入）
- ✅ 新增: `from openai import OpenAI` - 用于调用 vLLM API

### 2. **配置修改**
添加了新的 vLLM API 配置常量：
```python
VLLM_API_BASE = "http://localhost:8080/v1"  # vLLM API 地址
VLLM_MODEL = "OpenGVLab/InternVL2-8B"       # 模型名称
VLLM_MAX_TOKENS = 512                        # 最大生成 token 数
VLLM_TEMPERATURE = 0.7                       # 温度
```

### 3. **初始化函数修改**
- ❌ 移除: `init_agent()` 函数
- ✅ 新增: `init_client()` 函数
  - 初始化 OpenAI 兼容客户端
  - 指向本地 vLLM 服务

### 4. **核心推理函数修改**
完全重写了 `call_chat_api()` 函数：

**新流程：**
1. 检查图像文件是否存在
2. 初始化 OpenAI 客户端
3. 读取并转换图像为 base64 格式
4. 识别图像格式（jpeg/png）
5. 调用 vLLM API 进行推理
6. 提取并清理回复文本
7. 返回处理后的答案

**关键特性：**
- 支持 base64 编码图像传输
- 自动识别图像格式
- 集成提示词模板
- 异常处理和日志记录

### 5. **主函数修改**
- 更新打印信息，反映使用 OpenGVLab/InternVL2-8B 模型
- 调用 `init_client()` 替代 `init_agent()`

## 配置要求

### 1. **vLLM 服务要求**
确保 vLLM 服务已启动，命令如下：
```bash
python -m vllm.entrypoints.openai.api_server \
    --model OpenGVLab/InternVL2-8B \
    --port 8080 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code \
    --max-model-len 4096
```

### 2. **Python 依赖**
确保已安装必要的包：
```bash
pip install openai pandas requests
```

## 使用方法

### 1. **启动 vLLM 服务**
在服务器端启动 vLLM API 服务（已启动，端口 8080）

### 2. **验证连接**
运行测试脚本验证与 vLLM 的连接：
```bash
python benchmark/test_vllm_connection.py
```

### 3. **运行主程序**
```bash
# 使用默认配置处理前 200 条数据
python benchmark/agent-text-internvl.py

# 或在 Python 中运行
python -m benchmark.agent-text-internvl
```

## 配置参数调整

### 在文件中修改以下常量：
- `MAX_PROCESS_ROWS`: 处理的最大行数（默认 200）
- `MAX_WORKERS`: 并发线程数（默认 50）
- `START_ROW`: 起始行（默认 0）
- `VLLM_MAX_TOKENS`: 模型生成的最大 token 数
- `VLLM_TEMPERATURE`: 生成文本的温度参数

## 输出文件

程序会生成两个 CSV 文件（在 `./benchmark/result/agent-text/` 目录）：
1. **时间戳文件**: `agent-text-doubao-seed-20-mini_predicted_question_YYYYMMDD_HHMMSS.csv`
2. **最新文件**: `agent-text-doubao-seed-20-mini_predicted_question_latest.csv`

## 故障排查

### ❌ 连接被拒绝错误
- 检查 vLLM 服务是否启动
- 确认端口 8080 是否正确
- 检查防火墙设置

### ❌ 模块导入错误
- 安装 openai: `pip install openai`
- 检查 Python 版本（3.8+）

### ❌ 图像文件不存在
- 确认 `IMAGE_BASE_PATH` 配置正确
- 检查图像文件权限

## 性能指标

- **处理速度**: 取决于 vLLM 服务和网络延迟
- **并发数**: 可根据硬件和 API 限流调整 `MAX_WORKERS`
- **生成长度**: 通过 `VLLM_MAX_TOKENS` 控制
