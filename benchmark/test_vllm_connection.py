#!/usr/bin/env python3
"""
测试与 vLLM API 的连接和基本功能
"""

from openai import OpenAI
import base64
import os

# vLLM 配置
VLLM_API_BASE = "http://i-2.gpushare.com:24667/v1/"
VLLM_MODEL = "OpenGVLab/InternVL2-8B"

def test_vllm_connection():
    """测试与 vLLM API 的连接"""
    print("=" * 60)
    print("🧪 vLLM 连接测试")
    print("=" * 60)
    
    try:
        # 初始化客户端
        print(f"\n📍 连接地址: {VLLM_API_BASE}")
        print(f"📍 模型名称: {VLLM_MODEL}")
        
        client = OpenAI(
            api_key="not-needed",
            base_url=VLLM_API_BASE
        )
        
        # 测试文本调用（不需要图像）
        print("\n🔄 测试文本调用...")
        response = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "What is the capital of France? Answer in one sentence."
                }
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        print("✅ 文本调用成功！")
        print(f"📝 模型回复: {response.choices[0].message.content}")
        
        # 测试图像调用
        print("\n🔄 测试图像调用...")
        
        # 查找一个测试图像
        test_image_dir = "C:\\dataset/SAR-TEXT"
        test_image_path = None
        
        if os.path.exists(test_image_dir):
            for file in os.listdir(test_image_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    test_image_path = os.path.join(test_image_dir, file)
                    break
        
        if test_image_path and os.path.exists(test_image_path):
            print(f"📷 测试图像: {test_image_path}")
            
            # 读取并编码图像
            with open(test_image_path, 'rb') as img_file:
                image_data = base64.standard_b64encode(img_file.read()).decode('utf-8')
            
            # 获取图像格式
            _, ext = os.path.splitext(test_image_path)
            image_format = ext.lower().lstrip('.')
            if image_format in ['jpg', 'jpeg']:
                image_format = 'jpeg'
            else:
                image_format = 'png'
            
            # 调用带图像的 API
            response = client.chat.completions.create(
                model=VLLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_format};base64,{image_data}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "Briefly describe what you see in this SAR image in one sentence."
                            }
                        ]
                    }
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            print("✅ 图像调用成功！")
            print(f"📝 模型回复: {response.choices[0].message.content}")
        else:
            print("⚠️  未找到测试图像，跳过图像测试")
        
        print("\n" + "=" * 60)
        print("✨ 所有测试通过！vLLM 连接正常")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print("\n请确保:")
        print("1. vLLM 服务已启动（端口 8080）")
        print("2. 已安装 openai Python 包: pip install openai")
        print("3. 网络连接正常")

if __name__ == "__main__":
    test_vllm_connection()
