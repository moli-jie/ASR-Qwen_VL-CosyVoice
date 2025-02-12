# multimodal_app/config.py

import os

# 参数设置
API_KEY_ = 'sk-8fa8947722864fd98926148b4497a25d'  # 替换为您的API密钥
AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
CHUNK = 1024
OUTPUT_DIR = "./output"
FOLDER_PATH = "./Test_QWen2_VL/"
COSYVOICE_MODEL = "cosyvoice-clone-v1"
COSYVOICE_VOICE_ID = "cosyvoice-prefix-920d4c353c9446e692bf1969decfb663" # 使用 cosyvoice 预设的 voice_id
dashscope_api_key = API_KEY_ # 假设dashscope API key 和 OpenAI API Key 相同，如果不同请修改

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FOLDER_PATH, exist_ok=True)