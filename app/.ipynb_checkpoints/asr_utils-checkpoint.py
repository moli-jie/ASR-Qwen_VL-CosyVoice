# multimodal_app/asr_utils.py

import os
from dashscope import MultiModalConversation

class QwenAudioASR:
    """
    使用 qwen-audio-asr-latest 模型进行语音识别的类
    """
    def call_qwen_audio_asr_api(self, audio_file_path: str) -> str:
        """
        调用 qwen-audio-asr-latest API 进行语音识别。
        """
        audio_file_path_for_api = f"file://{os.path.abspath(audio_file_path)}" # 构建符合API要求的本地文件路径格式
        messages = [
            {
                "role": "user",
                "content": [{"audio": audio_file_path_for_api}],
            }
        ]
        try:
            response = MultiModalConversation.call(model="qwen-audio-asr-latest", messages=messages)
            print("完整 API 响应:", response) # 打印完整的 API 响应，用于调试
            return response["output"]["choices"][0]["message"]["content"][0]["text"]
        except Exception as e:
            print(f"调用 Qwen-Audio-ASR API 异常: {e}")
            return None