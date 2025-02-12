# multimodal_app/vl_inference.py

import os
import cv2
import base64
from openai import OpenAI
from multimodal_app.config import API_KEY_, FOLDER_PATH

class MultimodalInference:
    """多模态推理类，调用 Qwen-VL 模型"""
    def __init__(self, openai_api_key=API_KEY_, openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", folder_path=FOLDER_PATH):
        self.client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
        self.folder_path = folder_path

    def encode_image(self, image_path):
        """Base64 编码函数"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def capture_video_frames(self, video_path, frame_indices_ratio=[0.2, 0.4, 0.6, 0.8]):
        """从视频中截取指定比例的帧"""
        captured_images = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return captured_images # 返回空列表

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = [int(total_frames * ratio) for ratio in frame_indices_ratio]

        for idx in frame_indices:
            if idx >= total_frames:
                print(f"帧索引 {idx} 超出范围，视频总帧数为 {total_frames}")
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                print(f"无法读取帧索引 {idx}")
                continue

            file_path = os.path.join(self.folder_path, f"captured_image{idx}.jpg")
            cv2.imwrite(file_path, frame)
            captured_images.append(file_path)
            print(f"帧保存至 {file_path}")
        cap.release()
        return captured_images

    def run_inference(self, audio_output_path, video_path, prompt_text):
        """执行推理"""
        captured_images_paths = self.capture_video_frames(video_path)
        if not captured_images_paths:
            print("未成功截取任何视频帧，推理可能效果不佳。")

        base64_images = [f"data:image/jpeg;base64,{self.encode_image(img_path)}" for img_path in captured_images_paths]
        video_content = {"type": "video", "video": base64_images, "fps": 1.0}
        messages = [{"role": "user", "content": [video_content, {"type": "text", "text": prompt_text}]}]

        try:
            completion = self.client.chat.completions.create(model="qwen-vl-max-latest", messages=messages)
            output_text = completion.choices[0].message.content
            print("模型输出:", output_text)
            return output_text
        except Exception as e:
            print(f"调用模型失败: {e}")
            return None