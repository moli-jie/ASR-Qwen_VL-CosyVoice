# multimodal_app/main_app.py

import time
import threading
import os
import asyncio
from multimodal_app.audio_utils import AudioRecorder, AudioPlayer
from multimodal_app.video_utils import VideoRecorder
from multimodal_app.media_utils import MediaSaver
from multimodal_app.asr_utils import QwenAudioASR
from multimodal_app.vl_inference import MultimodalInference
import dashscope
from multimodal_app.config import dashscope_api_key
dashscope.api_key = dashscope_api_key # 从 config.py 中加载 dashscope API Key

class MainApp:
    """主应用类，负责协调各个模块"""
    def __init__(self):
        self.audio_recorder = AudioRecorder()
        self.video_recorder = VideoRecorder()
        self.media_saver = MediaSaver()
        self.audio_player = AudioPlayer()
        self.qwen_audio_asr = QwenAudioASR() # 添加 QwenAudioASR 语音识别器
        self.multimodal_inference = MultimodalInference()
        self.is_recording = False
        self.prompt_text=""

    def start_dialog(self):
        if not self.is_recording:
            self.is_recording = True
            self.audio_recorder.start_recording()
            self.video_recorder.start_recording()
            print("对话开始...")
        else:
            self.stop_dialog()

    def stop_dialog(self):
        if self.is_recording:
            self.is_recording = False
            self.audio_recorder.stop_recording()
            self.video_recorder.stop_recording()
            print("对话结束，保存音频视频...")
            audio_data = self.audio_recorder.get_audio_data()
            video_frames = self.video_recorder.get_video_frames()
            audio_output_path, video_output_path = self.media_saver.save_media(audio_data, video_frames)
            self.prompt_text = self.qwen_audio_asr.call_qwen_audio_asr_api(audio_output_path) # 使用 QwenAudioASR 语音识
            if video_output_path:
                # 启动新的线程来执行推理和语音合成
                threading.Thread(target=self.process_inference, args=(audio_output_path, video_output_path)).start()

    def process_inference(self, audio_output_path, video_output_path):
        """处理推理流程 (现在在单独的线程中运行)"""
        print("--- 推理线程已启动 ---") # 添加线程启动的标识

        if self.prompt_text:
            print("ASR 输出:", self.prompt_text)
            output_text = self.multimodal_inference.run_inference(audio_output_path, video_output_path, self.prompt_text)
            if output_text:
                output_audio_file = os.path.join("./Test_QWen2_VL/", f"sft_{self.media_saver.audio_file_count}.mp3") # 注意路径修改
                asyncio.run(self.audio_player.synthesize_and_play(output_text, output_audio_file)) # 调用新的 CosyVoice TTS 合成和播放方法
                print("--- 推理线程已结束 ---") # 添加线程结束的标识
            else:
                print("--- 推理线程已结束 ---")
        else:
            print("语音识别失败，跳过后续处理。")
            print("--- 推理线程已结束 ---") # 即使推理失败，也标识线程结束


if __name__ == "__main__":
    app = MainApp()
    print("按 Enter 开始/结束对话...")
    while True:
        input()
        app.start_dialog()