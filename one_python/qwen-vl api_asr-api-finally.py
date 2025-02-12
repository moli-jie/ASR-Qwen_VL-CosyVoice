import time
import threading
import os
import pygame
import edge_tts
import asyncio
import langid
import requests
import json
import subprocess
import wave
import base64
import asyncio
import langid
import cv2
from openai import OpenAI
from queue import Queue
import dashscope
from dashscope import MultiModalConversation # 导入 dashscope 库
dashscope.api_key = 'sk-8fa8947722864fd98926148b4497a25d'
# --- 配置huggingFace国内镜像 ---
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 参数设置
AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
CHUNK = 1024
OUTPUT_DIR = "./output"
FOLDER_PATH = "./Test_QWen2_VL/"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FOLDER_PATH, exist_ok=True)

class AudioRecorder:
    """音频录制类"""
    def __init__(self, rate=AUDIO_RATE, channels=AUDIO_CHANNELS, chunk=CHUNK):
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        self.recording_active = False
        self.audio_buffer = []
        self.process = None
        self.lock = threading.Lock()
        threading.Thread(target=self._record_audio).start()

    def start_recording(self):
        with self.lock:
            self.recording_active = True
            self.audio_buffer = [] # 开始录音时清空buffer
        

    def stop_recording(self):
        with self.lock:
            self.recording_active = False

    def _record_audio(self):
        arecord_command = [
            'arecord',
            '-f', 'S16_LE',
            '-r', str(self.rate),
            '-c', str(self.channels),
            '-t', 'raw'
        ]
        self.process = None # 确保每次开始录音时 process 都被重置为 None

        print("音频录制线程已启动")
        while True:
            with self.lock:
                is_recording = self.recording_active

            if is_recording:
                if self.process is None:
                    try:
                        self.process = subprocess.Popen(arecord_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        print("音频录制已开始")
                    except Exception as e:
                        print(f"启动 arecord 失败: {e}")
                        return

                data = self.process.stdout.read(self.chunk * 2)
                if not data:
                    break

                with self.lock:
                    self.audio_buffer.append(data)
            else:
                if self.process is not None:
                    self.process.terminate()
                    self.process.wait()
                    self.process = None
                    print("音频录制已停止")
                time.sleep(0.1)

        if self.process is not None:
            self.process.terminate()
            self.process.wait()

    def get_audio_data(self):
        with self.lock:
            return bytes().join(self.audio_buffer) # 返回bytes数据而不是列表


class VideoRecorder:
    """视频录制类"""
    def __init__(self):
        self.recording_active = False
        self.video_queue = Queue()
        self.cap = None
        threading.Thread(target=self._record_video).start()

    def start_recording(self):
        self.recording_active = True
        self.video_queue = Queue() # 开始录像时清空队列

    def stop_recording(self):
        self.recording_active = False

    def _record_video(self):
        self.cap = cv2.VideoCapture(0)
        print("视频录制线程已启动")
        while True:
            if self.recording_active:
                ret, frame = self.cap.read()
                if ret:
                    self.video_queue.put((frame, time.time()))
                    cv2.imshow("Real Camera", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("无法获取摄像头画面")

        if self.cap and self.cap.isOpened(): # 退出循环后，再次确保释放
            self.cap.release()
            cv2.destroyAllWindows()

    def get_video_frames(self):
        frames = []
        while not self.video_queue.empty():
            frames.append(self.video_queue.get()[0]) # 仅获取帧数据，不包含时间戳
        return frames


class MediaSaver:
    """媒体文件保存类"""
    def __init__(self, output_dir=OUTPUT_DIR, folder_path=FOLDER_PATH):
        self.output_dir = output_dir
        self.folder_path = folder_path
        self.audio_file_count = 0
        self.saved_intervals = []

    def save_media(self, audio_data, video_frames):
        if not audio_data:
            return

        self.audio_file_count += 1
        audio_output_path = os.path.join(self.output_dir, f"audio_{self.audio_file_count}.wav")
        video_output_path = os.path.join(self.output_dir, f"video_{self.audio_file_count}.avi")

        # 保存音频
        wf = wave.open(audio_output_path, 'wb')
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(audio_data)
        wf.close()
        print(f"音频保存至 {audio_output_path}")

        # 保存视频
        if video_frames:
            out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
            for frame in video_frames:
                out.write(frame)
            out.release()
            print(f"视频保存至 {video_output_path}")

        # 记录保存的区间
        self.saved_intervals.append((time.time(), time.time()))

        return audio_output_path, video_output_path


class AudioPlayer:
    """音频播放类"""
    def play_audio_file(self, file_path):
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(1)
            print("播放完成！")
        except Exception as e:
            print(f"播放失败: {e}")
        finally:
            pygame.mixer.quit()

    async def synthesize_and_play(self, text, voice, output_file):
        """使用 edge-tts 合成语音并播放"""
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_file)
            self.play_audio_file(output_file)
        except Exception as e:
            print(f"语音合成或播放失败: {e}")

# -------- QwenAudioASR 类 ---------
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


class MultimodalInference:
    """多模态推理类，调用 Qwen-VL 模型"""
    def __init__(self, openai_api_key="sk-23fb1739f3a14b3c99666469715b9bdd", openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", folder_path=FOLDER_PATH):
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


class TextToSpeechSynthesizer:
    """文本转语音合成类"""
    def __init__(self):
        self.language_speaker_map = {
            "ja": "ja-JP-NanamiNeural", "fr": "fr-FR-DeniseNeural",
            "es": "ca-ES-JoanaNeural",  "de": "de-DE-KatjaNeural",
            "zh": "zh-CN-XiaoyiNeural", "en": "en-US-AnaNeural",
        }

    def get_speaker_by_language(self, text_output):
        """根据语种选择合适的音色"""
        language, confidence = langid.classify(text_output)
        if language not in self.language_speaker_map:
            used_speaker = "zh-CN-XiaoyiNeural" # 默认中文
        else:
            used_speaker = self.language_speaker_map[language]
            print("检测到语种：", language, "使用音色：", used_speaker)
        return used_speaker, language

class MainApp:
    """主应用类，负责协调各个模块"""
    def __init__(self):
        self.audio_recorder = AudioRecorder()
        self.video_recorder = VideoRecorder()
        self.media_saver = MediaSaver()
        self.audio_player = AudioPlayer()
        # self.speech_recognizer = SpeechRecognizer() # 移除 SenseVoice 语音识别器
        self.qwen_audio_asr = QwenAudioASR() # 添加 QwenAudioASR 语音识别器
        self.multimodal_inference = MultimodalInference()
        self.tts_synthesizer = TextToSpeechSynthesizer()
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
        # prompt_text = self.speech_recognizer.call_sensevoice_api(audio_output_path) # 使用 SenseVoice 语音识别
    
        if self.prompt_text:
            print("ASR 输出:", self.prompt_text)
            output_text = self.multimodal_inference.run_inference(audio_output_path, video_output_path, self.prompt_text)
            if output_text:
                speaker, language = self.tts_synthesizer.get_speaker_by_language(output_text)
                output_audio_file = os.path.join(FOLDER_PATH, f"sft_{self.media_saver.audio_file_count}.mp3")
                asyncio.run(self.audio_player.synthesize_and_play(output_text, speaker, output_audio_file))
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