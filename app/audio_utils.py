# multimodal_app/audio_utils.py

import time
import threading
import subprocess
import pygame
import wave
from multimodal_app.config import AUDIO_RATE, AUDIO_CHANNELS, CHUNK, COSYVOICE_MODEL, COSYVOICE_VOICE_ID
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer
import asyncio
import os

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
            '-t', 'raw',
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

    async def synthesize_and_play(self, text, output_file):
        """使用 CosyVoice API 合成语音并播放"""
        try:
            synthesizer = SpeechSynthesizer(model=COSYVOICE_MODEL, voice=COSYVOICE_VOICE_ID) # 实例化语音合成器
            audio_content = synthesizer.call(text) # 调用 API 进行语音合成
            print("requestId: ", synthesizer.get_last_request_id()) # 打印请求ID
            with open(output_file, "wb") as f: # 保存合成的音频到文件
                f.write(audio_content)
            self.play_audio_file(output_file) # 播放保存的音频文件
            print("播放完毕！！！")
        except Exception as e:
            print(f"CosyVoice 语音合成或播放失败: {e}")