# multimodal_app/media_utils.py

import os
import wave
import cv2
import time
from multimodal_app.config import AUDIO_CHANNELS, AUDIO_RATE, OUTPUT_DIR, FOLDER_PATH

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