# multimodal_app/video_utils.py

import cv2
import time
import threading
from queue import Queue

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