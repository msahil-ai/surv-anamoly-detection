import cv2
import threading
import time
import numpy as np
from collections import deque
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

class CloudStreamBuffer:
    def __init__(self, stream_url, buffer_seconds=10, fps=30):
        self.cap = cv2.VideoCapture(stream_url)
        self.fps = fps
        self.buffer_size = buffer_seconds * fps
        # Deque acts as our rolling memory buffer
        self.frame_buffer = deque(maxlen=self.buffer_size) 
        self.stopped = False
        
    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                print("Stream disconnected. Attempting to reconnect...")
                time.sleep(2)
                continue
            
            timestamp = time.time()
            self.frame_buffer.append({'frame': frame, 'time': timestamp})
            # Remove sleep if reading an actual live RTSP stream
            time.sleep(1.0 / self.fps) 

    def read_live(self):
        if len(self.frame_buffer) > 0:
            return self.frame_buffer[-1]['frame']
        return None

    def get_historical_clip(self):
        return list(self.frame_buffer)

    def stop(self):
        self.stopped = True
        self.cap.release()