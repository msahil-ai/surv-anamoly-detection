# visualizer.py
import cv2
import math
import numpy as np

class Visualizer:
    @staticmethod
    def draw_dashed_rectangle(img, pt1, pt2, color, thickness=2, dash_length=15):
        x1, y1 = pt1
        x2, y2 = pt2

        for i in range(x1, x2, dash_length * 2):
            cv2.line(img, (i, y1), (min(i + dash_length, x2), y1), color, thickness)
            cv2.line(img, (i, y2), (min(i + dash_length, x2), y2), color, thickness)
        for i in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, i), (x1, min(i + dash_length, y2)), color, thickness)
            cv2.line(img, (x2, i), (x2, min(i + dash_length, y2)), color, thickness)

    @staticmethod
    def draw_alert(frame, x, y):
        text = "ABANDONED BAG!"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw, y + 10), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.circle(frame, (x, y), 30, (0, 0, 255), 4)

    @staticmethod
    def draw_dashed_circle(img, center, radius, color, thickness=2, dash_length=15):
        circumference = 2 * math.pi * radius
        dashes = int(circumference / dash_length)

        for i in range(dashes):
            if i % 2 == 0:
                start_angle = i * (360 / dashes)
                end_angle = (i + 1) * (360 / dashes)
                cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, color, thickness)