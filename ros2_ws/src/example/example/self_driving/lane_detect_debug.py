#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
import numpy as np
import queue
import math
import os

# --- Lane Detector class (간단화) ---
class LaneDetector:
    def __init__(self, color='yellow'):
        self.target_color = color
        self.weight_sum = 1.0
        # ROI: top, bottom, left, right, weight
        self.rois = ((300, 480, 0, 640, 0.7), (200, 300, 0, 640, 0.2), (100, 200, 0, 640, 0.1))

    def get_binary(self, image):
        # 간단히 yellow lane threshold
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([20, 100, 100])
        upper = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        return mask

    def __call__(self, image, result_image):
        h, w = image.shape[:2]
        centroid_sum = 0
        max_center_x = -1
        center_x = []
        for roi in self.rois:
            blob = image[roi[0]:roi[1], roi[2]:roi[3]]
            contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(max_contour)
                box = np.intp(cv2.boxPoints(rect))
                for j in range(4):
                    box[j,1] += roi[0]
                cv2.drawContours(result_image, [box], -1, (0,255,0), 2)
                # 중심점
                pt1, pt3 = box[0], box[2]
                line_center_x = (pt1[0] + pt3[0]) / 2
                center_x.append(line_center_x)
            else:
                center_x.append(-1)

        for i, cx in enumerate(center_x):
            if cx != -1:
                if cx > max_center_x:
                    max_center_x = cx
                centroid_sum += cx * self.rois[i][-1]

        if centroid_sum == 0:
            return result_image, None
        center_pos = centroid_sum / self.weight_sum
        angle = math.degrees(-math.atan((center_pos - w/2) / (h/2)))
        return result_image, angle

# --- ROS Node ---
class LaneDetectNode(Node):
    def __init__(self):
        super().__init__('lane_detect')
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, '/lane_detect/debug_image', 10)
        self.queue = queue.Queue(2)
        self.detector = LaneDetector('yellow')

        # 실제 카메라 없으면 동영상으로 테스트
        self.cap = cv2.VideoCapture(0)  # 0번 카메라
        threading.Thread(target=self.capture_thread, daemon=True).start()
        threading.Thread(target=self.process_thread, daemon=True).start()

    def capture_thread(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                continue
            if self.queue.full():
                self.queue.get()
            self.queue.put(frame)

    def process_thread(self):
        while rclpy.ok():
            try:
                frame = self.queue.get(timeout=1)
            except queue.Empty:
                continue

            binary = self.detector.get_binary(frame)
            overlay = frame.copy()
            overlay[binary>0] = [0,0,255]  # 빨간색으로 표시
            result, angle = self.detector(binary, overlay)
            msg = self.bridge.cv2_to_imgmsg(result, encoding='bgr8')
            self.pub.publish(msg)

            cv2.imshow('LaneDetect', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectNode()
    rclpy.spin(node)
    node.cap.release()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
