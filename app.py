import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
import mediapipe as mp
import numpy as np

# Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Color palette
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (0, 255, 255), (255, 0, 255), (0, 140, 255), (0, 0, 0)
]
color_names = ["Blue", "Green", "Red", "Yellow", "Purple", "Orange", "Black"]
brush_size = 15


def fingers_up(landmarks):
    fingers = []
    fingers.append(landmarks[4][0] < landmarks[3][0])  # Thumb
    fingers.append(landmarks[8][1] < landmarks[6][1])  # Index
    fingers.append(landmarks[12][1] < landmarks[10][1])  # Middle
    fingers.append(landmarks[16][1] < landmarks[14][1])  # Ring
    fingers.append(landmarks[20][1] < landmarks[18][1])  # Pinky
    return fingers


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.prev_x, self.prev_y = 0, 0
        self.eraser = False
        self.current_color = colors[0]
        self.frame_width = 1280
        self.frame_height = 720

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (self.frame_width, self.frame_height))

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        overlay = img.copy()
        alpha = 0.5
        cv2.rectangle(overlay, (0, 0), (self.frame_width, 100), (255, 255, 255), -1)

        for i, color in enumerate(colors):
            cv2.rectangle(overlay, (i * 100, 0), ((i + 1) * 100, 100), color, -1)
            cv2.putText(overlay, color_names[i], (i * 100 + 10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255) if i != 6 else (200, 200, 200), 2)

        frame = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark
            lm_list = [(int(lm[i].x * self.frame_width), int(lm[i].y * self.frame_height)) for i in range(21)]
            finger_status = fingers_up(lm_list)
            total_fingers = finger_status.count(True)

            x1, y1 = lm_list[8]

            if total_fingers >= 3:
                self.canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                self.prev_x, self.prev_y = 0, 0

            elif finger_status[1] and finger_status[2]:
                self.prev_x, self.prev_y = 0, 0
                if y1 < 100:
                    index = x1 // 100
                    if index < len(colors):
                        self.current_color = colors[index]
                        self.eraser = False

            elif finger_status[1] and not finger_status[2]:
                if self.prev_x == 0 and self.prev_y == 0:
                    self.prev_x, self.prev_y = x1, y1

                if self.eraser:
                    cv2.line(self.canvas, (self.prev_x, self.prev_y), (x1, y1), (255, 255, 255), 50)
                else:
                    cv2.line(self.canvas, (self.prev_x, self.prev_y), (x1, y1), self.current_color, brush_size)

                self.prev_x, self.prev_y = x1, y1
            else:
                self.prev_x, self.prev_y = 0, 0

        mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

        frame = cv2.bitwise_and(frame, mask_inv)
        frame = cv2.add(frame, self.canvas)

        return frame


# Streamlit UI
st.title("🖐️ AI Air Whiteboard using Mediapipe")
st.markdown("Use your **index finger** to draw. Raise two fingers to select color.")

webrtc_streamer(key="whiteboard", video_transformer_factory=VideoTransformer)
