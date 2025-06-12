import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Define colors and palette labels
colors = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Purple
    (0, 140, 255),   # Orange
    (0, 0, 0),       # Black
]
color_names = ["Blue", "Green", "Red", "Yellow", "Purple", "Orange", "Black"]
brush_size = 15
eraser = False
current_color = colors[0]
prev_x, prev_y = 0, 0

# Canvas setup
canvas_h, canvas_w = 720, 1280
canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, canvas_w)
cap.set(4, canvas_h)

def fingers_up(landmarks):
    fingers = []
    fingers.append(landmarks[4][0] < landmarks[3][0])  # Thumb
    fingers.append(landmarks[8][1] < landmarks[6][1])  # Index
    fingers.append(landmarks[12][1] < landmarks[10][1])  # Middle
    fingers.append(landmarks[16][1] < landmarks[14][1])  # Ring
    fingers.append(landmarks[20][1] < landmarks[18][1])  # Pinky
    return fingers

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (canvas_w, canvas_h))  # Ensure fixed size

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Translucent palette area overlay
    overlay = frame.copy()
    alpha = 0.5
    cv2.rectangle(overlay, (0, 0), (canvas_w, 100), (255, 255, 255), -1)

    for i, color in enumerate(colors):
        cv2.rectangle(overlay, (i * 100, 0), ((i + 1) * 100, 100), color, -1)
        cv2.putText(overlay, color_names[i], (i * 100 + 10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255) if i != 6 else (200, 200, 200), 2)

    cv2.rectangle(overlay, (700, 0), (800, 100), (50, 50, 50), -1)
    cv2.putText(overlay, "Eraser", (710, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.rectangle(overlay, (800, 0), (900, 100), (100, 100, 100), -1)
    cv2.putText(overlay, "Clear", (810, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Add translucent bar to frame
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark
        lm_list = [(int(lm[i].x * canvas_w), int(lm[i].y * canvas_h)) for i in range(21)]

        finger_status = fingers_up(lm_list)
        total_fingers = finger_status.count(True)

        x1, y1 = lm_list[8]  # Index tip

        if total_fingers >= 3:
            canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)

        elif finger_status[1] and finger_status[2]:
            prev_x, prev_y = 0, 0
            if y1 < 100:
                index = x1 // 100
                if index < len(colors):
                    current_color = colors[index]
                    eraser = False
                elif 700 < x1 < 800:
                    eraser = True
                elif 800 < x1 < 900:
                    canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)

        elif finger_status[1] and not finger_status[2]:
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x1, y1

            if eraser:
                cv2.line(canvas, (prev_x, prev_y), (x1, y1), (255, 255, 255), 50)
            else:
                cv2.line(canvas, (prev_x, prev_y), (x1, y1), current_color, brush_size)

            prev_x, prev_y = x1, y1
        else:
            prev_x, prev_y = 0, 0

    # Merge canvas with frame safely
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

    # Resize mask if needed (prevent future errors)
    mask_inv = cv2.resize(mask_inv, (frame.shape[1], frame.shape[0]))
    canvas = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))

    frame = cv2.bitwise_and(frame, mask_inv)
    frame = cv2.add(frame, canvas)

    cv2.imshow("Air Whiteboard", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('1'):
        brush_size = 5
    elif key == ord('2'):
        brush_size = 15
    elif key == ord('3'):
        brush_size = 30
    elif key == ord('s'):
        cv2.imwrite("my_drawing.png", canvas)

cap.release()
cv2.destroyAllWindows()
