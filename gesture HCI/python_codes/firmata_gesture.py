from pyfirmata import Arduino
import serial.tools.list_ports
import cv2
import mediapipe as mp
import math
import numpy as np
from PIL import Image

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

FINGER_BASES = [4, 8, 12, 16, 20]
WRIST = 0

INITIAL_ANGLE = [90, 180, 0, 0, 0, 0]
servo_pin_nums = [3, 11, 9, 6, 5, 10]

arduino_ports = [port.device for port in serial.tools.list_ports.comports() if ('Arduino' in port.description or "COM" in port.description)]

# 아두이노 연결
for port in arduino_ports:
    try:
        board = Arduino(port)
    except:
        print(f"{port} Connection failed")

Servo_pins = [board.get_pin(f'd:{i}:s') for i in servo_pin_nums]
for i, Servopin in enumerate(Servo_pins):
    Servopin.write(INITIAL_ANGLE[i])

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

def get_servo_angles(distances):
    a = [1500, -375, -272.7272, -333.3333, -486.4864]
    b = [-465, 277.5, 218.1817, 253.3333, 296.7567]
    angles = [90]
    for i, distance in enumerate(distances):
        angles.append(np.clip(distance * a[i] + b[i], 0, 180))
    return angles

def load_png_image(file_path):
    try:
        pil_image = Image.open(file_path).convert("RGBA")
        np_image = np.array(pil_image)
        return cv2.cvtColor(np_image, cv2.COLOR_RGBA2BGRA)
    except Exception as e:
        print(f"Error loading PNG image: {e}")
        return None
    
def overlay_image(frame, overlay, position=(0, 0)):
    x_offset, y_offset = position
    y1, y2 = y_offset, y_offset + overlay.shape[0]
    x1, x2 = x_offset, x_offset + overlay.shape[1]

    alpha_overlay = overlay[:, :, 3] / 255.0  # 알파 채널
    alpha_frame = 1.0 - alpha_overlay

    for c in range(3):  # RGB 채널
        frame[y1:y2, x1:x2, c] = (alpha_overlay * overlay[:, :, c] + alpha_frame * frame[y1:y2, x1:x2, c])

# UI 이미지 불러오기
ui_image_path = r'gesture HCI\UI_image\UI_image.png'
ui_image = load_png_image(ui_image_path)

cap = cv2.VideoCapture(0)

# 손 추적 및 제어
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("웹캠에서 영상을 가져올 수 없습니다.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks.landmark
                wrist = landmarks[WRIST]
                wrist_coords = (wrist.x, wrist.y, wrist.z)

                outputs = []
                for base_idx in FINGER_BASES:
                    base = landmarks[base_idx]
                    base_coords = (base.x, base.y, base.z)

                    distance = calculate_distance(base_coords, wrist_coords)
                    outputs.append(math.floor(distance * 100) / 100)

                    text_position = (int(base.x * frame.shape[1]), int(base.y * frame.shape[0]))
                    cv2.putText(frame, f'{distance:.2f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                angles = get_servo_angles(distances=outputs)
                
                for i,Servopin in enumerate(Servo_pins):
                    Servopin.write(angles[i])


        overlay_image(frame, ui_image, position=(0, 0))

        cv2.imshow('MCP to Wrist Length Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
board.exit()
