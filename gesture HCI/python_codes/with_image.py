import cv2
import mediapipe as mp
import math
import serial
import serial.tools.list_ports
from PIL import Image
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

r = 3  # 모터 회전자의 반지름 (cm)

FINGER_BASES = [4, 8, 12, 16, 20]
WRIST = 0


def find_arduino_port(baudrate=9600, timeout=1):
    """ 아두이노 포트를 찾아 연결하는 함수 """
    ports = serial.tools.list_ports.comports()
    arduino_port = None

    for port in ports:
        if "Arduino" in port.description or "USB" in port.description:
            arduino_port = port.device
            break

    if arduino_port:
        print(f"Arduino found on port: {arduino_port}")
        try:
            return serial.Serial(arduino_port, baudrate=baudrate, timeout=timeout)
        except serial.SerialException as e:
            print(f"Failed to connect to Arduino: {e}")
            return None
    else:
        print("No Arduino device found.")
        return None


def calculate_distance(point1, point2):
    """ 두 점 사이의 거리 계산 """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_servo_angles(distances):
    """ 거리 기반으로 서보 모터 각도 계산 """
    a = [1500, -375, -272.7272, -333.3333, -486.4864]
    b = [-465, 277.5, 218.1817, 253.3333, 296.7567]

    angles = []
    for i, distance in enumerate(distances):    
        angles.append(np.clip(distance * a[i] + b[i], 0, 180))
    return angles


def load_png_image(file_path):
    """ Pillow를 사용하여 투명 PNG 이미지를 OpenCV 형식으로 불러오는 함수 """
    try:
        pil_image = Image.open(file_path).convert("RGBA")
        np_image = np.array(pil_image)
        return cv2.cvtColor(np_image, cv2.COLOR_RGBA2BGRA)
    except Exception as e:
        print(f"Error loading PNG image: {e}")
        return None


def overlay_image(frame, overlay, position=(0, 0)):
    """ 투명 PNG 이미지를 프레임 위에 합성하는 함수 """
    x_offset, y_offset = position
    y1, y2 = y_offset, y_offset + overlay.shape[0]
    x1, x2 = x_offset, x_offset + overlay.shape[1]

    alpha_overlay = overlay[:, :, 3] / 255.0  # 알파 채널 (투명도)
    alpha_frame = 1.0 - alpha_overlay

    for c in range(3):  # RGB 채널
        frame[y1:y2, x1:x2, c] = (alpha_overlay * overlay[:, :, c] +
                                  alpha_frame * frame[y1:y2, x1:x2, c])


# 아두이노 포트 설정
py_serial = find_arduino_port()

# PNG 이미지 불러오기
ui_image_path = r'C:\Users\user\Desktop\프로그래밍\projects\UI_image.png'
ui_image = load_png_image(ui_image_path)

if ui_image is None:
    print("Error: UI image could not be loaded. Exiting.")
    exit()

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        # 화면 좌우 반전 및 RGB 변환
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 손 인식
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks.landmark
                wrist = landmarks[WRIST]
                wrist_coords = (wrist.x, wrist.y)

                outputs = []

                for base_idx in FINGER_BASES:
                    base = landmarks[base_idx]
                    base_coords = (base.x, base.y)

                    distance = calculate_distance(base_coords, wrist_coords)
                    outputs.append(math.floor(distance * 100) / 100)

                    text_position = (int(base.x * frame.shape[1]), int(base.y * frame.shape[0]))
                    cv2.putText(frame, f'{distance:.2f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                angles = get_servo_angles(distances=outputs)
                print(angles)
                py_serial.write(("090" + "".join(f"{math.floor(angle):03d}" for angle in angles) + "\n").encode('ascii'))

        # PNG 이미지를 프레임 최상단에 합성
        overlay_image(frame, ui_image, position=(0, 0))

        cv2.imshow('Hand Tracking with UI Overlay', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()