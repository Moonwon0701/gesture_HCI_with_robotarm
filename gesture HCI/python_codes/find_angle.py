import cv2
import mediapipe as mp
import math
import serial
import serial.tools.list_ports


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

r = 3       # 모터 회전자의 반지름 (cm)

FINGER_BASES = [4, 8, 12, 16, 20]
WRIST = 0 


def find_arduino_port(baudrate=9600, timeout=1):        # 아두이노 포트 찾아 연결하는 코드
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

py_serial = find_arduino_port()

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def get_servo_angles(distances):
    i = 0

    a = [10, 10, 10, 10, 10]        # 첫번째 상수 리스트
    b = [5, 5, 5, 5, 5]             # 두번째 상수 리스트

    angles = []

    for i,distance in enumerate(distances):
        angles.append(distance * a[i] + b[i])

    return angles

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands = 1) as hands:
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
                wrist_coords = (wrist.x, wrist.y)

                outputs = []

                for base_idx in FINGER_BASES:
                    base = landmarks[base_idx]
                    base_coords = (base.x, base.y)

                    distance = calculate_distance(base_coords, wrist_coords)
                    outputs.append(math.floor(distance*100) / 100)

                    text_position = (int(base.x * frame.shape[1]), int(base.y * frame.shape[0]))
                    cv2.putText(frame, f'{distance:.2f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                angles = get_servo_angles(distances=outputs)
                print(angles)
                #py_serial.write(("090" + "".join(f"{math.floor(angle):03d}" for angle in angles) + "\n").encode('ascii'))

        cv2.imshow('MCP to Wrist Length Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()