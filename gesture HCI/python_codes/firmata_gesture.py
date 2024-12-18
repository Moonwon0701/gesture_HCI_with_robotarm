from pyfirmata import Arduino, util
import time
import serial.tools.list_ports
import cv2
import mediapipe as mp
import math


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

FINGER_BASES = [4, 8, 12, 16, 20]
WRIST = 0 

servo_pin_nums = [2, 3, 5, 9, 10, 11]
arduino_ports = [port.device for port in serial.tools.list_ports.comports() if 'Arduino' in port.description]
'''
for port in arduino_ports:
    try:
        board = Arduino(port)     # 포트 번호 확인하고 연결시키기기
    except:
        print(f"{port} Connection failed")      # 연결 오류 코드
'''
#Servo_pins = [board.get_pin(f'd:{i}:s') for i in servo_pin_nums]

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

                #for i,Servopin in enumerate(Servo_pins):
                #    Servopin.write(angles[i])
                
                #board.pass_time(1)


        cv2.imshow('MCP to Wrist Length Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#board.exit()
cap.release()
cv2.destroyAllWindows()