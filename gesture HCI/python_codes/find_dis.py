'''구부림 정도 파악'''
import cv2
import mediapipe as mp
import math

# MediaPipe 손 추적 모듈 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 유클리드 거리 계산 함수
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

# MCP와 WRIST의 랜드마크 인덱스
FINGER_BASES = [4, 8, 12, 16, 20]  # 엄지, 검지, 중지, 약지, 소지의 끝 관절
WRIST = 0  # 손목 랜드마크

# 웹캠 영상 처리
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("웹캠에서 영상을 가져올 수 없습니다.")
            break

        # 영상 좌우 반전 및 RGB 변환
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe에서 손 탐지
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 랜드마크 좌표 추출
                landmarks = hand_landmarks.landmark
                wrist = landmarks[WRIST]
                wrist_coords = (wrist.x, wrist.y, wrist.z)

                # 손가락 첫 관절(MCP)과 손목(WRIST) 거리 계산
                for base_idx in FINGER_BASES:
                    base = landmarks[base_idx]
                    base_coords = (base.x, base.y, base.z)

                    # 거리 계산
                    distance = calculate_distance(base_coords, wrist_coords)

                    # 화면에 거리 표시
                    text_position = (int(base.x * frame.shape[1]), int(base.y * frame.shape[0]))
                    cv2.putText(frame, f'{distance:.2f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 화면 출력
        cv2.imshow('MCP to Wrist Length Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
