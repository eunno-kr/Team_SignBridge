"""
SignBridge - 실시간 수어 인식 + 번역
Google MediaPipe + 숫자 커스텀 모델 통합
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ai_edge_litert.interpreter import Interpreter
import numpy as np
import pickle
import time
from collections import deque, Counter

# ────────────────────────────────────────────
# 경로 설정
# ────────────────────────────────────────────
BASE          = r"C:\Users\AISW_203_101\Desktop\Team_SignBridge\model"
GOOGLE_MODEL  = BASE + r"\gesture_recognizer.task"
NUMBER_MODEL  = BASE + r"\number_model.pkl"
NUMBER_LABEL  = BASE + r"\number_labels.pkl"

# ────────────────────────────────────────────
# 구글 기본 제스처 한국어 매핑
# ────────────────────────────────────────────
GESTURE_KO = {
    "None":        "손을 보여주세요",
    "Closed_Fist": "주먹",
    "Open_Palm":   "손바닥",
    "Pointing_Up": "위로",
    "Thumb_Up":    "좋아요 ",
    "Thumb_Down":  "싫어요 ",
    "Victory":     "브이 ",
    "ILoveYou":    "사랑해요 ",
}

GESTURE_TRANS = {
    "Closed_Fist":  {"en": "Fist",       "ja": "拳",     "zh": "拳头"},
    "Open_Palm":    {"en": "Palm of the hand",  "ja": "手のひら",     "zh": "手掌"},
    "Pointing_Up":  {"en": "Point Up",   "ja": "上", "zh": "多于"},
    "Thumb_Up":     {"en": "Good",       "ja": "いいね",   "zh": "伟大的"},
    "Thumb_Down":   {"en": "Bad",        "ja": "嫌い",     "zh": "不，我不想"},
    "Victory":      {"en": "V",    "ja": "V",   "zh": "V"},
    "ILoveYou":     {"en": "I Love You", "ja": "愛してる", "zh": "我爱你"},
}

# 손 연결선
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ────────────────────────────────────────────
# 한글 출력
# ────────────────────────────────────────────
def put_korean_text(img, text, pos, font_size=28, color=(0, 255, 100)):
    try:
        from PIL import ImageFont, ImageDraw, Image
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/simsun.ttc", font_size)
        except:
            font = ImageFont.load_default()
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return img

# ────────────────────────────────────────────
# 손 그리기
# ────────────────────────────────────────────
def draw_hand(frame, hand_landmarks, w, h):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 200, 150), 2)
    for pt in points:
        cv2.circle(frame, pt, 5, (0, 255, 200), -1)

# ────────────────────────────────────────────
# 모델 로딩
# ────────────────────────────────────────────
# 구글 제스처 모델
base_options = python.BaseOptions(model_asset_path=GOOGLE_MODEL)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=2
)
recognizer = vision.GestureRecognizer.create_from_options(options)
print("✅ 구글 제스처 모델 로드 완료!")

# 숫자 모델
try:
    with open(NUMBER_MODEL, "rb") as f:
        number_model = pickle.load(f)
    with open(NUMBER_LABEL, "rb") as f:
        number_labels = pickle.load(f)
    NUMBER_OK = True
    print(f"✅ 숫자 모델 로드 완료! ({list(number_labels)})")
except Exception as e:
    NUMBER_OK = False
    print(f"⚠️ 숫자 모델 로드 실패: {e}")

# ────────────────────────────────────────────
# 숫자 예측
# ────────────────────────────────────────────
def predict_number(hand_landmarks):
    base_x = hand_landmarks[0].x
    base_y = hand_landmarks[0].y
    vec = []
    for lm in hand_landmarks:
        vec.append(lm.x - base_x)
        vec.append(lm.y - base_y)
    X = np.array([vec])
    pred = number_model.predict(X)[0]
    prob = number_model.predict_proba(X)[0]
    conf = float(np.max(prob))
    return str(number_labels[pred]), conf

# ────────────────────────────────────────────
# 예측 안정화
# ────────────────────────────────────────────
google_buffer = deque(maxlen=8)
number_buffer = deque(maxlen=8)

def stable(buffer, value):
    buffer.append(value)
    return Counter(buffer).most_common(1)[0][0]

# ────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 찾을 수 없습니다!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("📷 카메라 열림! Q키로 종료하세요.")

    prev_time = time.time()
    g_gesture = "None"
    g_conf    = 0.0
    n_label   = ""
    n_conf    = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = recognizer.recognize(mp_image)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # 손 그리기
        if result.hand_landmarks:
            for hand_landmark in result.hand_landmarks:
                draw_hand(frame, hand_landmark, w, h)

        # 구글 제스처
        if result.gestures:
            g_gesture = stable(google_buffer, result.gestures[0][0].category_name)
            g_conf    = result.gestures[0][0].score
        else:
            g_gesture = stable(google_buffer, "None")
            g_conf    = 0.0

        # 숫자 예측
        if NUMBER_OK and result.hand_landmarks:
            try:
                label, conf = predict_number(result.hand_landmarks[0])
                n_label = stable(number_buffer, label)
                n_conf  = conf
            except:
                n_label = ""
                n_conf  = 0.0

        # ── UI ──
        cv2.rectangle(frame, (0, 0), (w, 115), (0, 0, 0), -1)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # 구글 제스처
        ko_text = GESTURE_KO.get(g_gesture, g_gesture)
        text_color = (0, 255, 100) if g_gesture != "None" else (100, 100, 100)
        frame = put_korean_text(frame, ko_text, (10, 10), font_size=36, color=text_color)

        # 숫자 표시
        if NUMBER_OK and n_label and n_conf >= 0.6:
            num_text = f"숫자: {n_label}  ({n_conf:.0%})"
            frame = put_korean_text(frame, num_text, (10, 58),
                                    font_size=26, color=(255, 220, 0))

        # 신뢰도 바
        bar_w = int((w - 20) * g_conf)
        cv2.rectangle(frame, (10, 90), (w - 10, 105), (40, 40, 40), -1)
        if bar_w > 0:
            bar_color = (0, 200, 100) if g_conf > 0.7 else (0, 150, 255)
            cv2.rectangle(frame, (10, 90), (10 + bar_w, 105), bar_color, -1)

        # 다국어 번역 (하단)
        if g_gesture in GESTURE_TRANS:
            trans = GESTURE_TRANS[g_gesture]
            trans_text = f"EN: {trans['en']}  |  JA: {trans['ja']}  |  ZH: {trans['zh']}"
            cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
            frame = put_korean_text(frame, trans_text, (10, h - 35),
                                    font_size=18, color=(200, 200, 200))

        cv2.imshow("SignBridge", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()
    print("종료!")


if __name__ == "__main__":
    main()
