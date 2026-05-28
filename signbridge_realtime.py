"""
SignBridge - Real-time Sign Language Recognition
Google MediaPipe + Number Model + Word Model (dongsaeng/busang)
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import json
import time
import torch
import torch.nn as nn
from collections import deque, Counter

# ────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────
BASE         = r"C:\Users\AISW_203_101\Desktop\Team_SignBridge\model"
OUTPUT_DIR   = r"C:\Users\AISW_203_101\Desktop\Team_SignBridge\keypoints_output"
GOOGLE_MODEL = BASE + r"\gesture_recognizer.task"
NUMBER_MODEL = BASE + r"\number_model.pkl"
NUMBER_LABEL = BASE + r"\number_labels.pkl"
WORD_MODEL   = r"C:\Users\AISW_203_101\Desktop\Team_SignBridge\sign_model.pt"
LABEL_MAP_F  = OUTPUT_DIR + r"\label_map.json"

# ────────────────────────────────────────────
# Word model (LSTM)
# ────────────────────────────────────────────
MAX_FRAMES  = 30
FEATURE_DIM = 42  # 21 * 2 (x, y)

class SignLSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, 128, batch_first=True)
        self.drop1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.drop2 = nn.Dropout(0.3)
        self.fc1   = nn.Linear(64, 64)
        self.relu  = nn.ReLU()
        self.fc2   = nn.Linear(64, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x[:, -1, :])
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ────────────────────────────────────────────
# Korean text rendering
# ────────────────────────────────────────────
def put_korean_text(img, text, pos, font_size=28, color=(0, 255, 100)):
    try:
        from PIL import ImageFont, ImageDraw, Image
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", font_size)
        except:
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
# Hand drawing
# ────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_hand(frame, hand_landmarks, w, h):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 200, 150), 2)
    for pt in points:
        cv2.circle(frame, pt, 5, (0, 255, 200), -1)

# ────────────────────────────────────────────
# Gesture mappings
# ────────────────────────────────────────────
GESTURE_KO = {
    "None":        "손을 보여주세요",
    "Closed_Fist": "주먹",
    "Open_Palm":   "손바닥",
    "Pointing_Up": "위로",
    "Thumb_Up":    "좋아요",
    "Thumb_Down":  "싫어요",
    "Victory":     "브이",
    "ILoveYou":    "사랑해요",
}

GESTURE_TRANS = {
    "Closed_Fist":  {"en": "Fist",       "ja": "拳",     "zh": "拳头"},
    "Open_Palm":    {"en": "Palm of the Hand",  "ja": "手のひら",     "zh": "手掌"},
    "Pointing_Up":  {"en": "above",      "ja": "上",       "zh": "多于"},
    "Thumb_Up":     {"en": "Good",       "ja": "いいね",   "zh": "伟大的"},
    "Thumb_Down":   {"en": "Bad",        "ja": "嫌い",     "zh": "不，我不想"},
    "Victory":      {"en": "V",          "ja": "V",        "zh": "V"},
    "ILoveYou":     {"en": "I Love You", "ja": "愛してる", "zh": "我爱你"},
}

# ────────────────────────────────────────────
# Load models
# ────────────────────────────────────────────
# Google gesture model
base_options = python.BaseOptions(model_asset_path=GOOGLE_MODEL)
options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=2)
recognizer = vision.GestureRecognizer.create_from_options(options)
print("Google gesture model loaded!")

# Number model
try:
    import numpy as np
    np.core = np  # numpy 2.x 호환 패치
    with open(NUMBER_MODEL, "rb") as f:
        number_model = pickle.load(f)
    with open(NUMBER_LABEL, "rb") as f:
        number_labels = pickle.load(f)
    NUMBER_OK = True
    print(f"Number model loaded! ({list(number_labels)})")
except Exception as e:
    NUMBER_OK = False
    print(f"Number model failed: {e}")

# Word model (dongsaeng/busang)
try:
    ckpt = torch.load(WORD_MODEL, map_location="cpu", weights_only=False)
    with open(LABEL_MAP_F, encoding="utf-8") as f:
        word_label_map = json.load(f)
    word_model = SignLSTM(FEATURE_DIM, len(ckpt["le_classes"]))
    word_model.load_state_dict(ckpt["model"])
    word_model.eval()
    WORD_OK = True
    print(f"Word model loaded! Labels: {word_label_map}")
except Exception as e:
    WORD_OK = False
    print(f"Word model failed: {e}")

# ────────────────────────────────────────────
# Stabilizer buffers
# ────────────────────────────────────────────
google_buffer = deque(maxlen=8)
number_buffer = deque(maxlen=8)
word_buffer   = deque(maxlen=10)

def stable(buffer, value):
    buffer.append(value)
    return Counter(buffer).most_common(1)[0][0]

# ────────────────────────────────────────────
# Number prediction
# ────────────────────────────────────────────
def predict_number(hand_landmarks):
    base_x = hand_landmarks[0].x
    base_y = hand_landmarks[0].y
    vec = []
    for lm in hand_landmarks:
        vec.append(lm.x - base_x)
        vec.append(lm.y - base_y)
    pred = number_model.predict(np.array([vec]))[0]
    prob = number_model.predict_proba(np.array([vec]))[0]
    return str(number_labels[pred]), float(np.max(prob))

# ────────────────────────────────────────────
# Word prediction (LSTM - uses frame buffer)
# ────────────────────────────────────────────
frame_seq = deque(maxlen=MAX_FRAMES)   # rolling window of keypoints

def update_word_pred(hand_landmarks, w, h):
    # 0~1 정규화 (해상도 무관하게 통일)
    flat = np.array([[lm.x, lm.y] for lm in hand_landmarks],
                    dtype=np.float32).flatten()  # (42,) 이미 0~1
    # 손목 기준 상대좌표로 변환 (위치 이동 무관)
    base_x, base_y = flat[0], flat[1]
    for i in range(0, len(flat), 2):
        flat[i]   -= base_x
        flat[i+1] -= base_y

    frame_seq.append(flat)

    if len(frame_seq) < MAX_FRAMES:
        return None, 0.0

    seq = np.array(frame_seq, dtype=np.float32)   # (30, 42)
    x   = torch.tensor(seq[np.newaxis], dtype=torch.float32)
    with torch.no_grad():
        probs = torch.softmax(word_model(x), dim=1)[0]
    idx  = probs.argmax().item()
    conf = float(probs[idx])
    label = word_label_map[str(idx)]
    return label, conf

# ────────────────────────────────────────────
# Main
# ────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Camera open! Press Q to quit.")

    prev_time  = time.time()
    g_gesture  = "None"
    g_conf     = 0.0
    n_label    = ""
    n_conf     = 0.0
    w_label    = ""
    w_conf     = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = recognizer.recognize(mp_image)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # Draw hand
        if result.hand_landmarks:
            for hand_landmark in result.hand_landmarks:
                draw_hand(frame, hand_landmark, w, h)

        # Google gesture
        if result.gestures:
            g_gesture = stable(google_buffer, result.gestures[0][0].category_name)
            g_conf    = result.gestures[0][0].score
        else:
            g_gesture = stable(google_buffer, "None")
            g_conf    = 0.0

        # Number prediction
        if NUMBER_OK and result.hand_landmarks:
            try:
                lbl, conf = predict_number(result.hand_landmarks[0])
                n_label = stable(number_buffer, lbl)
                n_conf  = conf
            except:
                pass

        # Word prediction (LSTM)
        if WORD_OK and result.hand_landmarks:
            try:
                lbl, conf = update_word_pred(result.hand_landmarks[0], w, h)
                if lbl and conf >= 0.70:   # 신뢰도 70% 이상만 버퍼에 추가
                    word_buffer.append(lbl)
                else:
                    word_buffer.append("None")
            except:
                word_buffer.append("None")
        else:
            frame_seq.clear()
            word_buffer.append("None")

        # 버퍼의 60% 이상이 같은 단어일 때만 표시
        if len(word_buffer) >= 10:
            counts = Counter(word_buffer)
            top_label, top_count = counts.most_common(1)[0]
            if top_label != "None" and top_count >= 6:
                w_label = top_label
                w_conf  = top_count / len(word_buffer)
            else:
                w_label = ""
                w_conf  = 0.0

        # ── UI ──────────────────────────────
        cv2.rectangle(frame, (0, 0), (w, 140), (0, 0, 0), -1)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # 메인 텍스트: 수어 단어 인식되면 수어 단어 우선 표시, 아니면 구글 제스처
        word_ko = {"dongsaeng": "동생", "busang": "부상", "unknown": "손을 보여주세요"}.get(w_label, "")
        if WORD_OK and word_ko and w_conf >= 0.6:
            if w_label == "unknown":
                # unknown이면 구글 제스처 표시
                ko_text    = GESTURE_KO.get(g_gesture, g_gesture)
                text_color = (0, 255, 100) if g_gesture != "None" else (100, 100, 100)
                frame = put_korean_text(frame, ko_text, (10, 10), font_size=34, color=text_color)
                bar_conf = g_conf
            else:
                # 동생/부상 인식됨 → 크게 표시
                frame = put_korean_text(frame, word_ko, (10, 10), font_size=48, color=(0, 255, 100))
                bar_conf = w_conf
        else:
            # 아직 판단 중 → 구글 제스처 표시
            ko_text    = GESTURE_KO.get(g_gesture, g_gesture)
            text_color = (0, 255, 100) if g_gesture != "None" else (100, 100, 100)
            frame = put_korean_text(frame, ko_text, (10, 10), font_size=34, color=text_color)
            bar_conf = g_conf

        # Number (line 2)
        if NUMBER_OK and n_label and n_conf >= 0.6:
            frame = put_korean_text(frame, f"숫자: {n_label} ({n_conf:.0%})",
                                    (10, 70), font_size=24, color=(255, 220, 0))

        # Confidence bar
        bar_w = int((w - 20) * bar_conf)
        cv2.rectangle(frame, (10, 115), (w - 10, 130), (40, 40, 40), -1)
        if bar_w > 0:
            bar_color = (0, 200, 100) if bar_conf > 0.7 else (0, 150, 255)
            cv2.rectangle(frame, (10, 115), (10 + bar_w, 130), bar_color, -1)

        # Translation (bottom) - 구글 제스처 있으면 항상 표시
        if g_gesture in GESTURE_TRANS:
            trans = GESTURE_TRANS[g_gesture]
            trans_text = f"EN: {trans['en']}  |  JA: {trans['ja']}  |  ZH: {trans['zh']}"
            cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
            frame = put_korean_text(frame, trans_text, (10, h - 35),
                                    font_size=18, color=(200, 200, 200))

        cv2.imshow("SignBridge", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()
    print("Done!")

if __name__ == "__main__":
    main()
