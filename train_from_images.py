"""
SignBridge - 이미지로 수어 학습
이미지 → MediaPipe 키포인트 추출 → 모델 학습
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────
DATASET_DIR = r"C:\Users\AISW_203_101\Desktop\Team_SignBridge\dataset"
MODEL_SAVE  = r"C:\Users\AISW_203_101\Desktop\Team_SignBridge\model\number_model.pkl"
LABEL_SAVE  = r"C:\Users\AISW_203_101\Desktop\Team_SignBridge\model\number_labels.pkl"
HAND_MODEL  = r"C:\Users\AISW_203_101\Desktop\Team_SignBridge\model\hand_landmarker.task"

# ────────────────────────────────────────────
# MediaPipe HandLandmarker 초기화
# ────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=HAND_MODEL)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)
landmarker = vision.HandLandmarker.create_from_options(options)
print("✅ MediaPipe 손 인식기 로드 완료!")

# ────────────────────────────────────────────
# 이미지에서 키포인트 추출
# ────────────────────────────────────────────
def extract_from_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    if w > 640:
        scale = 640 / w
        img = cv2.resize(img, (640, int(h * scale)))

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]
    base_x = hand[0].x
    base_y = hand[0].y

    vec = []
    for lm in hand:
        vec.append(lm.x - base_x)
        vec.append(lm.y - base_y)

    return vec  # 42차원

# ────────────────────────────────────────────
# 데이터 로딩
# ────────────────────────────────────────────
def load_dataset():
    X, y = [], []
    dataset_path = Path(DATASET_DIR)
    failed = []

    print("=" * 50)
    print("  이미지 데이터 로딩 중...")
    print("=" * 50)

    for label_folder in sorted(dataset_path.iterdir()):
        if not label_folder.is_dir():
            continue

        label = label_folder.name
        img_files = list(label_folder.glob("*.jpg")) + \
                    list(label_folder.glob("*.jpeg")) + \
                    list(label_folder.glob("*.png")) + \
                    list(label_folder.glob("*.webp"))

        success = 0
        for img_path in img_files:
            kp = extract_from_image(img_path)
            if kp:
                X.append(kp)
                y.append(label)
                success += 1
            else:
                failed.append(img_path.name)

        print(f"  [{label}]: {success}/{len(img_files)}개 성공")

    print(f"\n총 {len(X)}개 샘플 로드 완료")
    if failed:
        print(f"손 인식 실패: {len(failed)}개 건너뜀")

    return np.array(X, dtype=np.float32), np.array(y)

# ────────────────────────────────────────────
# 데이터 증강
# ────────────────────────────────────────────
def augment_data(X, y, factor=10):
    X_aug, y_aug = list(X), list(y)
    for i in range(len(X)):
        for _ in range(factor):
            noise = np.random.normal(0, 0.005, X[i].shape)
            X_aug.append(X[i] + noise)
            y_aug.append(y[i])
    return np.array(X_aug), np.array(y_aug)

# ────────────────────────────────────────────
# 모델 학습
# ────────────────────────────────────────────
def train(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"\n클래스: {list(le.classes_)}")
    print(f"클래스 수: {len(le.classes_)}")

    print("\n데이터 증강 중...")
    X_aug, y_aug = augment_data(X, y_enc, factor=10)
    print(f"증강 후 샘플 수: {len(X_aug)}개")

    X_train, X_val, y_train, y_val = train_test_split(
        X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug
    )

    print(f"학습: {len(X_train)}개 / 검증: {len(X_val)}개")

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )

    print("\n🧠 학습 시작...")
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    print(f"\n✅ 검증 정확도: {acc * 100:.1f}%")

    return model, le

# ────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────
def main():
    print("\n🤟 SignBridge 이미지 학습 시작!\n")

    X, y = load_dataset()

    if len(X) == 0:
        print("\n❌ 로드된 데이터가 없습니다!")
        print("dataset 폴더에 이미지가 있는지 확인하세요.")
        return

    if len(set(y)) < 2:
        print("\n❌ 클래스가 1개뿐입니다. 최소 2개 이상 필요해요.")
        return

    model, le = train(X, y)

    with open(MODEL_SAVE, "wb") as f:
        pickle.dump(model, f)
    with open(LABEL_SAVE, "wb") as f:
        pickle.dump(le.classes_, f)

    print(f"\n💾 모델 저장 완료!")
    print("\n🎉 학습 완료! main.py에 연결할 수 있어요!")

if __name__ == "__main__":
    main()
