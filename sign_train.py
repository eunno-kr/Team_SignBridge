# SignBridge - JSON Keypoint Training with Sliding Window
# WORD1518=dongsaeng / WORD1519=busang

import os, re, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# ── Settings ──────────────────────────────────
JSON_ROOT  = r"C:\Users\AISW_203_101\Desktop\Team_SignBridge\New_sample (1)\라벨링데이터\REAL\WORD\01_real_word_keypoint"
OUTPUT_DIR = r"C:\Users\AISW_203_101\Desktop\Team_SignBridge\keypoints_output"
MODEL_SAVE = r"C:\Users\AISW_203_101\Desktop\Team_SignBridge\sign_model.pt"

LABEL_MAP = {
    "WORD1518": "dongsaeng",
    "WORD1519": "busang",
    "WORD1501": "unknown",
    "WORD1502": "unknown",
    "WORD1503": "unknown",
    "WORD1504": "unknown",
    "WORD1505": "unknown",
    "WORD1506": "unknown",
    "WORD1507": "unknown",
    "WORD1508": "unknown",
    "WORD1509": "unknown",
    "WORD1510": "unknown",
    "WORD1511": "unknown",
    "WORD1512": "unknown",
    "WORD1513": "unknown",
    "WORD1514": "unknown",
    "WORD1515": "unknown",
    "WORD1516": "unknown",
    "WORD1517": "unknown",
    "WORD1520": "unknown",
}

MAX_FRAMES  = 30
FEATURE_DIM = 21 * 2  # x, y = 42
STRIDE      = 2       # 슬라이딩 윈도우 간격

# ── Label from folder name ─────────────────────
def get_label(folder_name):
    m = re.search(r"(WORD\d+)", folder_name, re.IGNORECASE)
    if not m:
        return None
    return LABEL_MAP.get(m.group(1).upper())

# ── Load all frames from one folder ───────────
def load_frames(folder_path):
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith("_keypoints.json")])
    if len(json_files) < MAX_FRAMES:
        return []

    # 해상도 추출
    try:
        with open(os.path.join(folder_path, json_files[0]), encoding="utf-8") as f:
            d = json.load(f)
        parts = d.get("camparam", {}).get("Intrinsics", {}).get("data", "").split()
        img_w = float(parts[2]) * 2 if len(parts) > 2 else 1920.0
        img_h = float(parts[5]) * 2 if len(parts) > 5 else 1080.0
    except:
        img_w, img_h = 1920.0, 1080.0

    frames = []
    for fname in json_files:
        try:
            with open(os.path.join(folder_path, fname), encoding="utf-8") as f:
                data = json.load(f)
            people = data.get("people", {})
            kp = people.get("hand_right_keypoints_2d") or people.get("hand_left_keypoints_2d")
            if kp and len(kp) >= 63:
                xy = np.array([kp[j] for j in range(len(kp)) if j % 3 != 2], dtype=np.float32)[:42]
                xy[0::2] /= img_w
                xy[1::2] /= img_h
                xy[0::2] -= xy[0]
                xy[1::2] -= xy[1]
                frames.append(xy)
            else:
                frames.append(np.zeros(FEATURE_DIM, dtype=np.float32))
        except:
            frames.append(np.zeros(FEATURE_DIM, dtype=np.float32))
    return frames

# ── Sliding window → multiple sequences ────────
def sliding_window(frames):
    seqs = []
    for start in range(0, len(frames) - MAX_FRAMES + 1, STRIDE):
        seq = np.array(frames[start:start + MAX_FRAMES], dtype=np.float32)
        seqs.append(seq)
    return seqs

# ── Build dataset ──────────────────────────────
def build_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    folders = [f for f in os.listdir(JSON_ROOT) if os.path.isdir(os.path.join(JSON_ROOT, f))]
    target = [(f, get_label(f)) for f in folders if get_label(f) is not None]
    print(f"Target folders: {len(target)}")

    X, y = [], []
    for fname, label in target:
        fpath = os.path.join(JSON_ROOT, fname)
        frames = load_frames(fpath)
        seqs = sliding_window(frames)
        print(f"  {fname} -> {label}: {len(frames)} frames -> {len(seqs)} sequences")
        for seq in seqs:
            X.append(seq)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    np.save(os.path.join(OUTPUT_DIR, "X_json.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y_json.npy"), y)
    print(f"\nSaved! X: {X.shape}, y: {y.shape}")
    return X, y

# ── Model ──────────────────────────────────────
class SignDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

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

# ── Train ──────────────────────────────────────
def train(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    label_info = {int(i): cls for i, cls in enumerate(le.classes_)}
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_info, f, ensure_ascii=False, indent=2)
    print(f"Label map: {label_info}")

    # 80/20 분할
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    print(f"Train: {len(X_tr)}  /  Val: {len(X_val)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SignLSTM(FEATURE_DIM, len(le.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    tr_loader  = DataLoader(SignDataset(X_tr, y_tr),   batch_size=32, shuffle=True)
    val_loader = DataLoader(SignDataset(X_val, y_val), batch_size=32)

    best_acc, patience, no_improve = 0.0, 15, 0

    for epoch in range(1, 201):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total   += len(yb)
        acc = correct / total

        if epoch % 10 == 0 or acc > best_acc:
            print(f"  Epoch {epoch:3d} | val_acc: {acc*100:.1f}%")

        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "label_map": label_info,
                "le_classes": list(le.classes_)
            }, MODEL_SAVE)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\nBest val acc: {best_acc*100:.1f}%")
    print(f"Model saved: {MODEL_SAVE}")

# ── Main ───────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  SignBridge - Sliding Window Training")
    print("  WORD1518=dongsaeng / WORD1519=busang")
    print("=" * 50)

    x_path = os.path.join(OUTPUT_DIR, "X_json.npy")
    y_path = os.path.join(OUTPUT_DIR, "y_json.npy")

    if os.path.exists(x_path) and os.path.exists(y_path):
        print("Loading existing data...")
        X = np.load(x_path, allow_pickle=True)
        y = np.load(y_path, allow_pickle=True)
        print(f"X: {X.shape}, y: {y.shape}")
    else:
        print("Building dataset from JSON...")
        X, y = build_dataset()

    print(f"\nTraining with {len(X)} sequences...")
    train(X, y)
    print("Done!")
