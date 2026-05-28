# SignBridge - Flask Web Server (Client Camera Version)
from flask import Flask, Response, render_template_string, jsonify, request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pickle, json, time, torch, torch.nn as nn, base64
from collections import deque, Counter

app = Flask(__name__)

BASE         = r"C:\Users\AISW_203_101\Desktop\Team_SignBridge\model"
OUTPUT_DIR   = r"C:\Users\AISW_203_101\Desktop\Team_SignBridge\keypoints_output"
GOOGLE_MODEL = BASE + r"\gesture_recognizer.task"
NUMBER_MODEL = BASE + r"\number_model.pkl"
NUMBER_LABEL = BASE + r"\number_labels.pkl"
WORD_MODEL   = r"C:\Users\AISW_203_101\Desktop\Team_SignBridge\sign_model.pt"
LABEL_MAP_F  = OUTPUT_DIR + r"\label_map.json"

MAX_FRAMES  = 30
FEATURE_DIM = 42

GESTURE_KO = {
    "None": "손을 보여주세요", "Closed_Fist": "주먹", "Open_Palm": "손바닥",
    "Pointing_Up": "위로", "Thumb_Up": "좋아요", "Thumb_Down": "싫어요",
    "Victory": "브이", "ILoveYou": "사랑해요",
}
GESTURE_TRANS = {
    "Closed_Fist":  {"en": "Fist",             "ja": "拳",       "zh": "拳头"},
    "Open_Palm":    {"en": "Palm of the Hand", "ja": "手のひら", "zh": "手掌"},
    "Pointing_Up":  {"en": "above",            "ja": "上",       "zh": "多于"},
    "Thumb_Up":     {"en": "Good",             "ja": "いいね",   "zh": "伟大的"},
    "Thumb_Down":   {"en": "Bad",              "ja": "嫌い",     "zh": "不，我不想"},
    "Victory":      {"en": "V",                "ja": "V",        "zh": "V"},
    "ILoveYou":     {"en": "I Love You",       "ja": "愛してる", "zh": "我爱你"},
}
WORD_KO = {"dongsaeng": "동생", "busang": "부상"}

class SignLSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, 128, batch_first=True)
        self.drop1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.drop2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x[:, -1, :])
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ── 모델 로드 ──────────────────────────────
base_options = python.BaseOptions(model_asset_path=GOOGLE_MODEL)
recognizer = vision.GestureRecognizer.create_from_options(
    vision.GestureRecognizerOptions(base_options=base_options, num_hands=2))

try:
    with open(NUMBER_MODEL, "rb") as f: number_model = pickle.load(f)
    with open(NUMBER_LABEL, "rb") as f: number_labels = pickle.load(f)
    NUMBER_OK = True
except: NUMBER_OK = False

try:
    ckpt = torch.load(WORD_MODEL, map_location="cpu", weights_only=False)
    with open(LABEL_MAP_F, encoding="utf-8") as f: word_label_map = json.load(f)
    word_model = SignLSTM(FEATURE_DIM, len(ckpt["le_classes"]))
    word_model.load_state_dict(ckpt["model"])
    word_model.eval()
    WORD_OK = True
except Exception as e:
    WORD_OK = False
    print(f"Word model failed: {e}")

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
]

# ── 세션별 상태 (간단하게 단일 세션) ──────
state = {
    "gesture": "None", "gesture_conf": 0.0,
    "word": "", "word_conf": 0.0,
    "history": [],
    "frame_seq": deque(maxlen=MAX_FRAMES),
    "word_buffer": deque(maxlen=10),
    "google_buffer": deque(maxlen=8),
}

def stable(buf, val):
    buf.append(val)
    return Counter(buf).most_common(1)[0][0]

def draw_hand(frame, lms, w, h):
    pts = [(int(lm.x*w), int(lm.y*h)) for lm in lms]
    for a,b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0,200,150), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, (251,191,36), -1)

# ── 프레임 처리 API ──────────────────────
@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    img_data = base64.b64decode(data['frame'].split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "invalid frame"})

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    # 손 그리기
    if result.hand_landmarks:
        for lm in result.hand_landmarks:
            draw_hand(frame, lm, w, h)

    # 구글 제스처
    if result.gestures:
        new_g = stable(state["google_buffer"], result.gestures[0][0].category_name)
        prev_g = state["gesture"]
        state["gesture"] = new_g
        state["gesture_conf"] = result.gestures[0][0].score
        if new_g != prev_g and prev_g != "None":
            now = time.strftime("%H:%M")
            state["history"].insert(0, {"time": now, "text": GESTURE_KO.get(prev_g, prev_g)})
            state["history"] = state["history"][:5]
    else:
        prev_g = state["gesture"]
        state["gesture"] = stable(state["google_buffer"], "None")
        state["gesture_conf"] = 0.0
        if prev_g != "None":
            now = time.strftime("%H:%M")
            state["history"].insert(0, {"time": now, "text": GESTURE_KO.get(prev_g, prev_g)})
            state["history"] = state["history"][:5]

    # 수어 단어 예측
    if WORD_OK and result.hand_landmarks:
        lms = result.hand_landmarks[0]
        flat = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32).flatten()
        flat[0::2] -= flat[0]; flat[1::2] -= flat[1]
        state["frame_seq"].append(flat)
        if len(state["frame_seq"]) >= MAX_FRAMES:
            seq = torch.tensor(np.array(state["frame_seq"])[np.newaxis], dtype=torch.float32)
            with torch.no_grad():
                probs = torch.softmax(word_model(seq), dim=1)[0]
            idx = probs.argmax().item()
            lbl = word_label_map[str(idx)]
            conf = float(probs[idx])
            if conf >= 0.70:
                state["word_buffer"].append(lbl)
            else:
                state["word_buffer"].append("None")
            counts = Counter(state["word_buffer"])
            top, cnt = counts.most_common(1)[0]
            if top != "None" and cnt >= 6:
                prev_w = state["word"]
                state["word"] = top
                state["word_conf"] = cnt / len(state["word_buffer"])
                if top != prev_w and top in WORD_KO:
                    now = time.strftime("%H:%M")
                    state["history"].insert(0, {"time": now, "text": WORD_KO[top]})
                    state["history"] = state["history"][:5]
            else:
                state["word"] = ""
    else:
        state["frame_seq"].clear()
        state["word_buffer"].append("None")

    # 처리된 프레임 반환
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    processed_frame = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    g = state["gesture"]
    trans = GESTURE_TRANS.get(g, {})
    word_ko = WORD_KO.get(state["word"], "")
    display = word_ko if word_ko else GESTURE_KO.get(g, "손을 보여주세요")
    conf = state["word_conf"] if word_ko else state["gesture_conf"]

    return jsonify({
        "frame": processed_frame,
        "display": display,
        "conf": round(conf * 100, 1),
        "is_word": bool(word_ko),
        "trans_en": trans.get("en", ""),
        "trans_ja": trans.get("ja", ""),
        "trans_zh": trans.get("zh", ""),
        "history": state["history"],
    })

@app.route('/')
def index():
    return HTML

HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SignBridge</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#0a0a0a;color:#f0f0f0;font-family:'Noto Sans KR',sans-serif;min-height:100vh}
  .header{display:flex;align-items:center;justify-content:space-between;padding:16px 24px;border-bottom:1px solid #222}
  .header-title{display:flex;align-items:center;gap:10px;font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:700;color:#fbbf24}
  .header-title span{color:#f0f0f0}
  .live-badge{display:flex;align-items:center;gap:6px;font-size:13px;color:#fbbf24;font-weight:700}
  .live-dot{width:8px;height:8px;border-radius:50%;background:#ef4444;animation:pulse 1.5s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
  .lang-tabs{display:flex;gap:8px;padding:12px 24px;border-bottom:1px solid #222}
  .lang-tab{padding:6px 16px;border-radius:6px;font-size:13px;font-weight:500;cursor:pointer;border:1px solid #333;background:transparent;color:#888;transition:all .2s}
  .lang-tab.active{background:#fbbf24;color:#0a0a0a;border-color:#fbbf24}
  .main{display:grid;grid-template-columns:1fr 2fr 1fr;gap:16px;padding:16px 24px}
  .panel{background:#111;border:1px solid #222;border-radius:12px;padding:16px}
  .panel-title{font-size:12px;color:#666;margin-bottom:12px;font-weight:500}
  .ref-img{width:100%;aspect-ratio:1;background:#1a1a1a;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:64px}
  .cam-wrap{position:relative;background:#000;border-radius:8px;overflow:hidden}
  .cam-wrap img{width:100%;display:block;border-radius:8px}
  .conf-bar-wrap{padding:8px 0 4px}
  .conf-label{font-size:12px;color:#fbbf24;font-weight:700;margin-bottom:6px}
  .conf-bar{height:8px;background:#1e1e1e;border-radius:4px;overflow:hidden}
  .conf-fill{height:100%;background:linear-gradient(90deg,#fbbf24,#f59e0b);border-radius:4px;transition:width .3s}
  .result-box{background:#1a1a1a;border-radius:8px;padding:16px;text-align:center;min-height:120px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px}
  .result-word{font-size:28px;font-weight:700;color:#fbbf24;font-family:'Space Grotesk',sans-serif;word-break:keep-all;text-align:center}
  .result-conf{font-size:13px;color:#888}
  .subtitle-bar{background:#111;border-top:1px solid #222;padding:14px 24px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  .subtitle-label{font-size:12px;color:#666;white-space:nowrap}
  .subtitle-arrow{color:#fbbf24;font-weight:700}
  .subtitle-ko{font-size:16px;font-weight:700;color:#f0f0f0}
  .subtitle-trans{font-size:14px;color:#888;display:flex;gap:16px;flex-wrap:wrap}
  .history-bar{background:#0d0d0d;border-top:1px solid #1a1a1a;padding:10px 24px;display:flex;align-items:center;gap:6px;font-size:12px;color:#555;overflow:hidden}
  .history-item{white-space:nowrap;margin-right:16px}
  .history-time{color:#fbbf24;margin-right:4px}
  .cam-status{font-size:12px;color:#666;text-align:center;padding:40px;background:#0d0d0d;border-radius:8px}
</style>
</head>
<body>
<div class="header">
  <div class="header-title">◆ <span>실시간 <b style="color:#fbbf24">수어</b> 번역 자막 시스템</span></div>
  <div class="live-badge"><div class="live-dot"></div> LIVE</div>
</div>
<div class="lang-tabs">
  <div class="lang-tab active">🇰🇷 한국어</div>
  <div class="lang-tab">🇺🇸 ENG</div>
  <div class="lang-tab">🇯🇵 JPN</div>
  <div class="lang-tab">🇨🇳 CHN</div>
</div>
<div class="main">
  <div class="panel">
    <div class="panel-title">학습된 수어 이미지</div>
    <div class="ref-img" id="ref-emoji">🤟</div>
    <div style="margin-top:12px;font-size:13px;color:#666;text-align:center" id="ref-label">수어를 보여주세요</div>
  </div>
  <div class="panel">
    <div class="panel-title">📷 라이브 캠</div>
    <div class="cam-wrap">
      <img id="processed-frame" src="" alt="camera" style="display:none">
      <div class="cam-status" id="cam-status">카메라 시작 중...</div>
    </div>
    <div class="conf-bar-wrap">
      <div class="conf-label" id="conf-text">정확도 0%</div>
      <div class="conf-bar"><div class="conf-fill" id="conf-fill" style="width:0%"></div></div>
    </div>
  </div>
  <div class="panel">
    <div class="panel-title">분석결과</div>
    <div class="result-box">
      <div class="result-word" id="result-word">—</div>
      <div class="result-conf" id="result-conf"></div>
    </div>
    <div style="margin-top:12px">
      <div style="font-size:11px;color:#555;margin-bottom:8px">최근 인식</div>
      <div id="mini-history" style="display:flex;flex-direction:column;gap:4px"></div>
    </div>
  </div>
</div>
<div class="subtitle-bar">
  <div class="subtitle-label">실시간 번역 자막</div>
  <div class="subtitle-arrow">→</div>
  <div class="subtitle-ko" id="sub-ko">—</div>
  <div class="subtitle-trans">
    <span id="sub-en"></span>
    <span id="sub-ja"></span>
    <span id="sub-zh"></span>
  </div>
</div>
<div class="history-bar">
  <span style="color:#444;margin-right:8px">▶ 기록</span>
  <div id="history-list" style="display:flex;gap:0"></div>
</div>

<video id="video" style="display:none" autoplay playsinline></video>
<canvas id="canvas" style="display:none"></canvas>

<script>
const GESTURE_EMOJI = {
  'Closed_Fist':'✊','Open_Palm':'🖐','Pointing_Up':'☝️',
  'Thumb_Up':'👍','Thumb_Down':'👎','Victory':'✌️','ILoveYou':'🤟','None':'👋'
};
let activeLang = 'ko';
let processing = false;

document.querySelectorAll('.lang-tab').forEach((tab,i)=>{
  tab.addEventListener('click',()=>{
    document.querySelectorAll('.lang-tab').forEach(t=>t.classList.remove('active'));
    tab.classList.add('active');
    activeLang = ['ko','en','ja','zh'][i];
  });
});

// 카메라 시작
async function startCamera(){
  try{
    const stream = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}});
    const video = document.getElementById('video');
    video.srcObject = stream;
    document.getElementById('cam-status').style.display = 'none';
    document.getElementById('processed-frame').style.display = 'block';
    processFrame();
  } catch(e){
    document.getElementById('cam-status').textContent = '카메라 접근 실패: ' + e.message;
  }
}

// 프레임 서버로 전송
async function processFrame(){
  if(processing){ requestAnimationFrame(processFrame); return; }
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  if(video.readyState < 2){ requestAnimationFrame(processFrame); return; }

  canvas.width = 640; canvas.height = 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, 640, 480);
  const frameData = canvas.toDataURL('image/jpeg', 0.7);

  processing = true;
  try{
    const r = await fetch('/process', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({frame: frameData})
    });
    const d = await r.json();

    // 처리된 프레임 표시 (손 랜드마크 포함)
    document.getElementById('processed-frame').src = d.frame;

    document.getElementById('conf-text').textContent = `정확도 ${d.conf}%`;
    document.getElementById('conf-fill').style.width = d.conf + '%';

    let display = d.display;
    if(activeLang==='en' && d.trans_en) display = d.trans_en;
    if(activeLang==='ja' && d.trans_ja) display = d.trans_ja;
    if(activeLang==='zh' && d.trans_zh) display = d.trans_zh;

    document.getElementById('result-word').textContent = display || '—';
    document.getElementById('result-conf').textContent = d.conf > 0 ? d.conf+'%' : '';
    document.getElementById('sub-ko').textContent = d.display || '—';
    document.getElementById('sub-en').textContent = d.trans_en ? 'EN: '+d.trans_en : '';
    document.getElementById('sub-ja').textContent = d.trans_ja ? '| JA: '+d.trans_ja : '';
    document.getElementById('sub-zh').textContent = d.trans_zh ? '| ZH: '+d.trans_zh : '';

    const histEl = document.getElementById('history-list');
    histEl.innerHTML = d.history.map(h=>`<span class="history-item"><span class="history-time">${h.time}</span>${h.text}</span>`).join('');
    const miniHist = document.getElementById('mini-history');
    miniHist.innerHTML = d.history.slice(0,3).map(h=>`<div style="font-size:12px;color:#666;display:flex;justify-content:space-between"><span>${h.text}</span><span style="color:#444">${h.time}</span></div>`).join('');
  } catch(e){ console.error(e); }

  processing = false;
  requestAnimationFrame(processFrame);
}

startCamera();
</script>
</body>
</html>"""

if __name__ == '__main__':
    print("SignBridge 서버 시작!")
    print("같은 네트워크에서: http://192.168.x.x:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
