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

# ── 다국어 텍스트 ──────────────────────────
TEXTS = {
    "ko": {
        "no_hand": "손을 보여주세요",
        "recognizing": "인식 중...",
        "live_cam": "📷 라이브 캠",
        "accuracy": "정확도",
        "analysis": "분석결과",
        "recent": "최근 인식",
        "subtitle": "실시간 번역 자막",
        "history": "기록",
        "learned": "학습된 수어 이미지",
        "show_sign": "수어를 보여주세요",
    },
    "en": {
        "no_hand": "Show your hand",
        "recognizing": "Recognizing...",
        "live_cam": "📷 Live Cam",
        "accuracy": "Accuracy",
        "analysis": "Result",
        "recent": "Recent",
        "subtitle": "Real-time Subtitle",
        "history": "History",
        "learned": "Learned Sign Image",
        "show_sign": "Show your sign",
    },
    "ja": {
        "no_hand": "手を見せてください",
        "recognizing": "認識中...",
        "live_cam": "📷 ライブカメラ",
        "accuracy": "精度",
        "analysis": "分析結果",
        "recent": "最近の認識",
        "subtitle": "リアルタイム字幕",
        "history": "履歴",
        "learned": "学習済み手話画像",
        "show_sign": "手話を見せてください",
    },
    "zh": {
        "no_hand": "请展示您的手",
        "recognizing": "识别中...",
        "live_cam": "📷 实时摄像",
        "accuracy": "准确度",
        "analysis": "分析结果",
        "recent": "最近识别",
        "subtitle": "实时字幕",
        "history": "历史",
        "learned": "已学习的手语图像",
        "show_sign": "请展示手语",
    },
}

GESTURE_KO = {
    "None": "손을 보여주세요", "Closed_Fist": "주먹", "Open_Palm": "손바닥",
    "Pointing_Up": "위로", "Thumb_Up": "좋아요", "Thumb_Down": "싫어요",
    "Victory": "브이", "ILoveYou": "사랑해요",
}
GESTURE_EN = {
    "None": "Show your hand", "Closed_Fist": "Fist", "Open_Palm": "Open Palm",
    "Pointing_Up": "Point Up", "Thumb_Up": "Good", "Thumb_Down": "Bad",
    "Victory": "V", "ILoveYou": "I Love You",
}
GESTURE_JA = {
    "None": "手を見せてください", "Closed_Fist": "拳", "Open_Palm": "手のひら",
    "Pointing_Up": "上", "Thumb_Up": "いいね", "Thumb_Down": "嫌い",
    "Victory": "V", "ILoveYou": "愛してる",
}
GESTURE_ZH = {
    "None": "请展示您的手", "Closed_Fist": "拳头", "Open_Palm": "手掌",
    "Pointing_Up": "多于", "Thumb_Up": "伟大的", "Thumb_Down": "不，我不想",
    "Victory": "V", "ILoveYou": "我爱你",
}

GESTURE_TRANS = {
    "Closed_Fist":  {"en": "Fist",          "ja": "拳",       "zh": "拳头"},
    "Open_Palm":    {"en": "Open Palm",     "ja": "手のひら", "zh": "手掌"},
    "Pointing_Up":  {"en": "Point Up",      "ja": "上",       "zh": "多于"},
    "Thumb_Up":     {"en": "Good",          "ja": "いいね",   "zh": "伟大的"},
    "Thumb_Down":   {"en": "Bad",           "ja": "嫌い",     "zh": "不，我不想"},
    "Victory":      {"en": "V",             "ja": "V",        "zh": "V"},
    "ILoveYou":     {"en": "I Love You",    "ja": "愛してる", "zh": "我爱你"},
}

WORD_LANG = {
    "dongsaeng": {"ko": "동생", "en": "Younger Sibling", "ja": "弟/妹", "zh": "弟弟/妹妹"},
    "busang":    {"ko": "부상", "en": "Injury",           "ja": "負傷",  "zh": "受伤"},
}

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

def get_gesture_text(gesture, lang):
    maps = {"ko": GESTURE_KO, "en": GESTURE_EN, "ja": GESTURE_JA, "zh": GESTURE_ZH}
    return maps.get(lang, GESTURE_KO).get(gesture, gesture)

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    lang = data.get('lang', 'ko')
    img_data = base64.b64decode(data['frame'].split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "invalid frame"})

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    if result.hand_landmarks:
        for lm in result.hand_landmarks:
            draw_hand(frame, lm, w, h)

    if result.gestures:
        new_g = stable(state["google_buffer"], result.gestures[0][0].category_name)
        prev_g = state["gesture"]
        state["gesture"] = new_g
        state["gesture_conf"] = result.gestures[0][0].score
        if new_g != prev_g and prev_g != "None":
            now = time.strftime("%H:%M")
            state["history"].insert(0, {
                "time": now,
                "ko": GESTURE_KO.get(prev_g, prev_g),
                "en": GESTURE_EN.get(prev_g, prev_g),
                "ja": GESTURE_JA.get(prev_g, prev_g),
                "zh": GESTURE_ZH.get(prev_g, prev_g),
            })
            state["history"] = state["history"][:5]
    else:
        
        state["gesture"] = stable(state["google_buffer"], "None")
        state["gesture_conf"] = 0.0
    

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
                if top != prev_w and top in WORD_LANG:
                    now = time.strftime("%H:%M")
                    state["history"].insert(0, {
                        "time": now,
                        "ko": WORD_LANG[top]["ko"],
                        "en": WORD_LANG[top]["en"],
                        "ja": WORD_LANG[top]["ja"],
                        "zh": WORD_LANG[top]["zh"],
                    })
                    state["history"] = state["history"][:5]
            else:
                state["word"] = ""
    else:
        state["frame_seq"].clear()
        state["word_buffer"].append("None")

    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    processed_frame = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    g = state["gesture"]
    trans = GESTURE_TRANS.get(g, {})
    word_entry = WORD_LANG.get(state["word"], {})

    # 현재 언어에 맞는 텍스트
    if word_entry:
        display = word_entry.get(lang, word_entry.get("ko", ""))
        conf = state["word_conf"]
    else:
        display = get_gesture_text(g, lang)
        conf = state["gesture_conf"]

    # 히스토리 현재 언어로
    history_lang = [{"time": h["time"], "text": h.get(lang, h.get("ko", ""))} for h in state["history"]]

    return jsonify({
        "frame": processed_frame,
        "display": display,
        "conf": round(conf * 100, 1),
        "is_word": bool(word_entry),
        "trans_en": trans.get("en", ""),
        "trans_ja": trans.get("ja", ""),
        "trans_zh": trans.get("zh", ""),
        "history": history_lang,
        "texts": TEXTS.get(lang, TEXTS["ko"]),
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
  <div class="header-title">◆ <span id="header-title">실시간 <b style="color:#fbbf24">수어</b> 번역 자막 시스템</span></div>
  <div class="live-badge"><div class="live-dot"></div> LIVE</div>
</div>
<div class="lang-tabs">
  <div class="lang-tab active" onclick="setLang('ko')">🇰🇷 한국어</div>
  <div class="lang-tab" onclick="setLang('en')">🇺🇸 ENG</div>
  <div class="lang-tab" onclick="setLang('ja')">🇯🇵 JPN</div>
  <div class="lang-tab" onclick="setLang('zh')">🇨🇳 CHN</div>
</div>
<div class="main">
  <div class="panel">
    <div class="panel-title" id="label-learned">학습된 수어 이미지</div>
    <div class="ref-img" id="ref-emoji">🤟</div>
    <div style="margin-top:12px;font-size:13px;color:#666;text-align:center" id="ref-label">수어를 보여주세요</div>
  </div>
  <div class="panel">
    <div class="panel-title" id="label-cam">📷 라이브 캠</div>
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
    <div class="panel-title" id="label-analysis">분석결과</div>
    <div class="result-box">
      <div class="result-word" id="result-word">—</div>
      <div class="result-conf" id="result-conf"></div>
    </div>
    <div style="margin-top:12px">
      <div style="font-size:11px;color:#555;margin-bottom:8px" id="label-recent">최근 인식</div>
      <div id="mini-history" style="display:flex;flex-direction:column;gap:4px"></div>
    </div>
  </div>
</div>
<div class="subtitle-bar">
  <div class="subtitle-label" id="label-subtitle">실시간 번역 자막</div>
  <div class="subtitle-arrow">→</div>
  <div class="subtitle-ko" id="sub-main">—</div>
  <div class="subtitle-trans">
    <span id="sub-en"></span>
    <span id="sub-ja"></span>
    <span id="sub-zh"></span>
  </div>
</div>
<div class="history-bar">
  <span style="color:#444;margin-right:8px" id="label-history">▶ 기록</span>
  <div id="history-list" style="display:flex;gap:0"></div>
</div>

<video id="video" style="display:none" autoplay playsinline></video>
<canvas id="canvas" style="display:none"></canvas>

<script>
let activeLang = 'ko';
let processing = false;

const HEADER_TITLES = {
  ko: '실시간 <b style="color:#fbbf24">수어</b> 번역 자막 시스템',
  en: 'Real-time <b style="color:#fbbf24">Sign Language</b> Subtitle System',
  ja: 'リアルタイム<b style="color:#fbbf24">手話</b>字幕システム',
  zh: '实时<b style="color:#fbbf24">手语</b>字幕系统',
};

function setLang(lang) {
  activeLang = lang;
  document.querySelectorAll('.lang-tab').forEach((tab, i) => {
    tab.classList.toggle('active', ['ko','en','ja','zh'][i] === lang);
  });
  document.getElementById('header-title').innerHTML = HEADER_TITLES[lang];
}

async function startCamera(){
  try{
    const stream = await navigator.mediaDevices.getUserMedia({video:true});
    const video = document.getElementById('video');
    video.srcObject = stream;
    document.getElementById('cam-status').style.display = 'none';
    document.getElementById('processed-frame').style.display = 'block';
    processFrame();
  } catch(e){
    document.getElementById('cam-status').textContent = 'Camera error: ' + e.message;
  }
}

async function processFrame(){
  if(processing){ requestAnimationFrame(processFrame); return; }
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  if(video.readyState < 2){ requestAnimationFrame(processFrame); return; }

  canvas.width = 320; canvas.height = 240;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, 320, 240);
  const frameData = canvas.toDataURL('image/jpeg', 0.5);

  processing = true;
  try{
    const r = await fetch('/process', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({frame: frameData, lang: activeLang})
    });
    const d = await r.json();

    document.getElementById('processed-frame').src = d.frame;

    // 정확도
    const accLabel = (d.texts && d.texts.accuracy) ? d.texts.accuracy : '정확도';
    document.getElementById('conf-text').textContent = `${accLabel} ${d.conf}%`;
    document.getElementById('conf-fill').style.width = d.conf + '%';

    // UI 라벨 업데이트
    if(d.texts) {
      document.getElementById('label-learned').textContent = d.texts.learned;
      document.getElementById('label-cam').textContent = d.texts.live_cam;
      document.getElementById('label-analysis').textContent = d.texts.analysis;
      document.getElementById('label-recent').textContent = d.texts.recent;
      document.getElementById('label-subtitle').textContent = d.texts.subtitle;
      document.getElementById('label-history').textContent = '▶ ' + d.texts.history;
      document.getElementById('ref-label').textContent = d.texts.show_sign;
    }

    // 메인 결과
    document.getElementById('result-word').textContent = d.display || '—';
    document.getElementById('result-conf').textContent = d.conf > 0 ? d.conf+'%' : '';

    // 자막 바
    document.getElementById('sub-main').textContent = d.display || '—';

    // 번역 표시 (현재 언어 제외)
    document.getElementById('sub-en').textContent = (activeLang !== 'en' && d.trans_en) ? 'EN: '+d.trans_en : '';
    document.getElementById('sub-ja').textContent = (activeLang !== 'ja' && d.trans_ja) ? '| JA: '+d.trans_ja : '';
    document.getElementById('sub-zh').textContent = (activeLang !== 'zh' && d.trans_zh) ? '| ZH: '+d.trans_zh : '';

    // 하단 기록 (현재 언어)
    const histEl = document.getElementById('history-list');
    histEl.innerHTML = d.history.map(h=>`<span class="history-item"><span class="history-time">${h.time}</span>${h.text}</span>`).join('');

    // 최근 인식 (현재 언어)
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
    print("브라우저에서: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
