# SignBridge - Gesture mapping data
# 이 파일을 signbridge_realtime.py에 import하거나 내용을 복사해서 사용

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
    "Closed_Fist":  {"en": "Fist",             "ja": "拳",       "zh": "拳头"},
    "Open_Palm":    {"en": "Palm of the Hand", "ja": "手のひら", "zh": "手掌"},
    "Pointing_Up":  {"en": "above",            "ja": "上",       "zh": "多于"},
    "Thumb_Up":     {"en": "Good",             "ja": "いいね",   "zh": "伟大的"},
    "Thumb_Down":   {"en": "Bad",              "ja": "嫌い",     "zh": "不，我不想"},
    "Victory":      {"en": "V",                "ja": "V",        "zh": "V"},
    "ILoveYou":     {"en": "I Love You",       "ja": "愛してる", "zh": "我爱你"},
}
