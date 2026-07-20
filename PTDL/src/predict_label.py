import joblib
import re
from src.preprocess import preprocess
from src.negative import get_negative_only
clf = joblib.load("model_primary.pkl")
vectorizer = joblib.load("vectorizer_primary.pkl")
labels = joblib.load("labels.pkl")


RULES = {

    "Biển Đảo": [
        "biển", "vịnh", "đảo", "tắm", "hải sản", "san hô", "bãi tắm", "ven biển",
        "sóng", "cát trắng", "du thuyền", "cano", "ngắm bình minh", "ngắm hoàng hôn",
        "lướt ván", "kayak", "mũi", "hòn", "ghềnh", "eo biển", "đầm", "lặn", "làng chài",
        "hải đăng", "đại dương", "scuba", "snorkeling", "mô tô nước", "bãi sau", "bãi trước"
    ],
    "Tâm linh": [
        # Nguyên bản của bạn
        "chùa", "đền", "miếu", "phủ", "điện", "am", "thiền", "thiền viện",
        "linh thiêng", "cầu an", "hành hương", "dâng hương", "tín ngưỡng",
        "thờ", "thánh", "tâm linh", "yên tịnh",

        # Bổ sung về Địa danh/Kiến trúc
        "tượng phật", "lễ phật", "đại tùng lâm", "tịnh xá", "nhà thờ", "tháp",
        "cổ tự", "phật giáo", "công giáo", "thánh thất", "tòa thánh", "từ đường",

        # Bổ sung về Hoạt động/Nghi lễ
        "chiêm bái", "vãn cảnh", "lễ hội", "trẩy hội", "phóng sinh", "cầu may",
        "xin lộc", "cúng bái", "kinh kệ", "tụng kinh", "ăn chay", "trải nghiệm tu hành",

        # Bổ sung về Cảm giác/Không gian
        "thanh tịnh", "tĩnh mịch", "an nhiên", "huyền bí", "tôn nghiêm",
        "trang nghiêm", "thoát tục", "bình yên" ,"chùa chiền"

],
    "Văn hóa": [
        "di tích", "lịch sử", "bảo tàng", "cổ kính",
        "kiến trúc", "triều đại", "phố cổ",
        "làng nghề", "truyền thống", "di sản",
        "unesco", "nhà cổ", "đình làng",
        "làng nghề", "nghề truyền thống", "thủ công",
        "quy trình", "sản xuất", "chế biến",
        "phơi", "làm mỳ", "làm bánh",
        "gia truyền", "truyền thống",
        "nghề làm", "người dân địa phương","văn hóa"
],
    "Sinh thái - Thiên nhiên": [
        "rừng", "vườn quốc gia", "thác", "thiên nhiên", "sinh thái", "trong lành",
        "hoang sơ", "suối", "hồ", "đồi chè", "ruộng bậc thang", "khu bảo tồn",
        "đa dạng sinh học", "miệt vườn", "sông nước", "rừng ngập mặn", "vườn trái cây",
        "thực vật", "động vật", "không khí", "thung lũng", "kênh rạch", "rừng nguyên sinh"
    ],
    "Mạo hiểm - Trải nghiệm": [
        "trekking", "leo núi", "hang động", "thử thách", "mạo hiểm", "chinh phục",
        "phượt", "bụi", "dù lượn", "thám hiểm", "vượt thác", "cắm trại", "camping",
        "đỉnh núi", "băng rừng", "vách đá", "địa hình", "zipline", "ngủ lều", "săn mây",
        "đèo", "vượt đèo", "bản địa","núi cao" ,"núi","băng rừng"
    ],
    "Giải trí - Nghỉ dưỡng": [
        "vui chơi", "resort", "nghỉ dưỡng", "cáp treo", "công viên", "giải trí",
        "thư giãn", "spa", "massage", "chữa lành", "healing", "yoga", "khoáng nóng",
        "onsen", "tắm bùn", "sang trọng", "tiện nghi", "đẳng cấp", "mua sắm",
        "casino", "hồ bơi", "villas", "complex", "vinpearl", "sunworld","homestay","nhẹ nhàng"
    ],
    "Ẩm thực": [
        "đặc sản", "món ăn", "ẩm thực", "nhà hàng", "quán ăn", "thưởng thức",
        "food tour", "đường phố", " ẩm thực truyền thống", "vị", "ngon", "ăn vặt", "vỉa hè"
    ]
}

def rule_score(text):
    text = preprocess(text)
    scores = {}
    for k, kws in RULES.items():
        cnt = sum(1 for kw in kws if re.search(rf"\b{kw}\b", text))
        if cnt > 0:
            scores[k] = cnt
    return scores

def get_label_by_ml(word):
    X = vectorizer.transform([word])
    probs = clf.predict_proba(X)[0]
    idx = probs.argmax()
    return labels[idx], probs[idx]

def apply_negative_penalty(user_text, label_scores, penalty=0.3):
    negs = get_negative_only(user_text)
    for neg in negs:
        matched = [lb for lb, kws in RULES.items() if neg in kws]
        if matched:
            for lb in matched:
                label_scores[lb] = max(0.0, label_scores[lb] - penalty)
        else:
            ml_lb, _ = get_label_by_ml(neg)
            label_scores[ml_lb] = max(0.0, label_scores[ml_lb] - penalty)
    return label_scores

def predict_primary(user_text):
    X = vectorizer.transform([preprocess(user_text)])
    probs = clf.predict_proba(X)[0]
    scores = dict(zip(labels, probs))

    r = rule_score(user_text)

    for k, v in r.items():

        scores[k] += min(0.15, 0.05 * v)

    scores = apply_negative_penalty(user_text, scores)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]