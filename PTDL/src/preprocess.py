import re

VIET_STOPWORDS = {
    "là","và","có","cho","của","ở","khi","được","một","những",
    "với","rất","nhiều","này","đó","để","mình","chúng","tôi"
}

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(
        r"[^a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
        r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
        r"ùúụủũưừứựửữỳýỵỷỹđ\s]",
        " ",
        text
    )
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in VIET_STOPWORDS]
    return " ".join(tokens)
