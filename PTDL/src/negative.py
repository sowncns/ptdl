from google import genai

# Khởi tạo Client
client = genai.Client(api_key="AIzaSyAC9ygHnjgl_L929-hdHAOw-FC9v8X0X5U")

def get_negative_only(user_input):
    sys_instruct = """
    Nhiệm vụ: Phân tích câu nhập của người dùng và chỉ trích xuất các yếu tố họ KHÔNG muốn, KHÔNG thích hoặc muốn TRÁNH.
    Yêu cầu trả về: Một mảng JSON thuần túy (Array of strings). 
    Ví dụ: tôi thích đi biển không thích leo núi thì negative là núi ,hoặc tôi muốn leo núi thì negative là []
    Nếu không có yếu tố tiêu cực, trả về mảng rỗng [].
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config={
            "system_instruction": sys_instruct,
            "response_mime_type": "application/json"
        },
        contents=user_input
    )

    # Chuyển đổi chuỗi JSON trả về thành list trong Python
    import json
    return json.loads(response.text)

# --- Chạy thử ---
