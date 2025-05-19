import google.generativeai as genai
genai.configure(api_key="AIzaSyCcvSA4ncbJexc4WLKCzhHb6yytR0_Klsw")

def generate_description(disease_name: str) -> str:
    """
    Sinh mô tả triệu chứng của bệnh nhân dựa trên tên bệnh với Gemini.
    
    Args:
        disease_name (str): Tên bệnh.
    
    Returns:
        str: Mô tả triệu chứng theo định dạng yêu cầu hoặc None nếu lỗi.
    """
    if not disease_name or not isinstance(disease_name, str):
        print("Tên bệnh không hợp lệ.")
        return None

    prompt = f"""
    Bạn là một bệnh nhân đang bị bệnh {disease_name}.
    
    Hãy mô tả triệu chứng của bạn theo các yếu tố sau:
    - Vị trí xuất hiện trên cơ thể
    - Thời gian kéo dài
    - Hình dạng và màu sắc của vùng bị ảnh hưởng
    - Cảm giác mà bạn cảm nhận được (như ngứa, rát, đau...)
    - Triệu chứng có lan rộng không

    Hãy trả lời theo định dạng sau:
    "Triệu chứng xuất hiện ở [vị trí], đã kéo dài [thời gian]. 
    Vùng da có biểu hiện [hình dạng và màu sắc] và cảm giác [cảm giác]. 
    Triệu chứng: [lan rộng hay không]."

    Ví dụ:
    "Triệu chứng xuất hiện ở tay, đã kéo dài 2 tuần. 
    Vùng da có biểu hiện đỏ và cảm giác ngứa. Triệu chứng: không lan rộng."
    """

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content([prompt])
        text = response.text.strip()
        return text
    except Exception as e:
        print(f"Lỗi khi tạo mô tả với Gemini: {e}")
        return None