import google.generativeai as genai
import re
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
    
def answer_question(question,disease_name):

    prompt=f"""
    Hãy tưởng tượng bạn là một bệnh nhân đang bị bệnh {disease_name}.
    Hãy trả lời câu hỏi sau về triệu chứng của bạn: {question} nên nhớ là trả lời theo cách của một bệnh nhân đang mắc bệnh {disease_name}.
    Trả lời "có" hoặc "không" và giải thích lý do tại sao bạn lại trả lời như vậy hãy giải thích như 1 con người.
    Nếu câu trả lời mà bạn không đưa ra được thì chọn ngẫu nhiên giữa có và không
    Loại bỏ markdown và các ký tự không cần thiết trong câu trả lời của bạn. ví dụ "```json" , "```",**,...
    
"""
    try:
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        result = response.text.strip()
        result = re.sub(r"^(?:\w+)?\n|\n$", "", result).strip()
        # xóa xuống dòng thừa
        result = re.sub(r"\n+", " ", result).strip()
        if not result:
            return "Không có thông tin"
        return result
    except Exception as e:
        print(f"Lỗi khi tổng hợp thông tin: {e}")
        return "Xảy ra lỗi trong quá trình tổng hợp thông tin"
    

disease_name = ["Hắc lào", "Bệnh vẩy nến", "Bệnh chàm", "Bệnh zona", "Bệnh ghẻ"," Bệnh nấm da", "Bệnh viêm da tiếp xúc", "Bệnh viêm da cơ địa", "Bệnh nấm móng", "Bệnh mề đay",
                "Bệnh viêm da tiết bã", "Bệnh viêm da dị ứng", "Bệnh viêm da mủ", "Bệnh viêm da do côn trùng cắn", "Bệnh viêm da do ánh sáng", "Bệnh viêm da do thuốc", "Bệnh viêm da do hóa chất",
                "Bệnh viêm da do vi khuẩn", "Bệnh viêm da do virus", "Bệnh viêm da do nấm", "Bệnh viêm da do ký sinh trùng", "Bệnh viêm da do di truyền", "Bệnh viêm da do môi trường", "Bệnh viêm da do stress",
                "Bệnh viêm da do dinh dưỡng", "Bệnh viêm da do nội tiết", "Bệnh viêm da do tâm lý", "Bệnh viêm da do miễn dịch", "Bệnh viêm da do tuổi tác", "Bệnh viêm da do di truyền", "Bệnh viêm da do môi trường sống",]

def mainProcess():
    """
    Hàm chính để xử lý và tạo mô tả triệu chứng cho các bệnh.
    """
    for disease in disease_name:
        description = generate_description(disease)
        if description:
            print(f"Mô tả triệu chứng cho {disease}: {description} \n")
        else:
            print(f"Không thể tạo mô tả triệu chứng cho {disease}.")

if __name__ == "__main__":
    mainProcess()