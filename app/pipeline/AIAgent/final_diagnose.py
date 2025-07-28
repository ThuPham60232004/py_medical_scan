import os
from dotenv import load_dotenv
import google.generativeai as genai
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def select_final_diagnosis_with_llm(
    caption: str,
    labels: list[str],
    questions: list[str],
    answers: list[str],
    model=None
) -> str:
    if model is None:
        model = genai.GenerativeModel("gemini-2.5-pro")

    qa_text = "\n".join(
        [f"- {q}\n  → {a}" for q, a in zip(questions, answers)]
    )

    prompt = f"""
Bạn là bác sĩ da liễu. Dưới đây là mô tả ảnh tổn thương da, danh sách bệnh nghi ngờ và các thông tin phân biệt thu được từ người bệnh.

--- MÔ TẢ ẢNH ---
{caption}

--- CÁC BỆNH NGHI NGỜ ---
{', '.join(labels)}

--- CÂU TRẢ LỜI PHÂN BIỆT ---
{qa_text}

Dựa vào tất cả thông tin trên, hãy chọn ra bệnh hợp lý nhất từ danh sách bệnh nghi ngờ.  
**Chỉ trả lời tên bệnh chính xác duy nhất (không giải thích thêm).**
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip().split("\n")[0]
    except Exception as e:
        logging.error(f"Lỗi chọn nhãn cuối bằng Gemini: {e}")
        return ""
