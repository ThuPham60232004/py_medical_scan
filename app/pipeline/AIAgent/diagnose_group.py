import google.generativeai as genai
from PIL import Image
from typing import Optional
import logging
import os
from dotenv import load_dotenv
import numpy as np
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
def extract_label_name(label):
    if isinstance(label, tuple) and isinstance(label[0], (str, np.str_)):
        return str(label[0])
    elif isinstance(label, str):
        return label
    return str(label)  
def generate_diagnosis_with_gemini(description, sorted_labels):
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')  
        labels_only = [extract_label_name(label) for label in sorted_labels[:30]]
        labels_text = "\n".join([f"- {label}" for label in labels_only])
        print(labels_text)
        prompt = f"""
Bạn là một chuyên gia da liễu.

Dưới đây là mô tả tổn thương da từ ảnh đầu vào:

\"\"\"{description}\"\"\"

Hệ thống AI đã dự đoán một số bệnh có thể gặp (từ ảnh toàn phần và vùng tổn thương), cùng độ phù hợp:

{labels_text}

Lưu ý: Các nhãn bệnh có thể bị lặp lại hoặc không nhất quán. Bạn **không được dựa hoàn toàn vào nhãn**, mà phải **phân tích kỹ mô tả tổn thương** để đưa ra **phán đoán chính xác**.

---

## Nhiệm vụ của bạn:

Hãy phân loại bệnh vào đúng **một trong 4 nhóm bệnh chính** sau:
---

**1. fungal-infections (nhiễm nấm):**  
- Da khô, bong vảy mịn như phấn, thường có bờ rõ hoặc hình vòng.  
- Trung tâm tổn thương có thể lành hơn ngoại vi.  
- Không có mủ, không sưng nóng đỏ, ngứa nhẹ hoặc vừa.  
- Vị trí: da đầu, thân mình, chi, hoặc bẹn.

**2. virus (nhiễm virus):**  
- Mụn nước, bóng nước, sẩn hoặc loét nông.  
- Đau rát hoặc ngứa.  
- Phân bố dọc theo dây thần kinh hoặc đối xứng.  
- Không có mủ vàng hoặc vảy tiết dày.

**3. bacterial-infections (nhiễm vi khuẩn):**  
- Da sưng nóng đỏ đau, có mủ, đóng mày vàng, có thể hoại tử nhẹ.  
- Có thể đơn độc (đặc biệt ở da đầu), với vảy dày, màu trắng vàng, bề mặt sần sùi.  
- Bờ tổn thương rõ hoặc không đều, thường lan rộng nhanh.  
- Nếu không có mủ nhưng có màu trắng vàng, vẫn cần cân nhắc nhóm này.

**4. parasitic-infections (nhiễm ký sinh trùng):**  
- Ngứa nhiều, nhất là vào ban đêm.  
- Có sẩn nhỏ, rãnh, vết xước do gãi.  
- Vị trí thường gặp: kẽ ngón tay, bẹn, quanh rốn, mông.  
- Có thể lây lan nhanh qua tiếp xúc.

---
## Luật loại trừ:

- Nếu **không có mủ, sưng, đau, dịch vàng** → loại trừ **bacterial-infections**
- Nếu **không có ngứa dữ dội, đường hầm** → loại trừ **parasitic-infections**
- Nếu **không có ban đỏ dạng mụn nước, không phân bố đối xứng hay theo dải** → loại trừ **virus**
- Nếu **tổn thương có vảy, bờ rõ, không có dịch** → nghiêng về **fungal-infections**
---
## Lưu ý bổ sung:
- **fungal-infections** thường không có biểu hiện toàn thân, không sưng mủ, bờ tổn thương rõ, hình tròn hoặc loang lổ.  
- **bacterial-infections** thường **sưng**, **đỏ**, **đau**, có thể **chảy mủ**, **lở loét**.  
- **virus** thường gây **mụn nước**, **ban đỏ**, **dạng đối xứng** hoặc **dải da**.  
- **parasitic-infections** luôn **ngứa dữ dội**, có thể có **đường hầm**.  
---
## QUAN TRỌNG:
Bạn chỉ được trả về đúng **một dòng duy nhất**, là một trong các nhóm bệnh sau:
- fungal_infections  
- virus  
- bacterial_infections  
- parasitic_infections

❌ Không viết hoa, không thêm dấu câu, không thêm giải thích hoặc lý do.  
❌ Không dùng từ gần nghĩa như: "fungal", "viral", "bacteria", "parasite"
⚠️ Đặc biệt lưu ý: Nếu mô tả tổn thương ở vùng da đầu, mặt, cổ, có đặc điểm như: nhỏ, đơn lẻ hoặc rải rác, có vảy, không sưng nề hay có mủ rõ ràng, nhưng nằm gần nang lông hoặc chân tóc — **rất có thể là viêm nang lông (bacterial-infections)**. Trong trường hợp này, KHÔNG nên kết luận là fungal-infections chỉ vì có vảy.
⚠️ Nếu tổn thương nhỏ (<5mm), hình tròn, rải rác, không có vảy, không có mủ, không sưng nề và phân bố không đối xứng, cần cân nhắc nhóm **parasitic-infections** như ghẻ hoặc phản ứng do côn trùng cắn. KHÔNG nhầm sang fungal nếu KHÔNG có vảy rõ, ranh giới tăng sừng, hoặc mảng lớn lan tỏa. KHÔNG nhầm sang virus nếu KHÔNG có cụm sẩn, ban, mụn nước, hoặc tổn thương đa hình thái.
✅ Trả lời cuối cùng của bạn (chỉ 1 dòng):
"""

        response = model.generate_content(prompt)

        caption = response.text.replace("\n", " ").strip().lower()

        return caption

    except Exception as e:
        logging.error(f"❌ Lỗi khi tạo chẩn đoán với Gemini: {e}")
        return "unknown"
