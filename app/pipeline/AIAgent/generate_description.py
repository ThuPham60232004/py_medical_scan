import google.generativeai as genai
from PIL import Image
from typing import Optional
import logging
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
def generate_description_with_Gemini(image_path: str) -> Optional[str]:
    try:
        img = Image.open(image_path)
        new_size = (img.width * 2, img.height * 2)
        img = img.resize(new_size, Image.LANCZOS)  
        model = genai.GenerativeModel('gemini-2.5-pro')
        prompt = """
Hãy quan sát kỹ bức ảnh da bên dưới và mô tả lại những đặc điểm có thể nhận biết bằng mắt thường, bằng tiếng Việt chính xác và trung lập.

Nội dung mô tả nên bao gồm các yếu tố sau:

1. **Vị trí** tổn thương:  
   - Xuất hiện ở vùng nào? (ví dụ: mặt, cổ, lưng, lòng bàn tay, lòng bàn chân, kẽ ngón tay, vùng sinh dục...)

2. **Số lượng và kích thước**:  
   - Có bao nhiêu tổn thương? (một hay nhiều)
   - Ước lượng kích thước tổn thương (dưới 5mm, khoảng 1–2cm, lan rộng toàn vùng...)

3. **Màu sắc**:  
   - Màu chủ đạo là gì? (đỏ, hồng, tím, nâu, trắng, vàng mủ, v.v.)
   - Có đồng nhất không, hay có nhiều vùng màu khác nhau?

4. **Bề mặt da**:  
   - Tổn thương có trơn láng, khô, bong vảy, sần sùi, đóng mày, loét, hay có mụn nước, mủ?

5. **Bờ tổn thương**:  
   - Ranh giới rõ hay mờ? Bờ đều hay không đều? Có dạng vòng hay lan tỏa?

6. **Tính đối xứng**:  
   - Tổn thương có xuất hiện hai bên cơ thể một cách đối xứng không?

7. **Kiểu phân bố**:  
   - Tổn thương rải rác, tập trung thành cụm, theo mảng lớn, hay theo đường (ví dụ: dọc theo dây thần kinh)?

8. **Dấu hiệu bất thường khác**:  
   - Có sưng nề, chảy dịch, có mủ vàng, hoại tử, lở loét, hoặc các dấu hiệu đặc biệt như **đường hầm dưới da** không?

---

🎯 **Lưu ý cực kỳ quan trọng**:  
Lưu ý cực kỳ quan trọng:
Bạn chỉ nên mô tả những gì quan sát được bằng mắt thường trong ảnh. Không dự đoán, không suy luận đặc điểm có thể liên quan đến bệnh lý. Không đưa vào các mô tả gợi ý từ kinh nghiệm y khoa, chỉ mô tả trung lập.
- **Nhiễm nấm (fungal-infections)**: bờ rõ ràng, hình tròn/vòng, bong vảy nhẹ, thường ở vùng ẩm (bẹn, kẽ tay, chân).  
- **Virus (virus)**: mụn nước nhỏ, ban đỏ dạng đối xứng, tổn thương theo cụm hoặc theo dây thần kinh.  
- **Vi khuẩn (bacterial-infections)**: có mủ, sưng nóng đỏ đau, loét, vảy vàng, tổn thương sâu.  
- **Ký sinh trùng (parasitic-infections)**: ngứa dữ dội, có đường hầm dưới da, sẩn nhỏ hoặc mụn nước rải rác ở kẽ ngón, thắt lưng, mông.

Lưu ý:
- Không được đưa ra bất kỳ chẩn đoán hoặc suy luận y tế nào.
- Chỉ mô tả khách quan những gì nhìn thấy được trong ảnh.
- Không sử dụng bullet point, markdown hay ký hiệu đặc biệt.

Trả về kết quả dưới dạng đoạn văn y khoa mô tả, rõ ràng, trung lập.

"""
        response = model.generate_content([prompt, img])
        caption = response.text.replace("\n", " ").strip()
        return caption
    except Exception as e:
        logging.error(f"Lỗi khi tạo caption với Gemini: {e}")
        return None