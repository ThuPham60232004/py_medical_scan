import cv2
import os
import numpy as np
import torch
import faiss
import json
from pathlib import Path
from google.cloud import storage
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv
import google.generativeai as genai
from transformers import pipeline
from huggingface_hub import login
import timm
from PIL import Image
from torchvision import transforms
from pathlib import Path
from typing import Optional
import logging
from PIL import Image
import textwrap
import re
from sklearn.metrics.pairwise import cosine_similarity
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

GCS_BUCKET = "kltn-2025"
GCS_IMAGE_PATH = "uploaded_images/"
GCS_KEY_PATH = storage.Client.from_service_account_json("app/gsc-key.json")

VECTOR_FILE = "static/processed/embedded_vectors.json"
GCS_FOLDER="handle_data"
GCS_DATASET = f"dataset"
GCS_DATASET_PATH = f"{GCS_DATASET}/dataset.json"
GCS_INDEX_PATH = f"{GCS_FOLDER}/faiss_index.bin"
GCS_LABELS_PATH = f"{GCS_FOLDER}/labels.npy"
GCS_TEXT_INDEX_PATH = f"{GCS_FOLDER}/faiss_text_index.bin"
GCS_TEXT_LABELS_PATH = f"{GCS_FOLDER}/text_labels.npy"
GCS_ANOMALY_INDEX_PATH = f"{GCS_FOLDER}/faiss_index_anomaly.bin"
GCS_ANOMALY_LABELS_PATH = f"{GCS_FOLDER}/labels_anomaly.npy"
LOCAL_INDEX_PATH = "app/static/faiss/faiss_index.bin"
LOCAL_LABELS_PATH = "app/static/labels/labels.npy"
LOCAL_TEXT_INDEX_PATH = "app/static/faiss/faiss_text_index.bin"
LOCAL_TEXT_LABELS_PATH = "app/static/labels/text_labels.npy"
LOCAL_ANOMALY_INDEX_PATH = "app/static/faiss/faiss_index_anomaly.bin"
LOCAL_ANOMALY_LABELS_PATH = "app/static/labels/labels_anomaly.npy"
LOCAL_DATASET_PATH = "app/static/json/dataset.json"
INDEX_DIM = 512  

index = None
labels = []
anomaly_index = None
anomaly_labels = []

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
vit_model = timm.create_model("vit_base_patch16_224", pretrained=True).to(device)
vit_model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
def download_from_gcs():
    storage_client = GCS_KEY_PATH
    bucket = storage_client.bucket(GCS_BUCKET)

    files_to_download = [
        (GCS_INDEX_PATH, LOCAL_INDEX_PATH),
        (GCS_LABELS_PATH, LOCAL_LABELS_PATH),
        (GCS_TEXT_INDEX_PATH, LOCAL_TEXT_INDEX_PATH),
        (GCS_TEXT_LABELS_PATH, LOCAL_TEXT_LABELS_PATH),
        (GCS_ANOMALY_INDEX_PATH, LOCAL_ANOMALY_INDEX_PATH),
        (GCS_ANOMALY_LABELS_PATH, LOCAL_ANOMALY_LABELS_PATH),
        (GCS_DATASET_PATH, LOCAL_DATASET_PATH),
    ]

    for gcs_path, local_path in files_to_download:
        blob = bucket.blob(gcs_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"Tải về {gcs_path} to {local_path}")
        
def upload_to_gcs(local_path, destination_blob_name):
    """Upload file lên Google Cloud Storage."""
    client = storage.Client.from_service_account_json("app/gsc-key.json")
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    print(f"Đã upload {local_path} lên GCS tại: gs://{GCS_BUCKET}/{destination_blob_name}")

def preprocess_image(image_path):
    """Tiền xử lý ảnh bằng Gaussian Blur và Canny Edge Detection."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    edges = cv2.Canny(equalized, 50, 150)
    return edges

def embed_image(image_path):
    """Nhúng ảnh thành vector sử dụng mô hình CLIP."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.cpu().numpy().astype(np.float32)
def generate_anomaly_map(image_path: str) -> Optional[np.ndarray]:
    """
    Sinh anomaly map từ ảnh đầu vào bằng ViT feature extractor và resize lại theo kích thước ảnh gốc.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        original_size = img.size  
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            features = vit_model.forward_features(img_tensor)  
        feature_map = features.mean(dim=1).squeeze().cpu().numpy()
        anomaly_map = (feature_map - np.min(feature_map)) / (np.ptp(feature_map) + 1e-6)
        anomaly_map = (anomaly_map * 255).astype(np.uint8)
        anomaly_map_resized = cv2.resize(anomaly_map, original_size, interpolation=cv2.INTER_CUBIC)

        return anomaly_map_resized

    except Exception as e:
        print(f"Lỗi tạo Anomaly Map: {e}")
        return None

def embed_anomaly_map(anomaly_map_path: str):
    """Nhúng anomaly map thành vector sử dụng mô hình CLIP."""
    anomaly_map = cv2.imread(anomaly_map_path, cv2.IMREAD_GRAYSCALE)
    if anomaly_map is None:
        return None
    anomaly_map_rgb = cv2.cvtColor(anomaly_map, cv2.COLOR_GRAY2RGB)
    inputs = processor(images=anomaly_map_rgb, return_tensors="pt").to(device)
    
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)

    return embedding.cpu().numpy().astype(np.float32)
def load_faiss_index():
    """Tải FAISS Index và nhãn bệnh từ file."""
    global index, labels,anomaly_index,anomaly_labels
    if os.path.exists(LOCAL_INDEX_PATH):
        try:
            index = faiss.read_index(LOCAL_INDEX_PATH)
            print(f"FAISS Index tải thành công! Tổng số vector: {index.ntotal}")
        except Exception as e:
            print(f"Lỗi tải FAISS Index: {e}")
            index = None
    else:
        print("FAISS Index không tồn tại!")

    if os.path.exists(LOCAL_TEXT_LABELS_PATH):
        labels = np.load(LOCAL_TEXT_LABELS_PATH, allow_pickle=True).tolist()
        print(f"Đã tải {len(labels)} nhãn bệnh từ labels.npy")
    else:
        print("labels.npy không tồn tại!")
        
    if os.path.exists(LOCAL_ANOMALY_INDEX_PATH):
        try:
            anomaly_index = faiss.read_index(LOCAL_ANOMALY_INDEX_PATH)
            print(f"FAISS Anomaly Index tải thành công! Tổng số vector: {anomaly_index.ntotal}")
        except Exception as e:
            print(f"Lỗi tải FAISS Anomaly Index: {e}")
            anomaly_index = None
    else:
        print("FAISS Anomaly Index không tồn tại!")
    if os.path.exists(LOCAL_ANOMALY_LABELS_PATH):
        anomaly_labels = np.load(LOCAL_ANOMALY_LABELS_PATH, allow_pickle=True).tolist()
        print(f"Đã tải {len(anomaly_labels)} nhãn bệnh từ labels-anomaly.npy")
    else:
        print("labels-anomaly.npy không tồn tại!")

def search_similar_images(query_vector, top_k=5):
    """Tìm ảnh tương tự bằng FAISS Index."""
    if index is None or index.ntotal == 0:
        print("FAISS index trống!")
        return []

    distances, indices = index.search(query_vector, top_k)
    print(f"Chỉ số tìm thấy: {indices}")
    similar_labels = []
    
    for i in indices[0]:
        if 0 <= i < len(labels):
            label_filename = list(labels.keys())[i]
            similar_labels.append(labels[label_filename])
        else:
            print(f"Index {i} vượt phạm vi labels ({len(labels)})!")
            similar_labels.append("unknown")

    return similar_labels
def search_anomaly_images(query_vector, top_k=5):
    """Tìm ảnh anomaly map tương tự bằng FAISS Index."""
    if anomaly_index is None or anomaly_index.ntotal == 0:
        print("FAISS Anomaly Index trống!")
        return []
    distances, indices = anomaly_index.search(query_vector, top_k)
    print(f"Chỉ số tìm thấy: {indices}")
    similar_labels = []
    
    for i in indices[0]:
        if 0 <= i < len(anomaly_labels):
            label_filename = list(anomaly_labels.keys())[i]
            similar_labels.append(anomaly_labels[label_filename])
        else:
            print(f"Index {i} vượt phạm vi labels-anomaly ({len(anomaly_labels)})!")
            similar_labels.append("unknown")

    return similar_labels
VALID_KEYWORDS = {
    "location": ["tay", "chân", "đầu gối", "cổ tay", "bụng", "lưng", "mặt", "cổ", "ngực"],
    "duration": ["ngày", "tuần", "tháng", "năm", "hôm nay", "hôm qua", "vài ngày", "lâu rồi", 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "appearance": ["đỏ", "sưng", "mưng mủ", "mụn", "nổi mẩn", "tróc vảy", "thâm"],
    "feeling": ["ngứa", "đau", "rát", "nhức", "khó chịu"],
    "spreading": ["có", "không"],
}

def extract_keywords(text, field):
    if not isinstance(text, str):
        text = str(text)
    keywords = {
        "location": ["tay", "chân", "đầu gối", "cổ tay", "ngón", "mặt", "bụng", "lưng"],
        "duration": ["ngày", "tuần", "tháng", "năm"],
        "appearance": ["đỏ", "sưng", "mụn", "loét", "nổi cục", "vàng", "trắng"],
        "feeling": ["ngứa", "đau", "rát", "khó chịu"],
        "spreading": ["có", "không"]
    }
    matched = [word for word in keywords.get(field, []) if word in text.lower()]
    return ", ".join(matched) if matched else text

def collect_user_description():
    print("Thu thập mô tả của bệnh nhân (hoặc nhấn Enter để bỏ qua):\n")
    try:
        print("Bắt đầu thu thập mô tả bệnh, bệnh nhân vui lòng trả lời các câu hỏi sau:")
        location = input("Cho mình hỏi bạn, bạn có thể cho biết vị trí của bệnh không? (Ví dụ: đầu gối, cổ tay,...)\n")
        duration = input("Thời gian bạn bị bệnh là bao lâu rồi? (Ví dụ: 1 tuần, 2 tháng,...)\n")
        appearance = input("Hình dạng của bệnh như thế nào? (Ví dụ: đỏ, sưng,...)\n")
        feeling = input("Bạn cảm thấy như thế nào? (Ví dụ: đau, ngứa,...)\n")
        spreading = input("Bệnh có lan rộng không? (Ví dụ: có, không)\n")
        location = extract_keywords(location, "location")
        duration = extract_keywords(duration, "duration")
        appearance = extract_keywords(appearance, "appearance")
        feeling = extract_keywords(feeling, "feeling")
        spreading = extract_keywords(spreading, "spreading")

        description = (
            f"Triệu chứng xuất hiện ở {location}, đã kéo dài {duration}. "
            f"Vùng da có biểu hiện {appearance} và cảm giác {feeling}. "
            f"Triệu chứng: {spreading} lan rộng.")
        print("Đây là mô tả bệnh bạn đã cung cấp:\n")
        print(description)

        confirm = input("Bạn có muốn xác nhận mô tả này không? (y/n): ").strip().lower()
        if confirm == 'y':
            print("Mô tả bệnh của bạn đã được ghi nhận")
            return description
        else:
            retry = input("Bạn muốn nhập lại mô tả bệnh? (y/n): ").strip().lower()
            if retry == 'y':
                return collect_user_description()
            else:
                print("Mô tả bệnh đã bị bỏ qua.")
                return None
    except Exception as e:
        print(f"Lỗi thu thập mô tả: {e}")
        return None

def generate_description_with_Gemini(image_path):
    try:
        img = Image.open(image_path)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = """
        Mô tả bức ảnh này bằng tiếng Việt, đây là ảnh y khoa nên hãy mô tả thật kỹ.
        Chỉ tập trung vào mô tả lâm sàng, không đưa ra chẩn đoán hay kết luận.
        Hãy mô tả các đặc điểm sau:
        - Vị trí của tổn thương (ví dụ: lòng bàn tay, mu bàn tay, ngón chân...)
        - Kích thước tổn thương (ước lượng theo mm hoặc cm)
        - Màu sắc (đồng nhất hay nhiều màu, đỏ, tím, hồng, v.v.)
        - Kết cấu bề mặt da (mịn, sần sùi, có vảy, loét...)
        - Độ rõ nét của các cạnh tổn thương (rõ ràng hay mờ, lan tỏa)
        - Tính đối xứng (tổn thương có đối xứng 2 bên hay không)
        - Phân bố (rải rác, tập trung thành đám, theo đường…)
        - Các đặc điểm bất thường khác nếu có (chảy máu, vảy, mụn nước, sưng nề…)
        Chỉ mô tả những gì có thể nhìn thấy trong ảnh, không đưa ra giả định hay chẩn đoán y khoa.
        Xóa mark down và các ký tự đặc biệt trong kết quả.
        """
        response = model.generate_content([prompt, img])
        caption = response.text.replace("\n", " ")
        return caption
    except Exception as e:
        print(f"Lỗi khi tạo caption với Gemini: {e}")
        return None
def combine_labels(normal_labels: list, anomaly_labels: list) -> str:
    """
    Gộp danh sách nhãn từ ảnh gốc và ảnh anomaly thành một chuỗi mô tả duy nhất, cách nhau bằng dấu cách.
    Args:
        normal_labels (list): Các nhãn từ ảnh gốc.
        anomaly_labels (list): Các nhãn từ anomaly map.
    Returns:
        str: Chuỗi mô tả tổng hợp sau chuẩn hóa, mỗi nhãn cách nhau một dấu cách.
    """
    all_labels = normal_labels + anomaly_labels

    return " ".join(all_labels).strip()


def generate_medical_entities(image_caption, user_description):
    combined_description = f"1. Mô tả từ người dùng: {user_description}. 2. Mô tả từ ảnh: {image_caption}."
    print(combined_description)

    prompt = textwrap.dedent(f"""
        Tôi có 2 đoạn mô tả sau về một vùng da bị bất thường: {combined_description}
        Hãy chuẩn hóa cả hai mô tả, loại bỏ từ dư thừa, hợp nhất lại, và trích xuất các đặc trưng y khoa quan trọng.
        Mỗi đặc trưng nên được gắn nhãn thuộc một trong ba loại sau:
        - "Triệu chứng": mô tả biểu hiện, dấu hiệu lâm sàng (ví dụ: phát ban, ngứa, đỏ, bong tróc…)
        - "Vị trí xuất hiện": vùng cơ thể bị ảnh hưởng (ví dụ: mu bàn tay, cẳng chân, ngón tay…)
        - "Nguyên nhân": yếu tố gây ra tình trạng đó nếu có xuất hiện trong mô tả (ví dụ: côn trùng cắn, dị ứng, tiếp xúc hóa chất…)
        Trả về kết quả dạng JSON Array. Mỗi phần tử là một object gồm:
        - "entity": cụm từ y khoa
        - "type": "Triệu chứng", "Vị trí xuất hiện", hoặc "Nguyên nhân"
        Ví dụ đầu ra:
        [
          {{ "entity": "vết đỏ", "type": "Triệu chứng" }},
          {{ "entity": "cẳng chân", "type": "Vị trí xuất hiện" }},
          {{ "entity": "dị ứng thời tiết", "type": "Nguyên nhân" }}
        ]
        Chỉ liệt kê các đặc trưng có trong mô tả. Không suy luận thêm.
    """)
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content([prompt])
        text = response.text.strip()
        clean_text = re.sub(r"```(?:json)?|```", "", text).strip()
        result = json.loads(clean_text)
        decoded_result = json.dumps(result, ensure_ascii=False)
        return decoded_result 

    except Exception as e:
        print(f"Lỗi khi tạo mô tả với Gemini: {e}")
        return None
 
def compare_descriptions_and_labels(description, label):

    prompt = textwrap.dedent(f"""
        Mô tả: "{description}"
        Nhãn: "{label}"
        So sánh sự khác biệt giữa mô tả và nhãn bệnh. Sau đó, tạo ra 3 câu hỏi giúp phân biệt chính xác hơn.
        Trả về kết quả theo định dạng:
        1. ...
        2. ...
        3. ...
    """)

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content([prompt])
        text = response.text.strip()
        questions = re.findall(r"\d+\.\s+(.*)", text)
        return questions
    except Exception as e:
        print(f"Lỗi khi gọi Gemini: {e}")
        return []
def ask_user_questions(questions):
    answers = []
    for idx, q in enumerate(questions, 1):
        print(f"Câu {idx}: {q}")
        answer = input("Trả lời của bạn: ")
        answers.append(f"Câu hỏi: {q}\nTrả lời: {answer}")
    return "\n\n".join(answers)
       
def process_image(image_path):
    """Xử lý ảnh đầu vào, tạo Anomaly Map và nhúng Anomaly Map, tìm kiếm nhãn bệnh."""
    processed = preprocess_image(image_path)
    anomaly_map = generate_anomaly_map(image_path)
    if processed is not None:
        processed_dir = Path("app/static/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)

        processed_path = processed_dir / f"{Path(image_path).stem}_processed.jpg"
        cv2.imwrite(str(processed_path), processed)
        upload_to_gcs(str(processed_path), GCS_IMAGE_PATH + str(processed_path.name))
        embedding = embed_image(image_path)
        if embedding is not None:
            result_labels = search_similar_images(embedding)
            print("Kết quả tìm kiếm ảnh tương tự:", result_labels)

        anomaly_map = generate_anomaly_map(image_path)
        if anomaly_map is not None:
            anomaly_map_path = processed_dir / f"{Path(image_path).stem}_anomaly_map.jpg"
            cv2.imwrite(str(anomaly_map_path), anomaly_map)
            anomaly_map_embedding = embed_anomaly_map(str(anomaly_map_path))
            if anomaly_map_embedding is not None:
                anomaly_result_labels = search_similar_images(anomaly_map_embedding)
                print("Kết quả tìm kiếm từ Anomaly Map:", anomaly_result_labels)
    final_labels = combine_labels(result_labels, anomaly_result_labels)
    print("Chuỗi mô tả bệnh tổng hợp:", final_labels)
    return final_labels, result_labels, anomaly_result_labels

def filter_incorrect_labels_by_user_description(description: str, labels: list[str]) -> str:
    prompt = textwrap.dedent(f"""
        Mô tả bệnh của người dùng: "{description}"
        Danh sách các nhãn bệnh nghi ngờ: [{labels}]

        Nhiệm vụ:
        1. Phân tích mô tả và so sánh với từng nhãn bệnh.
        2. Loại bỏ các nhãn bệnh không phù hợp với mô tả. Giải thích lý do loại bỏ rõ ràng.
        3. Giữ lại các nhãn phù hợp nhất, sắp xếp theo mức độ phù hợp giảm dần.

        Kết quả đầu ra phải ở định dạng JSON:
        {{
            "loai_bo": [{{"label": "nhãn không phù hợp", "ly_do": "..."}}],
            "giu_lai": [{{"label": "nhãn phù hợp", "do_phu_hop": "cao/trung bình/thấp"}}]
        }}
    """)

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content([prompt])
        text = response.text.strip()
        clean_text = re.sub(r"```(?:json)?|```", "", text).strip()
        result = json.loads(clean_text)
        return result 

    except Exception as e:
        print(f"Lỗi khi tạo mô tả với Gemini: {e}")
        return None
def mainclient():
    download_from_gcs()
    load_faiss_index()
    image_path = "app/static/img_test/cellulitis.webp"
    print("File tồn tại:", os.path.exists(image_path))
    final_labels, result_labels, anomaly_result_labels=process_image(image_path)
    user_description = collect_user_description()
    image_description = generate_description_with_Gemini(image_path)
    print("Mô tả từ ảnh (Gemini):", image_description)
    result_medical_entities = generate_medical_entities(user_description, image_description)
    print(result_medical_entities)
    questions = compare_descriptions_and_labels(result_medical_entities, final_labels)
    print(final_labels)
    if not questions:
        print("Không tạo được câu hỏi phân biệt.")
        return

    print("\n--- Trả lời các câu hỏi để phân biệt bệnh ---")
    user_answers = ask_user_questions(questions)
    
    print("\n--- Mô tả bổ sung từ người dùng ---")
    print(user_answers)
    combined_description = f"{result_medical_entities}\n\n{user_answers}"
    print("\n--- Đang loại trừ nhãn không phù hợp ---")
    result =filter_incorrect_labels_by_user_description(combined_description, final_labels)
    if not result:
        print("Không có kết quả từ Gemini.")
        return
    refined_labels = result.get("giu_lai", [])
    if not refined_labels:
        print("Không còn nhãn nào phù hợp. Đề xuất tham khảo bác sĩ.")
    else:
        print("Các nhãn còn lại sau loại trừ:")
        for label_info in refined_labels:
            label = label_info.get("label")
            ket_qua = "-".join(label.split("-")[1:])
            suitability = label_info.get("do_phu_hop")
            print(f"- {ket_qua} (Mức độ phù hợp: {suitability})")
if __name__ == "__main__":
    mainclient()