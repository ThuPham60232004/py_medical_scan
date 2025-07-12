from io import BytesIO
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
from typing import Optional
import logging
import textwrap
import re
import requests
import xml.etree.ElementTree as ET
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

GCS_BUCKET = "storage3000image"
GCS_IMAGE_PATH = "uploaded_images/"
GCS_KEY_PATH = storage.Client.from_service_account_json("app/gsc-key.json")

VECTOR_FILE = "static/processed/embedded_vectors.json"
GCS_FOLDER = "handle_data"
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
        (GCS_ANOMALY_INDEX_PATH, LOCAL_ANOMALY_INDEX_PATH),
        (GCS_ANOMALY_LABELS_PATH, LOCAL_ANOMALY_LABELS_PATH),
        (GCS_DATASET_PATH, LOCAL_DATASET_PATH),
    ]
    for gcs_path, local_path in files_to_download:
        blob = bucket.blob(gcs_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"Tải về {gcs_path} to {local_path}")

def preprocess_image(image_path: str) -> Optional[np.ndarray]:
    """Tiền xử lý ảnh bằng Gaussian Blur và Canny Edge Detection."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh từ {image_path}")
        return None
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    edges = cv2.Canny(equalized, 50, 150)
    return edges

def embed_image(image_path: str) -> Optional[np.ndarray]:
    """Nhúng ảnh thành vector sử dụng mô hình CLIP."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh từ {image_path}")
        return None
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.cpu().numpy().astype(np.float32)

def generate_anomaly_map(image_data: bytes) -> Optional[np.ndarray]:
    """Sinh anomaly map từ ảnh đầu vào bằng ViT feature extractor."""
    try:
        img = Image.open(BytesIO(image_data)).convert("RGB")
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

def embed_anomaly_map(anomaly_map: np.ndarray) -> Optional[np.ndarray]:
    """Nhúng anomaly map thành vector sử dụng mô hình CLIP."""
    if anomaly_map is None:
        print("Anomaly map is None.")
        return None
    if len(anomaly_map.shape) == 2:  # Grayscale image
        anomaly_map_rgb = cv2.cvtColor(anomaly_map, cv2.COLOR_GRAY2RGB)
    else:
        anomaly_map_rgb = anomaly_map
    inputs = processor(images=anomaly_map_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.cpu().numpy().astype(np.float32)

def extract_disease_named(results: list) -> list:
    """Trích xuất tên bệnh từ danh sách kết quả."""
    simplified = []
    for item in results:
        try:
            if isinstance(item, str):
                label = item.split('/')[-1] if '/' in item else item
                simplified.append({
                    'label': label,
                    'cosine_similarity': 0.0
                })
            elif isinstance(item, dict) and 'label' in item:
                label = item['label'].split('/')[-1] if '/' in item['label'] else item['label']
                cosine_similarity = item.get('cosine_similarity', 0.0)
                simplified.append({
                    'label': label,
                    'cosine_similarity': float(cosine_similarity)
                })
            else:
                print(f"Lỗi khi xử lý item {item}: Định dạng không hợp lệ")
        except Exception as e:
            print(f"Lỗi khi xử lý item {item}: {e}")
    return simplified

def load_faiss_index():
    """Tải FAISS Index và nhãn bệnh từ file."""
    global index, labels, anomaly_index, anomaly_labels
    if os.path.exists(LOCAL_INDEX_PATH):
        try:
            index = faiss.read_index(LOCAL_INDEX_PATH)
            print(f"FAISS Index tải thành công! Tổng số vector: {index.ntotal}")
        except Exception as e:
            print(f"Lỗi tải FAISS Index: {e}")
            index = None
    else:
        print("FAISS Index không tồn tại!")

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
        anomaly_labels = np.load(LOCAL_ANOMALY_LABELS_PATH, allow_pickle=True).item()
        print(f"Đã tải {len(anomaly_labels)} nhãn bệnh từ labels-anomaly.npy")
    else:
        print("labels-anomaly.npy không tồn tại!")
    
    if os.path.exists(LOCAL_LABELS_PATH):
        labels = np.load(LOCAL_LABELS_PATH, allow_pickle=True).item()
        print(f"Đã tải {len(labels)} nhãn bệnh từ labels.npy")
    else:
        print("labels.npy không tồn tại!")

def search_similar_images(query_vector, top_k=5):
    """Tìm ảnh tương tự bằng FAISS Index."""
    if index is None or index.ntotal == 0:
        print("FAISS index trống!")
        return []
    if index.ntotal != len(labels):
        print(f"Số vector ({index.ntotal}) không khớp với nhãn ({len(labels)}). Kết quả có thể không chính xác!")
    try:
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        faiss.normalize_L2(query_vector)
        distances, indices = index.search(query_vector, top_k)
        print(f"Chỉ số tìm thấy: {indices}")
        print(f"Cosine similarities: {distances}")
        similar_results = []
        labels_keys = list(labels.keys()) if labels else []
        for idx, sim in zip(indices[0], distances[0]):
            print(f"Xử lý idx: {idx}, similarity: {sim}")
            if sim < 55:  # Ngưỡng lọc
                continue
            if 0 <= idx < len(labels_keys):
                label_filename = labels_keys[idx]
                label = labels.get(label_filename, "unknown")
                similar_results.append({
                    "label": label,
                    "cosine_similarity": float(sim)
                })
            else:
                print(f"Index {idx} vượt phạm vi labels ({len(labels_keys)})!")
        return similar_results
    except Exception as e:
        print(f"Lỗi tìm kiếm ảnh tương tự: {e}")
        return []

def search_anomaly_images(query_vector, top_k=5):
    """Tìm ảnh anomaly map tương tự bằng FAISS Index."""
    if anomaly_index is None or anomaly_index.ntotal == 0:
        print("FAISS Anomaly Index trống!")
        return []
    if anomaly_index.ntotal != len(anomaly_labels):
        print(f"Số vector ({anomaly_index.ntotal}) không khớp với nhãn bất thường ({len(anomaly_labels)}). Kết quả có thể không chính xác!")
    try:
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        faiss.normalize_L2(query_vector)
        distances, indices = anomaly_index.search(query_vector, top_k)
        print(f"Chỉ số tìm thấy (anomaly): {indices}")
        print(f"Cosine similarities (anomaly): {distances}")
        similar_results = []
        anomaly_labels_keys = list(anomaly_labels.keys()) if anomaly_labels else []
        for idx, sim in zip(indices[0], distances[0]):
            print(f"Xử lý idx (anomaly): {idx}, similarity: {sim}")
            if sim < 55:  # Ngưỡng lọc
                continue
            if 0 <= idx < len(anomaly_labels_keys):
                label_filename = anomaly_labels_keys[idx]
                label = anomaly_labels.get(label_filename, "unknown")
                similar_results.append({
                    "label": label,
                    "cosine_similarity": float(sim)
                })
            else:
                print(f"Index {idx} vượt phạm vi labels_anomaly ({len(anomaly_labels_keys)})!")
        return similar_results
    except Exception as e:
        print(f"Lỗi tìm kiếm ảnh anomaly: {e}")
        return []

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

def generate_description_with_Gemini(image_path: str) -> Optional[str]:
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

def combine_labels(anomaly_labels: list, normal_labels: list) -> list:
    """Gộp danh sách nhãn từ ảnh gốc và ảnh anomaly."""
    combined = []
    for label in anomaly_labels + normal_labels:
        if isinstance(label, dict):
            combined.append(label['label'])
        else:
            combined.append(label)
    return combined

def generate_medical_entities(user_description: str, image_description: str) -> Optional[str]:
    combined_description = f"1. Mô tả từ người dùng: {user_description}. 2. Mô tả từ ảnh: {image_description}."
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

def compare_descriptions_and_labels(description: str, labels: list) -> list:
    labels_str = ", ".join(labels)
    prompt = textwrap.dedent(f"""
        Mô tả: "{description}"
        Nhãn: "{labels_str}"
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

def ask_user_questions(questions: list, disease_name: str) -> str:
    answers = []
    for idx, q in enumerate(questions, 1):
        print(f"Câu {idx}: {q}")
        answer = answer_question(q, disease_name)
        answers.append(f"Câu hỏi: {q}\nTrả lời: {answer}")
    return "\n\n".join(answers)

def answer_question(question: str, disease_name: str) -> str:
    prompt = f"""
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
        result = re.sub(r"\n+", " ", result).strip()
        if not result:
            return "Không có thông tin"
        return result
    except Exception as e:
        print(f"Lỗi khi tổng hợp thông tin: {e}")
        return "Xảy ra lỗi trong quá trình tổng hợp thông tin"

def clean_image_name(image_name: str) -> str:
    name = os.path.splitext(image_name)[0]
    name = re.sub(r"\(\d+\)", "", name)
    return name.strip().lower()

def process_image(image_data: bytes) -> tuple:
    if not image_data:
        print("Không có dữ liệu ảnh để xử lý.")
        return None, [], [], [], []
    
    print("Đã nhận dữ liệu ảnh, bắt đầu xử lý...")
    
    # Convert bytes to image for embedding
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        print("Lỗi giải mã ảnh, dừng quy trình.")
        return None, [], [], [], []

    # Save image temporarily for embed_image
    processed_dir = Path("app/static/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    temp_image_path = processed_dir / "temp_image.jpg"
    cv2.imwrite(str(temp_image_path), img)

    embedding = embed_image(str(temp_image_path))
    result_labels_simple = []
    detailed_labels_normal = []
    if embedding is not None:
        detailed_labels_normal = search_similar_images(embedding)
        print("detailed_labels_normal", detailed_labels_normal)
        detailed_labels_normal = extract_disease_named(detailed_labels_normal)
        print("detailed_labels_normal sau khi cắt", detailed_labels_normal)
        result_labels_simple = [item["label"] for item in detailed_labels_normal]
        print("result_labels_simple", result_labels_simple)
        print("🔍 Ảnh gốc:")
        for item in detailed_labels_normal:
            print(f"- {item['label']} (cosine: {item['cosine_similarity']:.4f})")

    anomaly_map = generate_anomaly_map(image_data)
    print("Anomaly map generated:", anomaly_map is not None)
    anomaly_result_labels_simple = []
    detailed_labels_anomaly = []
    if anomaly_map is not None:
        anomaly_map_path = processed_dir / "anomaly_map_temp.jpg"
        cv2.imwrite(str(anomaly_map_path), anomaly_map)
        anomaly_map_embedding = embed_anomaly_map(anomaly_map)
        if anomaly_map_embedding is not None:
            detailed_labels_anomaly = search_anomaly_images(anomaly_map_embedding)
            print("detailed_labels_anomaly", detailed_labels_anomaly)
            detailed_labels_anomaly = extract_disease_named(detailed_labels_anomaly)
            print("detailed_labels_anomaly sau khi cắt", detailed_labels_anomaly)
            anomaly_result_labels_simple = [item["label"] for item in detailed_labels_anomaly]
            print("🔍 Anomaly Map:")
            for item in detailed_labels_anomaly:
                print(f"- {item['label']} (cosine: {item['cosine_similarity']:.4f})")

    final_labels = combine_labels(detailed_labels_anomaly, detailed_labels_normal)
    print("final_labels", final_labels)

    # Cleanup temporary files
    for temp_file in [temp_image_path, anomaly_map_path]:
        try:
            if temp_file and temp_file.exists():
                temp_file.unlink()
        except Exception as e:
            print(f"Lỗi khi xóa file tạm {temp_file}: {e}")
    
    return final_labels, result_labels_simple, anomaly_result_labels_simple, detailed_labels_normal, detailed_labels_anomaly

def filter_incorrect_labels_by_user_description(description: str, labels: list) -> dict:
    labels_str = ", ".join(labels)
    prompt = textwrap.dedent(f"""
        Mô tả bệnh của người dùng: "{description}"
        Danh sách các nhãn bệnh nghi ngờ: [{labels_str}]

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
        return {"loai_bo": [], "giu_lai": []}

def upload_json_to_gcs(bucket_name: str, destination_blob_name: str, source_file_name: str):
    client = storage.Client.from_service_account_json("app/gsc-key.json")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to {destination_blob_name}.")

def append_disease_to_json(file_path: str, new_disease: dict):
    with open(file_path, 'r+', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
        data.append(new_disease)
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()
    print(f"Added new disease: {new_disease.get('name', '')}")

def search_disease_in_json(file_path: str, disease_name: str) -> list:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("File JSON không phải là danh sách.")
        return []
    results = [
        entry for entry in data
        if isinstance(entry, dict) and disease_name.lower() in entry.get("Tên bệnh", "").lower()
    ]
    return results

def generate_keyword(keyword: str) -> str:
    prompt = f"""
    Bạn là một người có kiến thức sâu rộng về y khoa dựa vào keyword được truyền vào.
    Keyword là tên bệnh tôi cần bạn tạo ra danh sách tên bệnh liên quan đến keyword đó.
    Ví dụ tên bệnh là: Squamouscell thì bạn có thể liệt kê các keyword liên quan đến tên như: Squamouse, Squamouse cell, Squamouse Cancer,..
    Tối đa là 10 từ khóa liên quan đến tên bệnh đó.
    Trả về dưới dạng json với cấu trúc như sau:
    {{
        "keyword": [
            "keyword1",
            "keyword2",
            "keyword3",
            "keyword4",
            "keyword5",
            "keyword6",
            "keyword7",
            "keyword8",
            "keyword9",
            "keyword10"
        ]
    }}
    """
    try:
        if not keyword:
            return "Không có từ khóa truyền vào"
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        result = response.text.strip()
        result = re.sub(r"^(?:\w+)?\n|\n$", "", result).strip()
        if not result:
            return "Không có thông tin"
        return result
    except Exception as e:
        print(f"Lỗi khi tạo từ khóa: {e}")
        return "Xảy ra lỗi trong quá trình tạo từ khóa"

def search_medlineplus(ten_khoa_hoc: str) -> str:
    if not ten_khoa_hoc or ten_khoa_hoc == "Không tìm thấy":
        return "Không có tên bệnh truyền vào"
    print(f"Tìm kiếm thông tin bệnh '{ten_khoa_hoc}' trên MedLinePlus...")
    keyword = generate_keyword(ten_khoa_hoc)
    try:
        keyword_dict = json.loads(keyword)
    except json.JSONDecodeError:
        keyword_dict = {"keyword": [ten_khoa_hoc]}
    
    url = 'https://wsearch.nlm.nih.gov/ws/query'
    params = {
        'db': 'healthTopics',
        'term': f'{ten_khoa_hoc}'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        print("Tìm kiếm thành công!")
        cleaned_content = clean_xml_content(response.content)
        return cleaned_content
    
    for item in keyword_dict.get("keyword", []):
        print(f"Tìm kiếm thông tin bệnh '{item}' trên MedLinePlus...")
        params = {
            'db': 'healthTopics',
            'term': f'{item}'
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            print("Tìm kiếm thành công!")
            cleaned_content = clean_xml_content(response.content)
            return cleaned_content
    return ""

def clean_xml_content(xml_content: str) -> str:
    """Nhận vào một chuỗi XML, trích xuất và kết hợp nội dung văn bản từ tất cả các phần tử."""
    try:
        root = ET.fromstring(xml_content)
        text_nodes = [elem.text.strip() for elem in root.iter() if elem.text and elem.text.strip()]
        return ' '.join(text_nodes)
    except ET.ParseError as e:
        print(f"Lỗi phân tích XML: {e}")
        return ""

def decide_final_label(label_string: list) -> str:
    counter = Counter(label_string)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else "unknown"

def extract_medical_info(text: str) -> dict:
    prompt = f"""
    Dịch văn bản về tiếng việt
    Bạn là một chuyên gia y tế, bạn có khả năng trích xuất thông tin y khoa từ văn bản.
    Hãy trích xuất thông tin y khoa từ văn bản dưới dạng JSON hợp lệ **không chứa Markdown**.
    Chỉ lấy kết quả đầu tiên mà bạn tìm được
    Hãy trích xuất thật chi tiết
    Văn bản đầu vào là:
    {text}
    {{
        "Tên bệnh":"", Tên bệnh là tên của bệnh được ghi ở đầu tiêu đề
        "Tên khoa học": "", Tên khoa học thường được ghi bằng tiếng anh hoặc tiếng latinh, nếu không thấy thì tên khoa học sẽ bằng tên bệnh nhưng chỉ lấy phần tiếng anh
        "Triệu chứng": "", Triệu chứng là những dấu hiệu mà bệnh nhân có thể gặp phải khi mắc bệnh này, nếu không tìm thấy thì có thể đổi từ tên bệnh sang tiếng anh
        "Vị trí xuất hiện": "", Vị trí xuất hiện là nơi mà bệnh này có thể xảy ra trên cơ thể người,
        "Nguyên nhân": "",  Nguyên nhân là lý do mà bệnh này xảy ra
        "Tiêu chí chẩn đoán": "",  Tiêu chí chẩn đoán là những tiêu chí mà bác sĩ có thể sử dụng để xác định bệnh này
        "Chẩn đoán phân biệt": "",  Chẩn đoán phân biệt là những bệnh khác mà bác sĩ có thể xem xét khi xác định bệnh này
        "Điều trị": "",  Điều trị là những phương pháp mà bác sĩ có thể sử dụng để điều trị bệnh này
        "Phòng bệnh": "" , Phòng bệnh là những biện pháp mà bác sĩ có thể khuyên bệnh nhân thực hiện để ngăn ngừa bệnh này
        "Các loại thuốc":
        [{{
            "Tên thuốc": "", 
            "Liều lượng": "", 
            "Thời gian sử dụng": ""
        }}]
    }}
    - Nếu không có thông tin, đặt giá trị "Không tìm thấy".
    - Không thêm giải thích, không in Markdown, không thêm ký tự thừa.
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        raw_text = response.text if hasattr(response, "text") else response.parts[0].text
        raw_text = re.sub(r"^```json\n|\n```$", "", raw_text)
        extracted_info = json.loads(raw_text)
        completed_info = clean_text_json(extracted_info)
        return completed_info
    except json.JSONDecodeError:
        print("Lỗi: Không thể parse JSON từ Gemini.")
        return {}
    except Exception as e:
        print(f"Lỗi trích xuất thông tin y khoa: {e}")
        return {}

def clean_text(text: str) -> str:
    """Làm sạch một chuỗi văn bản."""
    if not isinstance(text, str):
        return text
    text = re.sub(r'[\\\n\r\t\*\]]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_text_json(data):
    """Làm sạch toàn bộ văn bản trong một cấu trúc JSON."""
    if isinstance(data, dict):
        return {key: clean_text_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_text_json(item) for item in data]
    else:
        return clean_text(data)

def translate_disease_name(disease_name: str) -> str:
    prompt = f"""
    Bạn là một chuyên gia y tế có hiểu biết sâu rộng về y khoa.
    Bạn có khả năng dịch tên bệnh từ tiếng anh sang tiếng việt.
    Tên bệnh được truyền vào là: {disease_name}
    Hãy dịch tên bệnh đó sang tiếng việt.
    Trả về tên bệnh đó
    """
    try:
        if not disease_name:
            return "Không có tên bệnh truyền vào"
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        result = response.text.strip()
        result = re.sub(r"^(?:\w+)?\n|\n$", "", result).strip()
        if not result:
            return "Chúng tôi không thể dịch tên bệnh này"
        return result
    except Exception as e:
        print(f"Lỗi khi dịch tên bệnh: {e}")
        return "Xảy ra lỗi trong quá trình dịch tên bệnh"

def search_final(name: str):
    translate_name = translate_disease_name(name)
    print(f"Tên bệnh đã dịch: {translate_name}")
    search_json_result = search_disease_in_json(LOCAL_DATASET_PATH, translate_name)
    if search_json_result:
        print(f"Kết quả tìm kiếm trong file JSON: {search_json_result}")
    else:
        print(f"Không tìm thấy tên bệnh '{translate_name}' trong file JSON.")
        print("Bắt đầu tìm kiếm bằng MedlinePlus...")
        search_medline_result = search_medlineplus(name)
        print(f"Kết quả tìm kiếm MedlinePlus: {search_medline_result}")
        print("Bắt đầu trích xuất thông tin y khoa từ MedlinePlus...")
        extract_medical_info_result = extract_medical_info(search_medline_result)
        if extract_medical_info_result:
            print(f"Kết quả trích xuất thông tin y khoa: {extract_medical_info_result}")
            print("Đang thêm thông tin vào file JSON...")
            append_disease_to_json(LOCAL_DATASET_PATH, extract_medical_info_result)
            print("Upload file JSON lên GCS...")
            upload_json_to_gcs(GCS_BUCKET, GCS_DATASET_PATH, LOCAL_DATASET_PATH)

def generate_description(disease_name: str) -> Optional[str]:
    """Sinh mô tả triệu chứng của bệnh nhân dựa trên tên bệnh với Gemini."""
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

def process_pipeline(image_path: str, disease_name: str) -> tuple:
    """Xử lý ảnh và sinh kết quả chẩn đoán."""
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
    except FileNotFoundError:
        print(f"Lỗi: File ảnh {image_path} không tồn tại.")
        return None, None, None, None
    
    final_labels, result_labels, anomaly_result_labels, detailed_labels_normal, detailed_labels_anomaly = process_image(image_data)
    if final_labels is None:
        print(f"Lỗi xử lý ảnh {image_path}, bỏ qua...")
        return None, None, None, None
    
    user_description = generate_description(disease_name)
    print("Mô tả từ người dùng:", user_description)
    if not user_description:
        print("\nĐang chọn nhãn dựa trên hình ảnh và mô hình...")
        final_diagnosis = decide_final_label(final_labels)
        print(f"\nKết quả sơ bộ: {final_diagnosis}")
        print("Thông báo: Vì bạn không cung cấp mô tả, kết quả có thể chưa chính xác.")
        return final_diagnosis, user_description, [], []
    
    image_description = generate_description_with_Gemini(image_path)
    print("Mô tả từ ảnh (Gemini):", image_description)
    result_medical_entities = generate_medical_entities(user_description, image_description)
    print("Medical entities:", result_medical_entities)
    
    questions = compare_descriptions_and_labels(result_medical_entities, final_labels)
    print("Generated questions:", questions)
    if not questions:
        print("Không tạo được câu hỏi phân biệt.")
        return final_labels, user_description, [], []
    
    print("\n--- Trả lời các câu hỏi để phân biệt bệnh ---")
    user_answers = ask_user_questions(questions, disease_name)
    
    print("\n--- Mô tả bổ sung từ người dùng ---")
    print(user_answers)
    combined_description = f"{result_medical_entities}\n\n{user_answers}"
    print("\n--- Đang loại trừ nhãn không phù hợp ---")
    result = filter_incorrect_labels_by_user_description(combined_description, final_labels)
    if not result:
        print("Không có kết quả từ Gemini.")
        return final_labels, user_description, questions, user_answers
    
    refined_labels = result.get("giu_lai", [])
    if not refined_labels:
        print("Không còn nhãn nào phù hợp. Đề xuất tham khảo bác sĩ.")
    else:
        print("Các nhãn còn lại sau loại trừ:")
        for label_info in refined_labels:
            label = label_info.get("label")
            ket_qua = "-".join(label.split("-")[1:]) if "-" in label else label
            suitability = label_info.get("do_phu_hop")
            print(f"- {ket_qua} (Mức độ phù hợp: {suitability})")
            search_final(ket_qua)
    
    return refined_labels, user_description, questions, user_answers

def get_all_images(directory: str) -> list:
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    return [p for p in Path(directory).rglob("*") if p.is_file() and p.suffix.lower() in exts]

def test_process_pipeline(choice: str) -> list:
    image_dir = "app/static/data_test"
    file_path = "app/static/test_result.json"

    # Lấy danh sách tất cả ảnh và sắp xếp để đảm bảo thứ tự nhất quán
    all_images = sorted(get_all_images(image_dir))
    if not all_images:
        print("[!] Không tìm thấy ảnh trong thư mục.")
        return []

    # Hiển thị danh sách ảnh để người dùng chọn
    print("\nDanh sách ảnh có sẵn:")
    for idx, img_path in enumerate(all_images, 1):
        print(f"{idx}. {os.path.basename(img_path)}")

    # Yêu cầu người dùng chọn ảnh bằng số thứ tự
    while True:
        try:
            if choice.lower() == 'q':
                print("Đã thoát chương trình.")
                return []
            choice = int(choice)
            if 1 <= choice <= len(all_images):
                break
            else:
                print(f"Vui lòng nhập số từ 1 đến {len(all_images)}.")
        except ValueError:
            print("Vui lòng nhập một số hợp lệ hoặc 'q' để thoát.")

    # Lấy ảnh được chọn
    image_path = all_images[choice - 1]
    image_name = os.path.basename(image_path)
    image_name_cleaned = clean_image_name(image_name)

    # Xử lý ảnh được chọn
    print(f"\n=== Đang xử lý ảnh: {image_path} ===")
    result, user_description, questions, user_answers = process_pipeline(str(image_path), image_name_cleaned)

    print(f"Dự đoán: {result}")
    print(f"Thực tế: {image_name_cleaned}")

    # Kiểm tra đúng/sai
    is_correct = False
    if result:
        if isinstance(result, str):
            is_correct = image_name_cleaned.lower() == result.lower()
        elif isinstance(result, list):
            is_correct = any(image_name_cleaned.lower() == label_info.get("label", "").lower() for label_info in result)
    ket_qua = "Đúng" if is_correct else "Sai"

    print(f"Kết quả: {ket_qua}")
    print(f"Đường dẫn ảnh: {image_path}")

    # Đọc file JSON hiện có (nếu tồn tại) để lấy danh sách kết quả cũ
    existing_results = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                existing_results = json.load(file)
                if not isinstance(existing_results, list):
                    print("File JSON không đúng định dạng, khởi tạo danh sách mới.")
                    existing_results = []
        except json.JSONDecodeError:
            print("Lỗi đọc file JSON, khởi tạo danh sách mới.")
            existing_results = []

    # Tạo kết quả mới
    new_result = {
        "STT": len(existing_results) + 1,  # Tăng STT dựa trên số lượng kết quả hiện có
        "Ảnh": image_name,
        "Dự đoán": result,
        "Thực tế": image_name_cleaned,
        "Kết quả": ket_qua,
        "Đường dẫn": str(image_path),
        "Mô tả người dùng": user_description,
        "Câu hỏi phân biệt": questions,
        "Trả lời câu hỏi": user_answers
    }

    # Thêm kết quả mới vào danh sách
    existing_results.append(new_result)

    # Ghi lại toàn bộ danh sách vào file JSON
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(existing_results, file, ensure_ascii=False, indent=4)
        print(f"\nHoàn tất. Kết quả mới đã được nối thêm vào {file_path}.")
    except Exception as e:
        print(f"Lỗi khi ghi vào file {file_path}: {e}")

    return existing_results

def mainclient():
    download_from_gcs()
    load_faiss_index()
    for i in range(91,92):
        print(f"\n=== Bắt đầu kiểm tra ảnh thứ {i} ===")
        results = test_process_pipeline(str(i))
        if not results:
            print("Không có kết quả nào được tạo ra.")
        else:
            print(f"Đã xử lý {len(results)} ảnh, kết quả đã được lưu vào file JSON.")

if __name__ == "__main__":
    mainclient()