from io import BytesIO
import cv2
import os
import numpy as np
import time
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
import re
import requests
import xml.etree.ElementTree as ET
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# ---------------------- CẤU HÌNH ----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------------- CONSTANTS ----------------------
PROCESSED_DIR = "app/static/processed"
ANOMALY_MAP_DIR = "app/static/anomaly_maps"
ROI_OUTPUT_DIR = "app/static/roi_outputs"
GCS_BUCKET = "group_dataset-nt"
GCS_FOLDER = "handle_data"
LOCAL_SAVE_DIR = "app/static/"
JSON_RESULT_PATH = "app/static/test_result.json"

# Ensure directories exist
for directory in [PROCESSED_DIR, ANOMALY_MAP_DIR, ROI_OUTPUT_DIR, LOCAL_SAVE_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)

REQUIRED_FILES = [
    "faiss_index.bin",
    "faiss_index_anomaly.bin",
    "faiss_index_bacterial_infections.bin",
    "faiss_index_fungal_infections.bin",
    "faiss_index_parasitic_infections.bin",
    "faiss_index_virus.bin",
    "faiss_index_anomaly_bacterial_infections.bin",
    "faiss_index_anomaly_fungal_infections.bin",
    "faiss_index_anomaly_parasitic_infections.bin",
    "faiss_index_anomaly_virus.bin",
    "labels.npy",
    "labels_anomaly.npy",
    "labels_bacterial_infections.npy",
    "labels_fungal_infections.npy",
    "labels_parasitic_infections.npy",
    "labels_virus.npy",
    "labels_anomaly_bacterial_infections.npy",
    "labels_anomaly_fungal_infections.npy",
    "labels_anomaly_parasitic_infections.npy",
    "labels_anomaly_virus.npy"
]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
vit_model = timm.create_model("vit_base_patch16_224", pretrained=True).to(device)
vit_model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---------------------- GOOGLE CLOUD CLIENT (DÙNG JSON) ----------------------
def get_gcs_client():
    try:
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path or not os.path.exists(credentials_path):
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set or file does not exist")
        return storage.Client.from_service_account_json(credentials_path)
    except Exception as e:
        logging.error(f"Lỗi tạo Google Cloud Client: {e}")
        raise

def download_gcs_file(bucket_name: str, source_blob_name: str, destination_file_name: str, retries: int = 5):
    for attempt in range(1, retries + 1):
        try:
            storage_client = get_gcs_client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(source_blob_name)
            blob.download_to_filename(destination_file_name, timeout=120)
            logging.info(f"Tải thành công {source_blob_name} → {destination_file_name}")
            return
        except Exception as e:
            logging.warning(f"Lỗi tải {source_blob_name} (lần {attempt}/{retries}): {e}")
            if attempt < retries:
                wait_time = 5 * attempt
                logging.info(f"Đợi {wait_time}s trước lần thử lại...")
                time.sleep(wait_time)
            else:
                logging.error(f"Thất bại sau {retries} lần: {source_blob_name}")

def download_all_required_files():
    Path(LOCAL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    for file in REQUIRED_FILES:
        gcs_path = f"{GCS_FOLDER}/{file}"
        local_path = os.path.join(LOCAL_SAVE_DIR, file)
        if not os.path.exists(local_path):
            download_gcs_file(GCS_BUCKET, gcs_path, local_path)
        else:
            logging.info(f"File {local_path} đã tồn tại, bỏ qua tải xuống.")

# ---------------------- BƯỚC 1: NHẬN ẢNH ----------------------
def receive_image_from_path(image_path: str) -> Image.Image:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh tại: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

# ---------------------- BƯỚC 2: TIỀN XỬ LÝ ẢNH ----------------------
def apply_clahe(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v)
    hsv_clahe = cv2.merge((h, s, v_clahe))
    return cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2RGB)

def preprocess_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = apply_clahe(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return Image.fromarray(image), image

# ---------------------- BƯỚC 3: SINH MÔ TẢ BẰNG GEMINI ----------------------
def generate_description_with_Gemini(image_path: str) -> Optional[str]:
    try:
        img = Image.open(image_path)
        model = genai.GenerativeModel('gemini-2.5-flash')
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
        Xóa markdown và các ký tự đặc biệt trong kết quả.
        """
        response = model.generate_content([prompt, img])
        caption = response.text.replace("\n", " ").strip()
        return caption
    except Exception as e:
        logging.error(f"Lỗi khi tạo caption với Gemini: {e}")
        return None

def generate_anomaly_overlay(image_pil):
    image_resized = image_pil.resize((224, 224))
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    image_np = (image_np - 0.5) / 0.5
    image_tensor = torch.tensor(image_np.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        features = vit_model.forward_features(image_tensor)

    anomaly_map = features[0].norm(dim=0).cpu().numpy()
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    anomaly_map = (anomaly_map * 255).astype(np.uint8)

    anomaly_map_resized = cv2.resize(anomaly_map, image_pil.size[::-1])
    anomaly_map_blur = cv2.GaussianBlur(anomaly_map_resized, (5, 5), 0)
    heatmap = cv2.applyColorMap(anomaly_map_blur, cv2.COLORMAP_JET)

    image_cv = np.array(image_pil)
    if heatmap.shape[:2] != image_cv.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_cv.shape[1], image_cv.shape[0]))

    if image_cv.shape[2] == 3:
        original = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    else:
        original = image_cv

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    return overlay, anomaly_map_blur

# ---------------------- PHÂN LOẠI ẢNH THƯỜNG (FULL IMAGE) ----------------------
def embed_image_clip(image_pil: Image.Image) -> np.ndarray:
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    image_embedding = outputs.cpu().numpy().astype("float32")
    return image_embedding

def search_faiss_index(embedding: np.ndarray, index_path: str, label_path: str, top_k: int = 5):
    if not os.path.exists(index_path) or not os.path.exists(label_path):
        raise FileNotFoundError(f"Không tìm thấy file index hoặc label: {index_path}, {label_path}")
    index = faiss.read_index(index_path)
    labels = np.load(label_path, allow_pickle=True)
    distances, indices = index.search(embedding, top_k)
    top_labels = [labels[idx] for idx in indices[0]]
    return list(zip(top_labels, distances[0]))

# ---------------------- PHÂN TÍCH BẤT THƯỜNG (ANOMALY PIPELINE) ----------------------
def save_anomaly_outputs(anomaly_overlay, anomaly_map, image_path: str):
    basename = Path(image_path).stem
    roi_output_path = os.path.join(ROI_OUTPUT_DIR, f"{basename}_overlay.jpg")
    anomaly_map_path = os.path.join(ANOMALY_MAP_DIR, f"{basename}_anomaly.jpg")

    os.makedirs(ROI_OUTPUT_DIR, exist_ok=True)
    os.makedirs(ANOMALY_MAP_DIR, exist_ok=True)

    cv2.imwrite(roi_output_path, anomaly_overlay)
    cv2.imwrite(anomaly_map_path, anomaly_map)

    return roi_output_path, anomaly_map_path

def embed_anomaly_heatmap(heatmap_path: str) -> np.ndarray:
    image = Image.open(heatmap_path).convert("RGB")
    return embed_image_clip(image)

def search_faiss_anomaly_index(embedding: np.ndarray, index_path: str, label_path: str, top_k: int = 5):
    return search_faiss_index(embedding, index_path, label_path, top_k)

def normalize_group_name(group_name: str) -> str:
    group_name = group_name.lower()
    group_name = group_name.strip()
    group_name = re.sub(r"\s+", "_", group_name)
    group_name = re.sub(r"[^a-z0-9_]", "", group_name)
    return group_name

def aggregate_combined_results(combined_results):
    score_dict = {}
    for label, distance in combined_results:
        sim = 1 / (1 + distance)
        score_dict[label] = score_dict.get(label, 0) + sim

    total_score = sum(score_dict.values()) or 1  # Avoid division by zero
    normalized_scores = {label: (score / total_score) * 100 for label, score in score_dict.items()}
    sorted_scores = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores

def generate_discriminative_questions(caption: str, labels: list[str], model=None) -> list[str]:
    if model is None:
        model = genai.GenerativeModel("gemini-2.5-pro")

    prompt = f"""
Bạn là bác sĩ da liễu. Tôi có ảnh da liễu với mô tả sau:

--- MÔ TẢ ẢNH ---
{caption}

Tôi đang phân vân giữa các bệnh sau: {', '.join(labels)}.

Hãy đưa ra 3 câu hỏi phân biệt giúp xác định đúng bệnh.
- Câu hỏi cần dễ hiểu, trực tiếp, liên quan đến triệu chứng đặc trưng.
- **Xưng hô trung lập, dùng “bạn” để hỏi.**
- Không được dùng từ như "bé", "em bé", "bệnh nhân", "anh/chị", "người bệnh", v.v.

Chỉ cần liệt kê 3 câu hỏi bằng tiếng Việt, không giải thích thêm.
"""
    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        questions = [q.strip(" -0123456789.") for q in raw_text.split("\n") if q.strip()]
        return questions[:3]
    except Exception as e:
        logging.error(f"Lỗi sinh câu hỏi phân biệt: {e}")
        return []

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

def detailed_group_analysis(image_vector: np.ndarray, anomaly_vector: np.ndarray, group_name: str, top_k: int = 5):
    print(f"\nSo sánh trong nhóm bệnh: {group_name}")
    index_path_full = os.path.join(LOCAL_SAVE_DIR, f"faiss_index_{group_name}.bin")
    label_path_full = os.path.join(LOCAL_SAVE_DIR, f"labels_{group_name}.npy")

    index_path_anomaly = os.path.join(LOCAL_SAVE_DIR, f"faiss_index_anomaly_{group_name}.bin")
    label_path_anomaly = os.path.join(LOCAL_SAVE_DIR, f"labels_anomaly_{group_name}.npy")

    print(f"\nKết quả so khớp ảnh thường với {group_name}:")
    try:
        full_group_results = search_faiss_index(
            embedding=image_vector,
            index_path=index_path_full,
            label_path=label_path_full,
            top_k=top_k
        )
        for label, score in full_group_results:
            similarity = 1 / (1 + score)
            print(f"  → {label} (similarity: {similarity*100:.2f}%)")
    except Exception as e:
        logging.error(f"Lỗi tìm kiếm FAISS cho ảnh thường ({group_name}): {e}")
        full_group_results = []

    print(f"\nKết quả so khớp anomaly heatmap với {group_name}:")
    try:
        anomaly_group_results = search_faiss_index(
            embedding=anomaly_vector,
            index_path=index_path_anomaly,
            label_path=label_path_anomaly,
            top_k=top_k
        )
        for label, score in anomaly_group_results:
            similarity = 1 / (1 + score)
            print(f"  → {label} (similarity: {similarity*100:.2f}%)")
    except Exception as e:
        logging.error(f"Lỗi tìm kiếm FAISS cho anomaly ({group_name}): {e}")
        anomaly_group_results = []

    print(f"\nGộp nhãn từ 2 pipeline (ảnh thường + anomaly) trong nhóm '{group_name}':")
    combined_results = full_group_results + anomaly_group_results

    label_scores_raw = {}
    for label, distance in combined_results:
        similarity = 1 / (1 + distance)
        label_scores_raw[label] = label_scores_raw.get(label, 0) + similarity

    total_similarity = sum(label_scores_raw.values()) or 1
    label_scores_percent = {label: (score / total_similarity) * 100 for label, score in label_scores_raw.items()}

    sorted_labels = sorted(label_scores_percent.items(), key=lambda x: x[1], reverse=True)

    print("Tổng điểm similarity trong nhóm bệnh (%):")
    for label, percent in sorted_labels:
        print(f"  → {label}: {percent:.2f}%")

    return sorted_labels

def answer_question(questions: list, disease_name: str) -> list:
    answers = []
    for question in questions:
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
                result = "Không có thông tin"
            answers.append(result)
        except Exception as e:
            logging.error(f"Lỗi khi tổng hợp thông tin cho câu hỏi '{question}': {e}")
            answers.append("Xảy ra lỗi trong quá trình tổng hợp thông tin")
    return answers

def process_pipeline(image_path, disease_name):
    try:
        preprocessed_pil, preprocessed_np = preprocess_image(image_path)
    except Exception as e:
        logging.error(f"Lỗi tiền xử lý ảnh {image_path}: {e}")
        return None, None, None, None

    description = generate_description_with_Gemini(image_path)
    if description:
        print("Mô tả ảnh:", description)
    else:
        print("Không thể tạo mô tả ảnh.")

    print("\nPhân loại ảnh đầy đủ (Full Image):")
    try:
        full_image_vector = embed_image_clip(preprocessed_pil)
        full_results = search_faiss_index(
            embedding=full_image_vector,
            index_path=os.path.join(LOCAL_SAVE_DIR, "faiss_index.bin"),
            label_path=os.path.join(LOCAL_SAVE_DIR, "labels.npy"),
            top_k=5
        )
        for label, score in full_results:
            print(f"  → {label} (score: {score:.4f})")
    except Exception as e:
        logging.error(f"Lỗi phân loại ảnh đầy đủ: {e}")
        full_results = []

    print("\nPhân tích bất thường (Anomaly Detection):")
    try:
        anomaly_overlay, anomaly_map = generate_anomaly_overlay(preprocessed_pil)
        overlay_path, anomaly_map_path = save_anomaly_outputs(anomaly_overlay, anomaly_map, image_path)

        anomaly_vector = embed_anomaly_heatmap(overlay_path)
        anomaly_results = search_faiss_anomaly_index(
            embedding=anomaly_vector,
            index_path=os.path.join(LOCAL_SAVE_DIR, "faiss_index_anomaly.bin"),
            label_path=os.path.join(LOCAL_SAVE_DIR, "labels_anomaly.npy"),
            top_k=5
        )
        for label, score in anomaly_results:
            print(f"  → {label} (score: {score:.4f})")
    except Exception as e:
        logging.error(f"Lỗi phân tích bất thường: {e}")
        anomaly_results = []

    print("\nKết hợp kết quả từ Full Image + Anomaly (voting theo % normalize):")
    combined_results = full_results + anomaly_results

    label_scores_raw = {}
    for label, distance in combined_results:
        similarity = 1 / (1 + distance)
        label_scores_raw[label] = label_scores_raw.get(label, 0) + similarity

    total_similarity = sum(label_scores_raw.values()) or 1
    label_scores_percent = {label: (score / total_similarity) * 100 for label, score in label_scores_raw.items()}

    sorted_labels = sorted(label_scores_percent.items(), key=lambda x: x[1], reverse=True)

    converted_data = [(str(label), float(score)) for label, score in sorted_labels]


    print("Tổng điểm similarity sau khi chuẩn hóa (%):")
    for label, percent in sorted_labels:
        print(f"  → {label}: {percent:.2f}%")

    if not sorted_labels:
        print("Không có nhãn nào được dự đoán.")
        return None, description, [], []

    top_label, top_percent = sorted_labels[0]
    print(f"\nNhãn được chọn (Top-1): {top_label} ({top_percent:.2f}%)")

    group_name_raw = top_label.split("/")[0]
    normalized_group_name = normalize_group_name(group_name_raw)
    print(f"Nhóm bệnh (chuẩn hoá): {normalized_group_name}")

    print("full_results:", full_results)
    print("..."*10)
    print("anomaly_results:", anomaly_results)

    try:
        combined_results = detailed_group_analysis(
            image_vector=full_image_vector,
            anomaly_vector=anomaly_vector,
            group_name=normalized_group_name,
            top_k=5
        )
        print("\nGộp kết quả từ 2 pipeline (Full Image + Anomaly):", combined_results)
        aggregated_results = aggregate_combined_results(combined_results)

        

        print("\nKết quả sau khi gộp nhãn giống nhau:", aggregated_results)
        #Tạo Json lưu danh sách kết quả gộp nhãn
        aggregated_results_combined = {label: percent for label, percent in aggregated_results}
        print("Kết quả gộp nhãn:", aggregated_results_combined)
        for label, percent in aggregated_results:
            print(f"  → {label}: {percent:.2f}%")

    except Exception as e:
        logging.error(f"Lỗi trong detailed_group_analysis: {e}")
        aggregated_results = []

    top_labels = [label for label, _ in aggregated_results[:3]]

    questions = generate_discriminative_questions(description, top_labels)
    if not questions:
        print("Không tạo được câu hỏi.")
        return top_label, description, [], []

    user_answers = answer_question(questions, disease_name)

    final_diagnosis = select_final_diagnosis_with_llm(
        caption=description,
        labels=top_labels,
        questions=questions,
        answers=user_answers
    )

    return final_diagnosis, description, questions, user_answers, top_labels,converted_data

def get_all_images(directory: str) -> list:
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    return [p for p in Path(directory).rglob("*") if p.is_file() and p.suffix.lower() in exts]

def clean_image_name(image_name: str) -> str:
    name = os.path.splitext(image_name)[0]
    name = re.sub(r"\(\d+\)", "", name)
    return name.strip().lower()

def save_to_json(result_data, file_path=JSON_RESULT_PATH):
    """Append a single result to the JSON file incrementally."""
    existing_results = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                existing_results = json.load(file)
                if not isinstance(existing_results, list):
                    logging.warning("File JSON không đúng định dạng, khởi tạo danh sách mới.")
                    existing_results = []
        except json.JSONDecodeError:
            logging.error("Lỗi đọc file JSON, khởi tạo danh sách mới.")
            existing_results = []

    existing_results.append(result_data)

    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(existing_results, file, ensure_ascii=False, indent=4)
        logging.info(f"Kết quả đã được lưu vào {file_path}")
    except Exception as e:
        logging.error(f"Lỗi khi ghi vào file {file_path}: {e}")

def test_process_pipeline():
    image_dir = "app/static/test_data_searching"

    all_images = sorted(get_all_images(image_dir))
    if not all_images:
        print("[!] Không tìm thấy ảnh trong thư mục.")
        return []

    print("\nDanh sách ảnh có sẵn:")
    for idx, img_path in enumerate(all_images, 1):
        print(f"{idx}. {os.path.basename(img_path)}")

    results = []
    for idx, image_path in enumerate(all_images, 1):
        image_name = os.path.basename(image_path)
        image_name_cleaned = clean_image_name(image_name)

        print(f"\n=== Đang xử lý ảnh {idx}/{len(all_images)}: {image_path} ===")
        try:
            result, description, questions, user_answers,tops_label,converted_data = process_pipeline(str(image_path), image_name_cleaned)

            if not result:
                print(f"Không thể xử lý ảnh {image_name}.")
                continue

            print(f"Dự đoán: {result}")
            print(f"Thực tế: {image_name_cleaned}")

            is_correct = False
            if result:
                if isinstance(result, str):
                    is_correct = image_name_cleaned.lower() == result.lower()
                elif isinstance(result, list):
                    is_correct = any(image_name_cleaned.lower() == label_info.get("label", "").lower() for label_info in result)
            ket_qua = "Đúng" if is_correct else "Sai"

            print(f"Kết quả: {ket_qua}")
            print(f"Đường dẫn ảnh: {image_path}")

            new_result = {
                "STT": idx,
                "Ảnh": image_name,
                "Thực tế": image_name_cleaned,
                "Kết quả": ket_qua,
                "Đường dẫn": str(image_path),
                "Mô tả ảnh từ gemini": description,
                "Câu hỏi phân biệt": questions,
                "Trả lời câu hỏi": user_answers,
                "Kết quả lần 1":tops_label,
                "Kết quả lần 2": result,
                "Test":converted_data
            }

            save_to_json(new_result)
            results.append(new_result)

        except Exception as e:
            logging.error(f"Lỗi khi xử lý ảnh {image_name}: {e}")
            new_result = {
                "STT": idx,
                "Ảnh": image_name,
                "Thực tế": image_name_cleaned,
                "Kết quả": "Lỗi",
                "Đường dẫn": str(image_path),
                "Mô tả ảnh từ gemini": description,
                "Câu hỏi phân biệt": [],
                "Trả lời câu hỏi": [],
                "Kết quả lần 1": tops_label,
                "Kết quả lần 2": None,
                "Kết quả gộp nhãn": {}
            }
            save_to_json(new_result)
            results.append(new_result)

    print(f"\nHoàn tất. Đã xử lý {len(results)} ảnh. Kết quả được lưu tại {JSON_RESULT_PATH}.")
    return results

if __name__ == "__main__":
    download_all_required_files()
    test_process_pipeline()