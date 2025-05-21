import os
import cv2
import numpy as np
import torch
import faiss
from pathlib import Path
from collections import Counter
from google.cloud import storage
from transformers import CLIPProcessor, CLIPModel
import timm
from PIL import Image
from torchvision import transforms
from dotenv import load_dotenv

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
import re
# --- Cấu hình ---
GCS_BUCKET = "kltn-2025"
GCS_IMAGE_PATH = "uploaded_images/"
GCS_KEY_PATH = "app/gsc-key.json"  # Đường dẫn file key json

LOCAL_INDEX_PATH = "app/static/faiss/faiss_index.bin"
LOCAL_LABELS_PATH = "app/static/labels/labels.npy"
LOCAL_ANOMALY_INDEX_PATH = "app/static/faiss/faiss_index_anomaly.bin"
LOCAL_ANOMALY_LABELS_PATH = "app/static/labels/labels_anomaly.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load model ---
logging.info(f"Khởi tạo model trên device: {device}")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

vit_model = timm.create_model("vit_base_patch16_224", pretrained=True).to(device)
vit_model.eval()

index = None
labels = {}
anomaly_index = None
anomaly_labels = {}

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logging.warning("Ảnh đầu vào không hợp lệ, không thể tiền xử lý.")
        return None
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    edges = cv2.Canny(equalized, 50, 150)
    return edges

def generate_anomaly_map(image_path: str) -> np.ndarray:
    img_pil = Image.open(image_path).convert("RGB")
    original_size = img_pil.size 
    img_np = np.array(img_pil)  
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        features = vit_model.forward_features(img_tensor)
    feature_map = features.mean(dim=1).squeeze().cpu().numpy()
    anomaly_map = (feature_map - np.min(feature_map)) / (np.ptp(feature_map) + 1e-6)
    anomaly_map_resized = cv2.resize(anomaly_map, original_size, interpolation=cv2.INTER_CUBIC)
    anomaly_map_uint8 = (anomaly_map_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    _, binary_mask = cv2.threshold(anomaly_map_uint8, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    return overlay

def embed_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Không thể đọc ảnh để nhúng: {image_path}")
        return None
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    return embedding.cpu().numpy().astype(np.float32)

def embed_anomaly_map(anomaly_map_path: str):
    anomaly_map = cv2.imread(anomaly_map_path, cv2.IMREAD_GRAYSCALE)
    if anomaly_map is None:
        logging.warning(f"Không thể đọc anomaly map để nhúng: {anomaly_map_path}")
        return None
    anomaly_map_rgb = cv2.cvtColor(anomaly_map, cv2.COLOR_GRAY2RGB)
    inputs = clip_processor(images=anomaly_map_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    return embedding.cpu().numpy().astype(np.float32)

def load_faiss_index():
    global index, labels, anomaly_index, anomaly_labels

    if os.path.exists(LOCAL_INDEX_PATH):
        try:
            index = faiss.read_index(LOCAL_INDEX_PATH)
            logging.info(f"Đã tải FAISS index thường, tổng số vector: {index.ntotal}")
        except Exception as e:
            logging.error(f"Lỗi tải FAISS index thường: {e}")
            index = None
    else:
        logging.warning("Không tìm thấy FAISS index thường!")

    if os.path.exists(LOCAL_LABELS_PATH):
        labels = np.load(LOCAL_LABELS_PATH, allow_pickle=True).item()
        logging.info(f"Đã tải labels thường, số lượng: {len(labels)}")
    else:
        logging.warning("Không tìm thấy file labels thường!")

    if os.path.exists(LOCAL_ANOMALY_INDEX_PATH):
        try:
            anomaly_index = faiss.read_index(LOCAL_ANOMALY_INDEX_PATH)
            logging.info(f"Đã tải FAISS anomaly index, tổng số vector: {anomaly_index.ntotal}")
        except Exception as e:
            logging.error(f"Lỗi tải FAISS anomaly index: {e}")
            anomaly_index = None
    else:
        logging.warning("Không tìm thấy FAISS anomaly index!")

    if os.path.exists(LOCAL_ANOMALY_LABELS_PATH):
        anomaly_labels = np.load(LOCAL_ANOMALY_LABELS_PATH, allow_pickle=True).item()
        logging.info(f"Đã tải anomaly labels, số lượng: {len(anomaly_labels)}")
    else:
        logging.warning("Không tìm thấy anomaly labels!")

def search_similar_images(query_vector, top_k=5):
    if index is None or index.ntotal == 0:
        logging.warning("FAISS index thường trống hoặc chưa được tải!")
        return []
    distances, indices = index.search(query_vector, top_k)
    logging.info(f"Chỉ số tìm được trong FAISS thường: {indices}")

    results = []
    label_keys = list(labels.keys())
    for idx in indices[0]:
        idx = int(idx)
        if 0 <= idx < len(label_keys):
            key = label_keys[idx]
            results.append(labels.get(key, "unknown"))
        else:
            results.append("unknown")
    return results

def search_anomaly_images(query_vector, top_k=5):
    if anomaly_index is None or anomaly_index.ntotal == 0:
        logging.warning("FAISS anomaly index trống hoặc chưa được tải!")
        return []
    distances, indices = anomaly_index.search(query_vector, top_k)
    logging.info(f"Chỉ số tìm được trong FAISS anomaly: {indices}")

    results = []
    label_keys = list(anomaly_labels.keys())
    for idx in indices[0]:
        idx = int(idx)
        if 0 <= idx < len(label_keys):
            key = label_keys[idx]
            results.append(anomaly_labels.get(key, "unknown"))
        else:
            results.append("unknown")
    return results

def combine_labels(normal_labels: list, anomaly_labels: list) -> str:
    all_labels = normal_labels + anomaly_labels
    all_labels = [label.strip() for label in all_labels if label and label != "unknown"]
    return " ".join(all_labels).strip()

def decide_final_label(labels_list):
    if not labels_list:
        return "Không xác định"
    count = Counter(labels_list)
    final_label = count.most_common(1)[0][0]
    return final_label
def process_pipeline(image_path: str):
    logging.info(f"Bắt đầu xử lý pipeline cho ảnh: {image_path}")

    processed_edges = preprocess_image(image_path)
    processed_dir = Path("app/static/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    if processed_edges is not None:
        processed_path = processed_dir / f"{Path(image_path).stem}_processed.jpg"
        cv2.imwrite(str(processed_path), processed_edges)
        logging.info(f"Ảnh tiền xử lý đã lưu và upload: {processed_path}")
    else:
        logging.warning("Không tạo được ảnh tiền xử lý.")

    anomaly_map = generate_anomaly_map(image_path)
    if anomaly_map is not None:
        anomaly_map_path = processed_dir / f"{Path(image_path).stem}_anomaly_map.jpg"
        cv2.imwrite(str(anomaly_map_path), anomaly_map)
        logging.info(f"Anomaly map đã lưu: {anomaly_map_path}")
    else:
        logging.warning("Không tạo được anomaly map.")
        anomaly_map_path = None

    embedding = embed_image(image_path)
    normal_labels = []
    if embedding is not None:
        normal_labels = search_similar_images(embedding)
        logging.info(f"Nhãn tìm được từ ảnh gốc: {normal_labels}")
    else:
        logging.warning("Không tạo được embedding ảnh gốc.")

    anomaly_labels_list = []
    if anomaly_map_path is not None:
        anomaly_embedding = embed_anomaly_map(str(anomaly_map_path))
        if anomaly_embedding is not None:
            anomaly_labels_list = search_anomaly_images(anomaly_embedding)
            logging.info(f"Nhãn tìm được từ anomaly map: {anomaly_labels_list}")
        else:
            logging.warning("Không tạo được embedding anomaly map.")
    else:
        logging.warning("Không có anomaly map để embedding.")

    combined_label_str = combine_labels(normal_labels, anomaly_labels_list)
    logging.info(f"Nhãn tổng hợp: {combined_label_str}")

    final_label = decide_final_label(normal_labels + anomaly_labels_list)
    logging.info(f"Nhãn cuối cùng được chọn: {final_label}")

    return final_label

def get_all_images(directory):
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    return [p for p in Path(directory).rglob("*") if p.is_file() and p.suffix.lower() in exts]
def clean_predicted_label(label: str) -> str:
    if not label:
        return ""
    parts = label.strip().split("-")
    if len(parts) > 1:
        return "-".join(parts[1:]).lower()
    return label.lower()

def clean_actual_label(label: str) -> str:
    return re.sub(r"\(\d+\)", "", label).strip().lower()
def test_process_pipeline():
    image_dir = "app/static/data_test"
    right_result = 0
    wrong_result = 0
    total_images = 0

    for image_path in get_all_images(image_dir):
        image_name = os.path.basename(image_path).split(".")[0].replace("_", " ").lower()

        print(f"\n=== Processing: {image_path} ===")
        result = process_pipeline(image_path)

        predicted_label = clean_predicted_label(result)
        actual_label = clean_actual_label(image_name)

        print(f"Dự đoán: {predicted_label}")
        print(f"Thực tế: {actual_label}")

        if predicted_label.strip() == actual_label.strip():
            print("Kết quả: ĐÚNG")
            right_result += 1
        else:
            print("Kết quả: SAI")
            wrong_result += 1
        total_images += 1

    if total_images == 0:
        print("Không tìm thấy ảnh nào trong thư mục test.")
        return

    print("\n================== TỔNG KẾT ==================")
    print(f"Kết quả đúng : {right_result}, Tỉ lệ đúng là: {right_result / total_images * 100:.2f}%")
    print(f"Kết quả sai  : {wrong_result}, Tỉ lệ sai là: {wrong_result / total_images * 100:.2f}%")    


def main():
    load_faiss_index()
    test_process_pipeline()
if __name__ == "__main__":
    main()
