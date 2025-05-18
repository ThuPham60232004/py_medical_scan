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

# --- Biến toàn cục ---
index = None
labels = {}
anomaly_index = None
anomaly_labels = {}

# --- Hàm upload ảnh lên GCS ---
def upload_to_gcs(local_path, destination_blob_name):
    client = storage.Client.from_service_account_json(GCS_KEY_PATH)
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    logging.info(f"Đã upload file lên GCS: gs://{GCS_BUCKET}/{destination_blob_name}")

# --- Nhận ảnh từ user và lưu tạm thời lên GCS ---
def save_image_from_user(image_bytes, filename):
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / filename
    with open(tmp_path, "wb") as f:
        f.write(image_bytes)
    logging.info(f"Lưu ảnh tạm thời thành công: {tmp_path}")
    upload_to_gcs(str(tmp_path), GCS_IMAGE_PATH + filename)
    return str(tmp_path)

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
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # (width, height)
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
        upload_to_gcs(str(processed_path), GCS_IMAGE_PATH + processed_path.name)
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

def test_process_pipeline():
    Image_dir = "app/static/data_test"
    get_all_images(Image_dir)
    right_result=0
    wrong_result=0
    total_images=0
    # duyệt qua từng ảnh trong thư mục
    for image_path in get_all_images(Image_dir):
        #Lấy tên ảnh bằng cách lấy tên file và thay thế các dấu _ bằng khoảng trắng
        image_name = os.path.basename(image_path).replace("_", " ")

        print(f"Processing {image_path}...")
        result=process_pipeline(image_path)
        print("Done.\n")
        # Kiểm tra kết quả
        if result == image_name:
            right_result+=1
        else:
            wrong_result+=1
        total_images+=1
    print(f"Kết quả đúng : {right_result},Tỉ lệ đúng là: {right_result/total_images*100}%")
    print(f"Kết quả sai : {wrong_result},Tỉ lệ sai là: {wrong_result/total_images*100}%")

def main():
    image_path = "app/static/img_test/cellulitis.webp"
    if not os.path.exists(image_path):
        logging.error("Ảnh đầu vào không tồn tại.")
        return
    load_faiss_index()
    label = process_pipeline(image_path)
    logging.info(f"Kết quả cuối cùng cho ảnh {image_path}: {label}")

if __name__ == "__main__":
    main()
