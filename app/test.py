import os
import cv2
import numpy as np
import torch
import faiss
from PIL import Image
from pathlib import Path
from collections import Counter
from google.cloud import storage
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

GCS_BUCKET = "kltn-2025"
GCS_IMAGE_PATH = "uploaded_images/"
GCS_KEY_PATH = "app/gsc-key.json"

LOCAL_INDEX_PATH = "app/static/faiss/faiss_index.bin"
LOCAL_LABELS_PATH = "app/static/labels/labels.npy"
INDEX_DIM = 512  

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

index = None
labels = {}

def get_all_images(directory):
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    return [p for p in Path(directory).rglob("*") if p.is_file() and p.suffix.lower() in exts]

def upload_to_gcs(local_path, destination_blob_name):
    client = storage.Client.from_service_account_json(GCS_KEY_PATH)
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded: gs://{GCS_BUCKET}/{destination_blob_name}")

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    edges = cv2.Canny(equalized, 50, 150)
    return edges

def embed_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.cpu().numpy().astype(np.float32)

def load_faiss_index():
    global index, labels
    if os.path.exists(LOCAL_INDEX_PATH):
        index = faiss.read_index(LOCAL_INDEX_PATH)
        print(f"FAISS index loaded: {index.ntotal} vectors.")
    else:
        print("FAISS index not found!")

    if os.path.exists(LOCAL_LABELS_PATH):
        labels = np.load(LOCAL_LABELS_PATH, allow_pickle=True).item()
        print(f"Labels loaded: {len(labels)}")
    else:
        print("Labels file not found!")

def search_similar_images(query_vector, top_k=5):
    if index is None or index.ntotal == 0:
        print("FAISS index is empty.")
        return []

    distances, indices = index.search(query_vector, top_k)
    results = []
    for i in indices[0]:
        if 0 <= i < len(labels):
            label_filename = list(labels.keys())[i]
            results.append(labels[label_filename])
        else:
            results.append("unknown")
    return results

def decide_final_label(result_labels):
    if not result_labels:
        return "Không xác định"
    label_counts = Counter(result_labels)
    final_label, _ = label_counts.most_common(1)[0]
    return final_label

def process_pipeline(image_path):
    upload_to_gcs(image_path, GCS_IMAGE_PATH + Path(image_path).name)

    processed = preprocess_image(image_path)
    if processed is not None:
        processed_path = f"app/static/processed/{Path(image_path).stem}_processed.jpg"
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        cv2.imwrite(processed_path, processed)
        upload_to_gcs(processed_path, GCS_IMAGE_PATH + Path(processed_path).name)

    embedding = embed_image(image_path)
    if embedding is None:
        print("Không thể nhúng ảnh.")
        return None

    result_labels = search_similar_images(embedding)
    print("Kết quả gán nhãn:")
    for idx, label in enumerate(result_labels, 1):
        print(f"{idx}. {label}")

    final_label = decide_final_label(result_labels)
    print(f"\nNhãn cuối cùng được chọn: {final_label}")
    return final_label

def clean_predicted_label(label: str) -> str:
    """
    Xóa tiền tố đầu tiên phân tách bằng dấu '-' và chuẩn hóa chữ thường.
    Ví dụ: 'ba-ai-impetigo' -> 'ai-impetigo'
    """
    if not label:
        return ""
    parts = label.strip().split("-")
    if len(parts) > 1:
        return "-".join(parts[1:]).lower()
    return label.lower()

def test_process_pipeline():
    image_dir = "app/static/data_test"
    right_result = 0
    wrong_result = 0
    total_images = 0

    for image_path in get_all_images(image_dir):
        image_name = os.path.basename(image_path).split(".")[0].replace("_", " ").lower()

        print(f"\n=== Processing: {image_path} ===")
        result = process_pipeline(image_path)
        clean_result = clean_predicted_label(result)

        print(f"Dự đoán: {clean_result}")
        print(f"Thực tế: {image_name}")

        if clean_result and clean_result in image_name:
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