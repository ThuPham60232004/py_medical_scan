import json
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
import re
load_dotenv()

GCS_BUCKET = "test_storage_1000000image"
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

    processed = preprocess_image(image_path)
    if processed is not None:
        processed_path = f"app/static/processed/{Path(image_path).stem}_processed.jpg"
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        cv2.imwrite(processed_path, processed)

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
    if not label:
        return ""
    parts = label.strip().split("-")
    if len(parts) > 1:
        return " ".join(parts[1:]).lower()  # Thay dấu gạch ngang bằng khoảng trắng
    return label.lower()

def clean_actual_label(label: str) -> str:
    # Thay gạch ngang bằng khoảng trắng
    text = label.replace("-", " ")
    # Chỉ giữ lại ký tự chữ (a-z, A-Z) và khoảng trắng
    result = ''.join(char for char in text if char.isalpha() or char.isspace())
    return result

def test_process_pipeline():
    image_dir = "app/static/test_data_searching-200000"
    right_result = 0
    wrong_result = 0
    total_images = 0
    results = []

    for idx, image_path in enumerate(get_all_images(image_dir), 1):
        image_name = os.path.basename(image_path).split(".")[0].replace("_", " ").lower()

        print(f"\n=== Processing: {image_path} ===")
        result = process_pipeline(image_path)

        predicted_label = clean_predicted_label(result)
        actual_label = clean_actual_label(image_name)

        print(f"Dự đoán: {predicted_label}")
        print(f"Thực tế: {actual_label}")

        is_correct = predicted_label.strip() == actual_label.strip()
        result_status = "ĐÚNG" if is_correct else "SAI"
        print(f"Kết quả: {result_status}")

        if is_correct:
            right_result += 1
        else:
            wrong_result += 1
        total_images += 1

        # Thêm thông tin vào danh sách kết quả, chuyển image_path thành string
        results.append({
            "stt": idx,
            "ảnh": str(image_path),  # Chuyển WindowsPath thành string
            "tên ảnh": image_name,
            "đầu vào": actual_label,
            "đầu ra": predicted_label,
            "kết quả": result_status
        })

    if total_images == 0:
        print("Không tìm thấy ảnh nào trong thư mục test.")
        return

    # Tổng kết
    summary = {
        "kết quả đúng": right_result,
        "tỉ lệ đúng": f"{right_result / total_images * 100:.2f}%",
        "kết quả sai": wrong_result,
        "tỉ lệ sai": f"{wrong_result / total_images * 100:.2f}%"
    }

    # Tạo dữ liệu JSON
    output_data = {
        "results": results,
        "summary": summary
    }

    # Lưu vào file JSON
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print("\n================== TỔNG KẾT ==================")
    print(f"Kết quả đúng : {right_result}, Tỉ lệ đúng là: {right_result / total_images * 100:.2f}%")
    print(f"Kết quả sai  : {wrong_result}, Tỉ lệ sai là: {wrong_result / total_images * 100:.2f}%")
    print("Kết quả đã được lưu vào file 'test_results.json'.")
    
def main():
    load_faiss_index()
    test_process_pipeline()

if __name__ == "__main__":
    main()