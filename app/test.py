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
import datetime
import xml.etree.ElementTree as ET
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from app.pipeline.preprocess_image import preprocess_image
from app.pipeline.AIAgent.generate_description import generate_description_with_Gemini
from app.pipeline.gsc.download_all_required_files import download_all_required_files
from app.pipeline.anomaly import generate_anomaly_overlay,save_anomaly_outputs,embed_anomaly_heatmap
from app.pipeline.embedding import embed_image_clip
from app.pipeline.AIAgent.different_question import generate_discriminative_questions
from app.pipeline.AIAgent.diagnose_group import generate_diagnosis_with_gemini
from app.pipeline.AIAgent.final_diagnose import select_final_diagnosis_with_llm
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
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
vit_model = timm.create_model("vit_base_patch16_224", pretrained=True).to(device)
vit_model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def search_faiss_index(embedding: np.ndarray, index_path: str, label_path: str, top_k: int = 5):
    index = faiss.read_index(index_path)
    labels = np.load(label_path, allow_pickle=True)
    distances, indices = index.search(embedding, top_k)
    top_labels = [labels[idx] for idx in indices[0]]
    return list(zip(top_labels, distances[0]))

def search_faiss_anomaly_index(embedding: np.ndarray, index_path: str, label_path: str, top_k: int = 5):
    return search_faiss_index(embedding, index_path, label_path, top_k)
def normalize_group_name(group_name: str) -> str:
    """
    Chuẩn hoá tên nhóm bệnh:
    - Chuyển thành chữ thường
    - Xoá khoảng trắng dư thừa
    - Thay khoảng trắng bằng dấu gạch dưới
    - Loại bỏ ký tự đặc biệt nếu cần (nếu tên nhóm có dấu chấm, dấu ngoặc,...)

    Ví dụ: 'Fungal infections' → 'fungal_infections'
    """
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

    total_score = sum(score_dict.values())
    normalized_scores = {label: (score / total_score) * 100 for label, score in score_dict.items()}
    sorted_scores = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores

def format_diagnosis_output(diagnosis):
    if isinstance(diagnosis, tuple) and len(diagnosis) == 2:
        label, _ = diagnosis
        return str(label)
    elif isinstance(diagnosis, (str, np.str_)):
        return str(diagnosis)
    else:
        return str(diagnosis)

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
def normalize_diagnosis(diagnosis: str) -> str:
    """
    Chuẩn hóa diagnosis về định dạng chuẩn có dạng 'fungal_infections'.
    Chuyển các dấu gạch ngang, khoảng trắng thành gạch dưới.
    """
    return diagnosis.strip().lower().replace("-", "_").replace(" ", "_")

# ---------------------- MAIN FLOW ----------------------
def main():
    image_path = "app/static/image/Nail Fungus (4).jpg"
    # download_all_required_files()
    preprocessed_pil, preprocessed_np = preprocess_image(image_path)
    # ---------------------- PIPELINE 1 ----------------------
    description = generate_description_with_Gemini(image_path)
    print("Mô tả ảnh:", description)
    print("\nPhân loại ảnh đầy đủ (Full Image):")
    full_image_vector = embed_image_clip(preprocessed_pil)
    full_results = search_faiss_index(
        embedding=full_image_vector,
        index_path=os.path.join(LOCAL_SAVE_DIR, "faiss_index.bin"),
        label_path=os.path.join(LOCAL_SAVE_DIR, "labels.npy"),
        top_k=15
    )
    for label, score in full_results:
        print(f"  → {label} (score: {score:.4f})")
    print("\nPhân tích bất thường (Anomaly Detection):")
    anomaly_overlay, anomaly_map = generate_anomaly_overlay(preprocessed_pil)
    overlay_path, anomaly_map_path = save_anomaly_outputs(anomaly_overlay, anomaly_map, image_path)
    anomaly_vector = embed_anomaly_heatmap(overlay_path)
    anomaly_results = search_faiss_anomaly_index(
        embedding=anomaly_vector,
        index_path=os.path.join(LOCAL_SAVE_DIR, "faiss_index_anomaly.bin"),
        label_path=os.path.join(LOCAL_SAVE_DIR, "labels_anomaly.npy"),
        top_k=15
    )
    for label, score in anomaly_results:
        print(f"  → {label} (score: {score:.4f})")
    print("\nKết hợp kết quả từ Full Image + Anomaly:")
    combined_results = full_results + anomaly_results
    label_scores_raw = {}
    for label, distance in combined_results:
        similarity = 1 / (1 + distance)
        label_scores_raw[label] = label_scores_raw.get(label, 0) + similarity
    total_similarity = sum(label_scores_raw.values())
    label_scores_percent = {
        label: (score / total_similarity) * 100
        for label, score in label_scores_raw.items()
    }
    sorted_labels = sorted(label_scores_percent.items(), key=lambda x: x[1], reverse=True)

    print("Tổng điểm similarity sau khi chuẩn hóa (%):")
    for label, percent in sorted_labels:
        print(f"  → {label}: {percent:.2f}%")
    print("\nChẩn đoán nhóm bệnh với Gemini:")
    diagnosis = generate_diagnosis_with_gemini(description, combined_results)
    normalized_group_diagnosis = normalize_diagnosis(diagnosis)
    print(f"Chẩn đoán nhóm bệnh: {format_diagnosis_output(normalized_group_diagnosis)}")
    # ---------------------- PIPELINE 2 ----------------------
    combined_results=detailed_group_analysis(
        image_vector=full_image_vector,
        anomaly_vector=anomaly_vector,
        group_name=normalized_group_diagnosis,
        top_k=15
    )
    print(f"combined_results: {combined_results}")
    combined_results_final= [(str(label), float(score)) for label, score in combined_results]
    print(f"combined_results_final: {combined_results_final}")
    disease_primary = [label for label, _ in combined_results_final]
    print(disease_primary)    
    # ======= SINH CÂU HỎI PHÂN BIỆT ========
    questions = generate_discriminative_questions(description, disease_primary,normalized_group_diagnosis)
    if not questions:
        print("Không tạo được câu hỏi.")
        return

    # ======= HỎI NGƯỜI DÙNG TỪNG CÂU ========
    user_answers = []
    for i, question in enumerate(questions):
        print(f"\nCâu hỏi {i+1}: {question}")
        answer = input("→ Trả lời của bạn: ")
        user_answers.append(answer.strip())

    # ======= CHỌN NHÃN CUỐI CÙNG BẰNG LLM ========
    final_diagnosis = select_final_diagnosis_with_llm(
        caption=description,
        labels=disease_primary,
        questions=questions,
        answers=user_answers,
        group_disease_name=normalized_group_diagnosis
    )

    print(f"\n Nhãn được chọn cuối cùng bởi LLM: {final_diagnosis}")
