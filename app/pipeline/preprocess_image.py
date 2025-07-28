import cv2
from PIL import Image

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
