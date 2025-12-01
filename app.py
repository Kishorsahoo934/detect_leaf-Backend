import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError


app = FastAPI()

# CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# Load YOLO Leaf Detector
# ============================
leaf_detector = None

try:
    leaf_detector = YOLO("best.pt")   # Load once
    print("✅ YOLO leaf detector loaded.")
except Exception as e:
    print("❌ Failed to load YOLO:", e)
    leaf_detector = None


# ============================
# Leaf Detection Function
# ============================
def detect_leaf(image):
    try:
        results = leaf_detector.predict(image, conf=0.50)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return False
        return True

    except Exception as e:
        print("YOLO error:", e)
        return False


# ============================
# API Endpoint
# ============================
@app.post("/detect-leaf")
async def detect_leaf_api(file: UploadFile = File(...)):

    if leaf_detector is None:
        raise HTTPException(status_code=500, detail="Leaf detector not loaded.")

    # Validate image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image.")

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Run YOLO detection
    is_leaf = detect_leaf(image)

    return {
        "leaf_detected": is_leaf
    }
