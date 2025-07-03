from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from ultralytics import YOLO
import os
import shutil
from typing import List, Dict, Any
import logging
import io
import numpy as np
import cv2

# ? setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cargovision Multi-Model API",
    description="API for detecting dangerous goods and item types in cargo scans.",
    version="2.0.0"
)

MODEL_PATH_DANGEROUS = "models/xray_cargo_dangerous.pt"
MODEL_PATH_ITEMS = "models/xray_cargo_items.pt"

# ? load models
models = {}

try:
    if os.path.exists(MODEL_PATH_DANGEROUS):
        models["dangerous_goods"] = YOLO(MODEL_PATH_DANGEROUS)
        logger.info(f"Dangerous Goods model loaded from: {MODEL_PATH_DANGEROUS}")
    elif os.path.exists(MODEL_PATH_ITEMS):
        models["item_types"] = YOLO(MODEL_PATH_DANGEROUS)
        logger.info(f"Item Types model model loaded from: {MODEL_PATH_ITEMS}")
    else:
        logger.warning(f"Model path not found")
except Exception as e:
    logger.error(f"Failed to load  model: {e}")

try:
    if os.path.exists(MODEL_PATH_ITEMS):
        models["item_types"] = YOLO(MODEL_PATH_ITEMS)
        logger.info(f"Item Types model loaded from: {MODEL_PATH_ITEMS}")
    else:
        logger.warning(f"Item Types model not found at: {MODEL_PATH_ITEMS}")
except Exception as e:
    logger.error(f"Failed to load Item Types model: {e}")

# ? response models (template basically)
class DetectionBox(BaseModel):
    x1: int = Field(..., description="X coordinate of top-left corner")
    y1: int = Field(..., description="Y coordinate of top-left corner")
    x2: int = Field(..., description="X coordinate of bottom-right corner")
    y2: int = Field(..., description="Y coordinate of bottom-right corner")

class DetectionResult(BaseModel):
    class_name: str = Field(..., description="Name of detected class/label")
    confidence: float = Field(..., description="Confidence level of prediction (0.0 - 1.0)")
    box: DetectionBox = Field(..., description="Detection box coordinates")

class InspectionResponse(BaseModel):
    filename: str
    model_used: str
    detections: List[DetectionResult]

# ? helper funcs
def draw_detections(image: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
    """Draws detection boxes and labels on an image."""
    for detection in detections:
        box = detection.box
        class_name = detection.class_name
        confidence = detection.confidence
        # ?  draw rectangle
        cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), color=(0, 255, 0), thickness=2)
        # ?  prep & draw label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(image, label, (box.x1, box.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return image

# ? endpionts

#  ? helper func to run inference and process results --> avoid code duplication
def process_inference(model: YOLO, image: np.ndarray) -> List[DetectionResult]:
    """Runs model inference and returns a list of detection results."""
    results = model(image)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            detection_data = DetectionResult(
                class_name=model.names[cls_id],
                confidence=round(conf, 2),
                box=DetectionBox(x1=x1, y1=y1, x2=x2, y2=y2)
            )
            detections.append(detection_data)
    return detections

# ?  json enpdionts (default maybe?)

@app.post("/inspect/dangerous-goods/", response_model=InspectionResponse, tags=["JSON Inspection"])
async def inspect_dangerous_goods(file: UploadFile = File(...)):
    """Analyzes for security threats (e.g., sharp objects) and returns JSON data."""
    model_key = "dangerous_goods"
    if model_key not in models:
        raise HTTPException(status_code=500, detail=f"Model '{model_key}' is not available.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    detections = process_inference(models[model_key], img)
    return InspectionResponse(filename=file.filename, model_used=model_key, detections=detections)

@app.post("/inspect/item-types/", response_model=InspectionResponse, tags=["JSON Inspection"])
async def inspect_item_types(file: UploadFile = File(...)):
    """Analyzes for general item types and returns JSON data."""
    model_key = "item_types"
    if model_key not in models:
        raise HTTPException(status_code=500, detail=f"Model '{model_key}' is not available.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    detections = process_inference(models[model_key], img)
    return InspectionResponse(filename=file.filename, model_used=model_key, detections=detections)

# ? visualize endpints

@app.post("/visualize/dangerous-goods/", summary="Visualize Dangerous Goods Detections", tags=["Visual Inspection"])
async def visualize_dangerous_goods(file: UploadFile = File(...)):
    """Runs dangerous goods detection and returns the visualized image."""
    model_key = "dangerous_goods"
    if model_key not in models:
        raise HTTPException(status_code=500, detail=f"Model '{model_key}' is not available.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    detections = process_inference(models[model_key], img)
    img_with_detections = draw_detections(img, detections)

    is_success, buffer = cv2.imencode(".jpg", img_with_detections)
    if not is_success:
        raise HTTPException(status_code=500, detail="Failed to encode visualized image.")

    return StreamingResponse(io.BytesIO(buffer), media_type="image/jpeg")

@app.post("/visualize/item-types/", summary="Visualize Item Type Detections", tags=["Visual Inspection"])
async def visualize_item_types(file: UploadFile = File(...)):
    """Runs item type detection and returns the visualized image."""
    model_key = "item_types"
    if model_key not in models:
        raise HTTPException(status_code=500, detail=f"Model '{model_key}' is not available.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    detections = process_inference(models[model_key], img)
    img_with_detections = draw_detections(img, detections)

    is_success, buffer = cv2.imencode(".jpg", img_with_detections)
    if not is_success:
        raise HTTPException(status_code=500, detail="Failed to encode visualized image.")

    return StreamingResponse(io.BytesIO(buffer), media_type="image/jpeg")

# ? is running test
@app.get("/", summary="Health Check", tags=["General"])
def read_root():
    """Root endpoint to check API status and loaded models."""
    return {
        "status": "Cargovision Multi-Model API is running!",
        "loaded_models": list(models.keys())
    }
