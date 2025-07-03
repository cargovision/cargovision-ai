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

# ?  SETUP
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cargovision Multi-Model API",
    description="API for detecting dangerous goods (detection) and item types (segmentation).",
    version="2.1.0"
)

# ?  MODEL PATHS
MODEL_PATH_DANGEROUS = "models/xray_cargo_dangerous.pt"
MODEL_PATH_ITEMS = "models/xray_cargo_items.pt"

# ?  LOAD MODELS
models = {}
try:
    if os.path.exists(MODEL_PATH_DANGEROUS):
        models["dangerous_goods"] = YOLO(MODEL_PATH_DANGEROUS)
        logger.info(f"Dangerous Goods (Detection) model loaded from: {MODEL_PATH_DANGEROUS}")
    else:
        logger.warning(f"Dangerous Goods model not found at: {MODEL_PATH_DANGEROUS}")
except Exception as e:
    logger.error(f"Failed to load Dangerous Goods model: {e}")

try:
    if os.path.exists(MODEL_PATH_ITEMS):
        models["item_types"] = YOLO(MODEL_PATH_ITEMS)
        logger.info(f"Item Types (Segmentation) model loaded from: {MODEL_PATH_ITEMS}")
    else:
        logger.warning(f"Item Types model not found at: {MODEL_PATH_ITEMS}")
except Exception as e:
    logger.error(f"Failed to load Item Types model: {e}")


# ?  RESPONSE MODELS

# ? For Object Detection (Dangerous Goods)
class DetectionBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    box: DetectionBox

class DetectionResponse(BaseModel):
    filename: str
    model_used: str
    detections: List[DetectionResult]

# ?  For Instance Segmentation (Item Types)
class Point(BaseModel):
    x: int
    y: int
class SegmentationResult(BaseModel):
    class_name: str
    confidence: float
    polygon: List[Point]

class SegmentationResponse(BaseModel):
    filename: str
    model_used: str
    detections: List[SegmentationResult]


# ?  HELPER funcs

# ? For drawing detection boxes
def draw_detections(image: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
    for detection in detections:
        box = detection.box
        label = f"{detection.class_name}: {detection.confidence:.2f}"
        cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), color=(0, 0, 255), thickness=2)
        cv2.putText(image, label, (box.x1, box.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return image

# ?  For drawing segmentation masks/polygons
def draw_segmentations(image: np.ndarray, detections: List[SegmentationResult]) -> np.ndarray:
    """Draws segmentation polygons and labels on an image."""
    overlay = image.copy()
    alpha = 0.4  # ? Transparency factor

    for detection in detections:
        # ? Convert list of Pydantic Point models to a NumPy array for OpenCV
        points = np.array([(p.x, p.y) for p in detection.polygon], dtype=np.int32)

        # ? Draw the filled polygon on the overlay
        cv2.fillPoly(overlay, [points], color=(0, 255, 0))

    # ? Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # ? draw the outlines and labels on the blended image
    for detection in detections:
        points = np.array([(p.x, p.y) for p in detection.polygon], dtype=np.int32)
        label = f"{detection.class_name}: {detection.confidence:.2f}"

        # ? Draw the polygon outline
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        # ? Put the label at the top-left corner of the polygon
        label_pos = (points[0][0], points[0][1] - 10)
        cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return image

# ?  INFERENCE PROCESSING

# ? For Object Detection models
def process_detection_inference(model: YOLO, image: np.ndarray) -> List[DetectionResult]:
    results = model(image)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append(DetectionResult(
                class_name=model.names[cls_id],
                confidence=round(conf, 2),
                box=DetectionBox(x1=x1, y1=y1, x2=x2, y2=y2)
            ))
    return detections

# ?  For Instance Segmentation models
def process_segmentation_inference(model: YOLO, image: np.ndarray) -> List[SegmentationResult]:
    """Runs segmentation model inference and returns polygon data."""
    # ? Set a lower confidence threshold to see more results, since the model is still learning
    results = model(image, conf=0.2)
    detections = []
    for r in results:
        # ? A segmentation model's result object has a 'masks' attribute
        if r.masks is None:
            continue

        # ? Iterate through both masks and boxes to get all info
        for i, mask in enumerate(r.masks):
            # ? The box contains the confidence and class ID
            box = r.boxes[i]
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # ? The mask contains the polygon points
            polygon_points = mask.xy[0] # ? Get the points for the first (and only) polygon

            # ? Convert numpy points to Pydantic Point models
            points = [Point(x=int(p[0]), y=int(p[1])) for p in polygon_points]

            detections.append(SegmentationResult(
                class_name=model.names[cls_id],
                confidence=round(conf, 2),
                polygon=points
            ))
    return detections


# ?  API ENDPOINTS

# ? DANGEROUS GOODS (DETECTION)
@app.post("/inspect/dangerous-goods/", response_model=DetectionResponse, tags=["JSON Inspection"])
async def inspect_dangerous_goods(file: UploadFile = File(...)):
    model_key = "dangerous_goods"
    if model_key not in models:
        raise HTTPException(status_code=500, detail=f"Model '{model_key}' is not available.")
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    detections = process_detection_inference(models[model_key], img)
    return DetectionResponse(filename=file.filename, model_used=model_key, detections=detections)

# ?  ITEM TYPES (SEGMENTATION)
@app.post("/inspect/item-types/", response_model=SegmentationResponse, tags=["JSON Inspection"])
async def inspect_item_types(file: UploadFile = File(...)):
    model_key = "item_types"
    if model_key not in models:
        raise HTTPException(status_code=500, detail=f"Model '{model_key}' is not available.")
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # ?  Calling the NEW segmentation function
    detections = process_segmentation_inference(models[model_key], img)
    return SegmentationResponse(filename=file.filename, model_used=model_key, detections=detections)


# ? DANGEROUS GOODS (VISUALIZATION)
@app.post("/visualize/dangerous-goods/", summary="Visualize Dangerous Goods Detections", tags=["Visual Inspection"])
async def visualize_dangerous_goods(file: UploadFile = File(...)):
    model_key = "dangerous_goods"
    if model_key not in models:
        raise HTTPException(status_code=500, detail=f"Model '{model_key}' is not available.")
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    detections = process_detection_inference(models[model_key], img)
    img_with_detections = draw_detections(img, detections)
    is_success, buffer = cv2.imencode(".jpg", img_with_detections)
    if not is_success:
        raise HTTPException(status_code=500, detail="Failed to encode visualized image.")
    return StreamingResponse(io.BytesIO(buffer), media_type="image/jpeg")

# ?  ITEM TYPES
@app.post("/visualize/item-types/", summary="Visualize Item Type Detections (Segmentation)", tags=["Visual Inspection"])
async def visualize_item_types(file: UploadFile = File(...)):
    model_key = "item_types"
    if model_key not in models:
        raise HTTPException(status_code=500, detail=f"Model '{model_key}' is not available.")
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # ?  call segmentation funcs
    detections = process_segmentation_inference(models[model_key], img)
    img_with_detections = draw_segmentations(img, detections)
    is_success, buffer = cv2.imencode(".jpg", img_with_detections)
    if not is_success:
        raise HTTPException(status_code=500, detail="Failed to encode visualized image.")
    return StreamingResponse(io.BytesIO(buffer), media_type="image/jpeg")


# ?  ROOT ENDPOINT
@app.get("/", summary="Health Check", tags=["General"])
def read_root():
    return {
        "status": "Cargovision Multi-Model API is running!",
        "loaded_models": list(models.keys())
    }
