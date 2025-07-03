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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Cargovision API",
    description="API for detecting anomalies in cargo scan images using the YOLOv8 model.",
    version="1.0.0"
)

MODEL_PATH = "models/xray_cargo_best.pt"


if not os.path.exists(MODEL_PATH):
    error_msg = f"Model file not found at path: {MODEL_PATH}. Make sure you have run train.py and the path is correct."
    logger.error(error_msg)


try:
    model = YOLO(MODEL_PATH)
    logger.info(f"Model successfully loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

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
    detections: List[DetectionResult]

@app.post("/inspect/", response_model=InspectionResponse)
async def inspect_cargo(file: UploadFile = File(..., description="Cargo scan image file (JPG, PNG)")):
    if not model:
        raise HTTPException(status_code=500, detail="AI model failed to load, server not ready.")
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Analyzing file: {file.filename}")

        results = model(file_path)
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

        logger.info(f"Found {len(detections)} detections in {file.filename}")

        return InspectionResponse(filename=file.filename, detections=detections)

    except Exception as e:
        logger.error(f"Error occurred while processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error occurred: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def draw_detections(image: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
    """Draws detection boxes and labels on an image."""
    for detection in detections:
        box = detection.box
        class_name = detection.class_name
        confidence = detection.confidence

        # Draw rectangle
        cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), color=(0, 255, 0), thickness=2)

        # Prepare and draw label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(image, label, (box.x1, box.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return image

@app.post("/inspect/visualize/",
          summary="Inspect and Visualize Detections",
          description="Uploads a cargo scan image, runs detection, and returns the image with bounding boxes drawn on it.")
async def inspect_cargo_and_visualize(file: UploadFile = File(..., description="Cargo scan image file (JPG, PNG)")):
    if not model:
        raise HTTPException(status_code=500, detail="AI model failed to load, server not ready.")

    # Read uploaded file into memory
    contents = await file.read()
    # Convert to a NumPy array
    nparr = np.frombuffer(contents, np.uint8)
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        logger.info(f"Analyzing and visualizing file: {file.filename}")

        # Run model prediction
        results = model(img)
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

        logger.info(f"Found {len(detections)} detections in {file.filename}")

        # Draw detections on the image
        img_with_detections = draw_detections(img, detections)

        # Encode the image to a byte buffer
        is_success, buffer = cv2.imencode(".jpg", img_with_detections)
        if not is_success:
            raise HTTPException(status_code=500, detail="Failed to encode visualized image.")

        # Return the image as a streaming response
        return StreamingResponse(io.BytesIO(buffer), media_type="image/jpeg")

    except Exception as e:
        logger.error(f"Error occurred while processing file for visualization: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error occurred: {e}")




@app.get("/", summary="Health Check")
def read_root():
    return {"status": "Cargovision API is running!"}
