from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from datetime import datetime
import shutil
import os
import json

from model import AnimalDetector

# ----------------- INIT -----------------

app = FastAPI(title="Animal Monitoring System")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
HISTORY_FILE = "history.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

detector = AnimalDetector(
    model_path="yolov8n.pt",
    conf=0.3
)

# ----------------- ROUTES -----------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detections = detector.predict(file_path)

    stats = {"dog": 0, "cat": 0}
    for d in detections:
        stats[d["class"]] += 1

    save_history(stats)

    return JSONResponse({
        "detections": detections,
        "summary": stats
    })


@app.get("/stats")
def get_stats():
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)

    total = {"dog": 0, "cat": 0}
    for h in history:
        for k in total:
            total[k] += h["detected"].get(k, 0)

    return {
        "total": total,
        "last_requests": history[-10:]
    }

# ----------------- HELPERS -----------------

def save_history(detected):
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "image",
        "detected": detected
    }

    with open(HISTORY_FILE, "r") as f:
        data = json.load(f)

    data.append(record)

    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

