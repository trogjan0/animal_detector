import os
import uuid
import json
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from model import AnimalDetector
import pandas as pd

UPLOAD_DIR = "uploads"
HISTORY_FILE = "history.json"
EXCEL_REPORT = "report.xlsx"

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Animal Monitoring")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

templates = Jinja2Templates(directory="templates")

detector = AnimalDetector()


def load_history():
    if not os.path.exists(HISTORY_FILE) or os.stat(HISTORY_FILE).st_size == 0:
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(entry: dict):
    data = load_history()
    data.append(entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_excel():
    data = load_history()
    if not data:
        return
    df = pd.DataFrame(data)
    df.to_excel(EXCEL_REPORT, index=False)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/detect", response_class=HTMLResponse)
def detect(request: Request, file: UploadFile = File(...)):
    ext = file.filename.split('.')[-1]
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(file.file.read())

    detections = detector.predict(filepath)

    animals = {d['class'] for d in detections}
    message = ""
    if "dog" in animals and "cat" in animals:
        message = "Обнаружены кошка и собака"
    elif "dog" in animals:
        message = "Обнаружена собака"
    elif "cat" in animals:
        message = "Обнаружена кошка"
    else:
        message = "Животные не обнаружены"

    stats = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image": filename,
        "detections": ", ".join(animals) if animals else "нет",
        "count": len(detections)
    }

    save_history(stats)
    generate_excel()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "image_url": f"/uploads/{filename}",
            "message": message,
            "stats": stats
        }
    )


@app.get("/report")
def download_report():
    return FileResponse(EXCEL_REPORT, filename="animal_report.xlsx")