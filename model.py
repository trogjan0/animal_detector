from ultralytics import YOLO

class AnimalDetector:

    def __init__(self, model_path="yolov8n.pt", conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf
        self.allowed_classes = {"dog", "cat"}

    def predict(self, image_path):
        """
        Start model
        """
        results = self.model(image_path, conf=self.conf)
        detections = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                confidence = float(box.conf[0])

                if cls_name in self.allowed_classes:
                    detections.append({
                        "class": cls_name,
                        "confidence": round(confidence, 3),
                        "bbox": box.xyxy[0].tolist()
                    })

        return detections

