import os

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n_18_01_26.pt")
RTSP_URL = os.getenv("RTSP_URL", "rtsp://localhost:8554/mystream")

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.4"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.3"))
MAX_CENTROID_DISTANCE = int(os.getenv("MAX_CENTROID_DISTANCE", "80"))
MAX_MISSING_FRAMES = int(os.getenv("MAX_MISSING_FRAMES", "5"))

SHOW_WINDOW = os.getenv("SHOW_WINDOW", "true").strip().lower() == "true"
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
