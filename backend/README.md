# Backend (MVP)

This is a minimal FastAPI service that:

- Accepts a video upload
- Runs an MVP centroid tracker (OpenCV background subtraction)
- Returns per-frame centroids as JSON

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Health check: `http://localhost:8000/api/health`

