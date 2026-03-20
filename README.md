# AnyMaze-like (Rodent tracking MVP)

This is a small “AnyMaze-like” web app for rodent tracking/behavior analysis workflows.

Current MVP:

- Upload a video
- Run centroid tracking (OpenCV background subtraction)
- View trajectory overlay on the video
- Export raw tracking JSON

## Run (backend)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Run (frontend)

```bash
npm install
npm run dev
```

Then open `http://localhost:5173`.

## Next up

- Metrics (distance, speed, time-in-zones) + CSV export
- Arena ROI + zone editor
- Advanced engines: SLEAP/DeepLabCut (pose)

## Advanced engine: SLEAP (pose)

This repo does **not** vendor SLEAP; instead the backend can call the SLEAP CLI if you have it installed.

- Backend request: `POST /api/jobs?engine=sleap&model_paths=/path/model1,/path/model2`
- The job stores `predictions.slp` and `analysis.csv` in `backend/data/jobs/<job_id>/`

Install SLEAP per their docs, then make sure these CLIs exist in your PATH:

- `sleap-nn`
- `sleap`

## Pose QC (DeepLabCut / SLEAP)

Canonical top-down keypoints are: `nose`, `left_ear`, `right_ear`, `neck`, `mid_spine`, `hips`, `tail_base` (`canonical_schema_version: 7pt_v1`).

After a pose job completes:

- `POST /api/jobs/{job_id}/pose_qc` — compute metrics, write `pose_qc.json` + `pose_valid_mask.npy` next to `result.json`.
- `GET /api/jobs/{job_id}/result` includes `pose_provenance` when QC exists.
- Cohort pipeline `POST /api/cohorts/{id}/compute_pose_features?use_pose_qc_mask=true` (default) ANDs the feature validity mask with `pose_valid_mask.npy` when present.

Thresholds for Acceptable / Borderline / Poor tiers live in `backend/app/pose_qc_config.py`.
