"""
cohort_store.py — Multi-animal cohort data model and CRUD operations.

A cohort is a named set of job IDs with genotype / metadata annotations.
Each cohort is persisted in data/cohorts/{cohort_id}/cohort.json.
Pose feature matrices and motif labels are stored as .npy files in the same dir.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cohorts")


def _cohort_dir(cohort_id: str) -> str:
    return os.path.join(_BASE_DIR, cohort_id)


def _cohort_json(cohort_id: str) -> str:
    return os.path.join(_cohort_dir(cohort_id), "cohort.json")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CohortAnimal:
    job_id: str
    animal_id: str
    genotype: str           # e.g. "WT" or "BPAN"
    sex: str | None = None
    age_weeks: float | None = None
    treatment: str | None = None


@dataclass
class Cohort:
    cohort_id: str
    name: str
    animals: list[CohortAnimal] = field(default_factory=list)
    motif_library: dict | None = None       # set after motif discovery
    umap_params: dict | None = None         # set after embedding
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    pipeline_status: dict[str, str] = field(default_factory=dict)
    # Steps: pose_features | window_features | motif_discovery | embedding |
    #        sequence_analysis | phenotypes | group_comparison | classifier
    # Values: "pending" | "running" | "done" | "error"


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _cohort_to_dict(c: Cohort) -> dict:
    d = asdict(c)
    return d


def _dict_to_cohort(d: dict) -> Cohort:
    animals = [CohortAnimal(**a) for a in d.pop("animals", [])]
    return Cohort(animals=animals, **d)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def create_cohort(name: str) -> Cohort:
    cohort_id = str(uuid.uuid4())
    os.makedirs(_cohort_dir(cohort_id), exist_ok=True)
    c = Cohort(cohort_id=cohort_id, name=name)
    _save(c)
    return c


def get_cohort(cohort_id: str) -> Cohort | None:
    path = _cohort_json(cohort_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return _dict_to_cohort(json.load(f))


def list_cohorts() -> list[Cohort]:
    os.makedirs(_BASE_DIR, exist_ok=True)
    cohorts = []
    for entry in sorted(os.listdir(_BASE_DIR)):
        path = os.path.join(_BASE_DIR, entry, "cohort.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    cohorts.append(_dict_to_cohort(json.load(f)))
            except Exception:
                pass
    return cohorts


def add_animal(cohort_id: str, animal: CohortAnimal) -> Cohort:
    c = _require(cohort_id)
    # Remove duplicate job_id if re-adding
    c.animals = [a for a in c.animals if a.job_id != animal.job_id]
    c.animals.append(animal)
    _save(c)
    return c


def remove_animal(cohort_id: str, job_id: str) -> Cohort:
    c = _require(cohort_id)
    c.animals = [a for a in c.animals if a.job_id != job_id]
    _save(c)
    return c


def delete_cohort(cohort_id: str) -> None:
    import shutil
    d = _cohort_dir(cohort_id)
    if os.path.exists(d):
        shutil.rmtree(d)


def update_pipeline_status(cohort_id: str, step: str, status: str) -> None:
    c = _require(cohort_id)
    c.pipeline_status[step] = status
    _save(c)


def set_motif_library(cohort_id: str, library: dict) -> None:
    c = _require(cohort_id)
    c.motif_library = library
    _save(c)


def set_umap_params(cohort_id: str, params: dict) -> None:
    c = _require(cohort_id)
    c.umap_params = params
    _save(c)


# ---------------------------------------------------------------------------
# NumPy artefact helpers
# ---------------------------------------------------------------------------

def save_npy(cohort_id: str, name: str, arr: np.ndarray) -> str:
    """Save a .npy file in the cohort directory. Returns path."""
    path = os.path.join(_cohort_dir(cohort_id), f"{name}.npy")
    np.save(path, arr)
    return path


def load_npy(cohort_id: str, name: str) -> np.ndarray | None:
    path = os.path.join(_cohort_dir(cohort_id), f"{name}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=False)


def save_json(cohort_id: str, name: str, data: Any) -> str:
    path = os.path.join(_cohort_dir(cohort_id), f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_json(cohort_id: str, name: str) -> Any | None:
    path = os.path.join(_cohort_dir(cohort_id), f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def job_pose_features_path(job_id: str) -> str:
    """Path to per-job pose_features.npy (stored next to result.json)."""
    base = os.path.join(os.path.dirname(__file__), "..", "data", "jobs", job_id)
    return os.path.join(base, "pose_features.npy")


def job_motif_labels_path(job_id: str) -> str:
    base = os.path.join(os.path.dirname(__file__), "..", "data", "jobs", job_id)
    return os.path.join(base, "motif_labels.npy")


def job_phenotype_path(job_id: str) -> str:
    base = os.path.join(os.path.dirname(__file__), "..", "data", "jobs", job_id)
    return os.path.join(base, "phenotype.json")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save(c: Cohort) -> None:
    os.makedirs(_cohort_dir(c.cohort_id), exist_ok=True)
    with open(_cohort_json(c.cohort_id), "w") as f:
        json.dump(_cohort_to_dict(c), f, indent=2)


def _require(cohort_id: str) -> Cohort:
    c = get_cohort(cohort_id)
    if c is None:
        raise KeyError(f"Cohort not found: {cohort_id}")
    return c
