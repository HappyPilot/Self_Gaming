"""Utilities to score local observer samples against HF vljepa probe datasets."""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

DATASETS_SERVER_FIRST_ROWS = "https://datasets-server.huggingface.co/first-rows"
DEFAULT_DATASET_IDS = [
    "ClarusC64/vljepa_structure_prediction_sentinel_v01",
    "ClarusC64/vljepa_physics_consistency_probe_v01",
    "ClarusC64/vljepa_agency_boundary_tests_v01",
    "ClarusC64/vljepa_occlusion_recovery_dataset_v01",
]

_ROW_FIELDS = (
    "expected_prediction",
    "expected_inference",
    "expected_behavior",
    "valid_inference",
    "structure_to_infer",
    "physics_principle",
    "boundary_to_respect",
    "continuity_expectation",
    "occluded_object",
    "actor_type",
    "video_scenario",
)
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "over",
    "into",
    "after",
    "before",
    "while",
    "still",
    "either",
    "unless",
    "then",
    "than",
    "when",
    "where",
    "near",
    "time",
    "frames",
    "frame",
    "stop",
}
_TOKEN_RE = re.compile(r"[a-z0-9_]{3,}")


def _tokenize(text: str) -> Set[str]:
    tokens = {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")}
    return {t for t in tokens if t not in _STOPWORDS}


def extract_probe_terms(row: Dict[str, Any]) -> Set[str]:
    terms: Set[str] = set()
    for field in _ROW_FIELDS:
        value = row.get(field)
        if isinstance(value, str):
            terms |= _tokenize(value)
    return terms


def _sample_blob(sample: Dict[str, Any]) -> str:
    chunks: List[str] = []

    scene = sample.get("scene")
    if isinstance(scene, dict):
        text = scene.get("text")
        if isinstance(text, list):
            chunks.extend(str(t) for t in text)
        elif isinstance(text, str):
            chunks.append(text)

        objects = scene.get("objects")
        if isinstance(objects, list):
            for obj in objects:
                if isinstance(obj, dict):
                    label = obj.get("label") or obj.get("class")
                    if label:
                        chunks.append(str(label))

    actions = sample.get("actions")
    if isinstance(actions, list):
        for entry in actions:
            payload = entry.get("payload") if isinstance(entry, dict) else None
            if isinstance(payload, dict):
                chunks.append(str(payload.get("action", "")))
                applied = payload.get("applied")
                if isinstance(applied, dict):
                    chunks.append(str(applied.get("action", "")))

    return " ".join(chunks)


def evaluate_probe_rows(rows: Sequence[Dict[str, Any]], samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    sample_tokens: Set[str] = set()
    for sample in samples:
        if isinstance(sample, dict):
            sample_tokens |= _tokenize(_sample_blob(sample))

    unmatched: List[int] = []
    matched = 0
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            unmatched.append(idx)
            continue
        terms = extract_probe_terms(row)
        if terms and (terms & sample_tokens):
            matched += 1
        else:
            unmatched.append(idx)

    total = len(rows)
    pct = round((100.0 * matched / total), 2) if total else 0.0
    return {
        "rows_total": total,
        "rows_matched": matched,
        "coverage_pct": pct,
        "unmatched_indices": unmatched,
        "sample_token_count": len(sample_tokens),
    }


def fetch_dataset_rows(dataset_id: str, config: str = "default", split: str = "train", timeout_sec: float = 20.0) -> List[Dict[str, Any]]:
    query = urllib.parse.urlencode({"dataset": dataset_id, "config": config, "split": split})
    url = f"{DATASETS_SERVER_FIRST_ROWS}?{query}"
    with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
        payload = json.load(resp)
    rows = payload.get("rows") or []
    out: List[Dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("row"), dict):
            out.append(row["row"])
    return out


def load_samples(samples_path: Path) -> List[Dict[str, Any]]:
    try:
        raw = json.loads(samples_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []


def run_probe_eval(samples: Sequence[Dict[str, Any]], dataset_ids: Iterable[str] | None = None) -> Dict[str, Any]:
    ids = list(dataset_ids or DEFAULT_DATASET_IDS)
    datasets: Dict[str, Any] = {}
    total_rows = 0
    total_matched = 0
    for dataset_id in ids:
        rows = fetch_dataset_rows(dataset_id)
        report = evaluate_probe_rows(rows, samples)
        datasets[dataset_id] = report
        total_rows += report["rows_total"]
        total_matched += report["rows_matched"]

    overall_pct = round((100.0 * total_matched / total_rows), 2) if total_rows else 0.0
    return {
        "datasets": datasets,
        "overall": {
            "rows_total": total_rows,
            "rows_matched": total_matched,
            "coverage_pct": overall_pct,
        },
    }
