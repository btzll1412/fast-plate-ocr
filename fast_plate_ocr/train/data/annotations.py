"""
Helpers to load annotations CSV files used for OCR training workflows.
"""

import os

import pandas as pd


def read_annotations_csv(annotations_file: str | os.PathLike) -> pd.DataFrame:
    """
    Read an annotations CSV with consistent `plate_text` handling.
    """
    annotations = pd.read_csv(annotations_file, dtype={"plate_text": str})
    annotations["plate_text"] = annotations["plate_text"].fillna("")
    return annotations
