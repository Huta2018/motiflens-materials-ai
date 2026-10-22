from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sample"

CANDIDATES_FILE = DATA_DIR / "gold_candidate_discovery.csv"
LITERATURE_FILE = DATA_DIR / "silver_material_literature.csv"


# =========================================================
# CANDIDATE DATA
# =========================================================

def load_candidates():
    """
    Load the processed MotifLens candidate-discovery dataset.
    """

    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"Candidate dataset not found: {CANDIDATES_FILE}"
        )

    return pd.read_csv(CANDIDATES_FILE)


# =========================================================
# LITERATURE DATA
# =========================================================

def load_literature():
    """
    Load the processed MotifLens scientific-literature dataset.
    """

    if not LITERATURE_FILE.exists():
        raise FileNotFoundError(
            f"Literature dataset not found: {LITERATURE_FILE}"
        )

    return pd.read_csv(LITERATURE_FILE)


# =========================================================
# MATERIAL SEARCH
# =========================================================

def search_candidates(
    min_band_gap=0.0,
    max_band_gap=10.0,
    max_energy_above_hull=1.0,
    limit=25
):
    """
    Search candidate materials using the same core filters
    used by the MotifLens frontend.
    """

    df = load_candidates()

    result = df[
        (df["band_gap"] >= min_band_gap)
        & (df["band_gap"] <= max_band_gap)
        & (df["energy_above_hull"] <= max_energy_above_hull)
    ].copy()

    result = result.sort_values(
        ["energy_above_hull", "band_gap"],
        ascending=[True, True]
    )

    return result.head(limit)


# =========================================================
# RELATED LITERATURE
# =========================================================

def get_related_literature(material_id, limit=3):
    """
    Retrieve processed literature associated with a material.
    """

    df = load_literature()

    result = df[
        df["material_id"].astype(str) == str(material_id)
    ].copy()

    if "relevance_score" in result.columns:
        result = result.sort_values(
            "relevance_score",
            ascending=False
        )

    return result.head(limit)