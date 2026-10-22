import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


# =========================================================
# CACHE PATHS
# =========================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

CACHE_DIR = PROJECT_ROOT / "data" / "cache"

MATERIALS_CACHE_DIR = CACHE_DIR / "materials"
LITERATURE_CACHE_DIR = CACHE_DIR / "literature"
MOTIFS_CACHE_DIR = CACHE_DIR / "motifs"


for directory in [
    MATERIALS_CACHE_DIR,
    LITERATURE_CACHE_DIR,
    MOTIFS_CACHE_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# =========================================================
# DEFAULT CACHE LIFETIMES
# =========================================================

MATERIAL_CACHE_HOURS = 24

LITERATURE_CACHE_HOURS = 24

MOTIF_CACHE_HOURS = 24 * 30


# =========================================================
# HELPERS
# =========================================================

def utc_now():
    return datetime.now(
        timezone.utc
    )


def safe_filename(value):
    """
    Convert a value into a filesystem-safe name.
    """

    value = str(value).strip()

    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value,
    )

    return value.strip("_")


def make_hash(data):
    """
    Generate a stable hash from JSON-compatible data.
    """

    encoded = json.dumps(
        data,
        sort_keys=True,
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()[:20]


def write_cache(
    path,
    data,
):
    """
    Write cache data with metadata.
    """

    payload = {
        "cached_at":
            utc_now().isoformat(),

        "data":
            data,
    }

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )


def read_cache(
    path,
    max_age_hours,
):
    """
    Read cached data if it exists and has not expired.

    Returns:
        cached data when valid
        None when missing, invalid, or expired
    """

    if not path.exists():
        return None

    try:

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        cached_at_text = payload.get(
            "cached_at"
        )

        if not cached_at_text:
            return None

        cached_at = datetime.fromisoformat(
            cached_at_text
        )

        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(
                tzinfo=timezone.utc
            )

        age = (
            utc_now()
            - cached_at
        )

        if age > timedelta(
            hours=max_age_hours
        ):
            return None

        return payload.get(
            "data"
        )

    except Exception:
        return None


# =========================================================
# MATERIAL SEARCH CACHE
# =========================================================

def material_search_cache_key(
    min_band_gap,
    max_band_gap,
    max_energy_above_hull,
    elements=None,
    exclude_elements=None,
    formula=None,
    limit=20,
):
    """
    Create a cache key for a Materials Project search.
    """

    key_data = {
        "min_band_gap":
            float(min_band_gap),

        "max_band_gap":
            float(max_band_gap),

        "max_energy_above_hull":
            float(max_energy_above_hull),

        "elements":
            sorted(elements or []),

        "exclude_elements":
            sorted(
                exclude_elements or []
            ),

        "formula":
            formula,

        "limit":
            int(limit),
    }

    return make_hash(
        key_data
    )


def get_cached_material_search(
    min_band_gap,
    max_band_gap,
    max_energy_above_hull,
    elements=None,
    exclude_elements=None,
    formula=None,
    limit=20,
):
    key = material_search_cache_key(
        min_band_gap,
        max_band_gap,
        max_energy_above_hull,
        elements,
        exclude_elements,
        formula,
        limit,
    )

    path = (
        MATERIALS_CACHE_DIR
        / f"search_{key}.json"
    )

    return read_cache(
        path,
        MATERIAL_CACHE_HOURS,
    )


def set_cached_material_search(
    data,
    min_band_gap,
    max_band_gap,
    max_energy_above_hull,
    elements=None,
    exclude_elements=None,
    formula=None,
    limit=20,
):
    key = material_search_cache_key(
        min_band_gap,
        max_band_gap,
        max_energy_above_hull,
        elements,
        exclude_elements,
        formula,
        limit,
    )

    path = (
        MATERIALS_CACHE_DIR
        / f"search_{key}.json"
    )

    write_cache(
        path,
        data,
    )


# =========================================================
# MOTIF CACHE
# =========================================================

def get_cached_motif(
    material_id,
):
    """
    Retrieve cached structural analysis.
    """

    filename = (
        safe_filename(material_id)
        + ".json"
    )

    path = (
        MOTIFS_CACHE_DIR
        / filename
    )

    return read_cache(
        path,
        MOTIF_CACHE_HOURS,
    )


def set_cached_motif(
    material_id,
    data,
):
    """
    Cache structural motif analysis.
    """

    filename = (
        safe_filename(material_id)
        + ".json"
    )

    path = (
        MOTIFS_CACHE_DIR
        / filename
    )

    write_cache(
        path,
        data,
    )


# =========================================================
# LITERATURE CACHE
# =========================================================

def literature_cache_key(
    formula,
    material_id=None,
    context=None,
    from_year=None,
    to_year=None,
    limit=5,
):
    """
    Create a cache key for an OpenAlex literature search.
    """

    key_data = {
        "formula":
            str(formula).strip(),

        "material_id":
            material_id,

        "context":
            context,

        "from_year":
            from_year,

        "to_year":
            to_year,

        "limit":
            int(limit),
    }

    return make_hash(
        key_data
    )


def get_cached_literature(
    formula,
    material_id=None,
    context=None,
    from_year=None,
    to_year=None,
    limit=5,
):
    key = literature_cache_key(
        formula,
        material_id,
        context,
        from_year,
        to_year,
        limit,
    )

    path = (
        LITERATURE_CACHE_DIR
        / f"search_{key}.json"
    )

    return read_cache(
        path,
        LITERATURE_CACHE_HOURS,
    )


def set_cached_literature(
    data,
    formula,
    material_id=None,
    context=None,
    from_year=None,
    to_year=None,
    limit=5,
):
    key = literature_cache_key(
        formula,
        material_id,
        context,
        from_year,
        to_year,
        limit,
    )

    path = (
        LITERATURE_CACHE_DIR
        / f"search_{key}.json"
    )

    write_cache(
        path,
        data,
    )


# =========================================================
# CACHE STATUS
# =========================================================

def get_cache_stats():
    """
    Return simple cache statistics for debugging/UI.
    """

    return {
        "material_searches":
            len(
                list(
                    MATERIALS_CACHE_DIR.glob(
                        "*.json"
                    )
                )
            ),

        "motif_analyses":
            len(
                list(
                    MOTIFS_CACHE_DIR.glob(
                        "*.json"
                    )
                )
            ),

        "literature_searches":
            len(
                list(
                    LITERATURE_CACHE_DIR.glob(
                        "*.json"
                    )
                )
            ),
    }