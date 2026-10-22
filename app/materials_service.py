import os

import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester


from cache_service import (
    get_cached_material_search,
    set_cached_material_search,
)


load_dotenv()


def get_mp_client():
    """
    Create a Materials Project API client.
    """

    api_key = os.getenv("MP_API_KEY")

    if not api_key:
        raise RuntimeError(
            "MP_API_KEY is not configured."
        )

    return MPRester(api_key)


def search_live_materials(
    min_band_gap=0.0,
    max_band_gap=10.0,
    max_energy_above_hull=1.0,
    elements=None,
    exclude_elements=None,
    formula=None,
    limit=25,
):
    """
    Search Materials Project live for candidate materials.

    Parameters
    ----------
    min_band_gap : float
        Minimum electronic band gap in eV.

    max_band_gap : float
        Maximum electronic band gap in eV.

    max_energy_above_hull : float
        Maximum energy above hull in eV/atom.

    elements : list[str] | None
        Elements that must be present.

    exclude_elements : list[str] | None
        Elements that must not be present.

    formula : str | None
        Optional chemical formula search.

    limit : int
        Maximum number of returned materials.
    """



        # =====================================================
    # CACHE CHECK
    # =====================================================

    cached = get_cached_material_search(
        min_band_gap=min_band_gap,
        max_band_gap=max_band_gap,
        max_energy_above_hull=max_energy_above_hull,
        elements=elements,
        exclude_elements=exclude_elements,
        formula=formula,
        limit=limit,
    )

    if cached is not None:

        print(
            "CACHE HIT: Materials Project search"
        )

        df = pd.DataFrame(cached)

        if not df.empty:
            df["cache_status"] = "HIT"

        return df


    print(
        "CACHE MISS: Materials Project search"
    )
    

    limit = max(
        1,
        min(int(limit), 100)
    )

    fields = [
        "material_id",
        "formula_pretty",
        "band_gap",
        "formation_energy_per_atom",
        "energy_above_hull",
        "density",
        "is_stable",
        "elements",
        "nelements",
        "volume",
        "symmetry",
    ]

    search_kwargs = {
        "band_gap": (
            float(min_band_gap),
            float(max_band_gap),
        ),
        "energy_above_hull": (
            0.0,
            float(max_energy_above_hull),
        ),
        "fields": fields,
    }

    if elements:
        search_kwargs["elements"] = elements

    if exclude_elements:
        search_kwargs["exclude_elements"] = (
            exclude_elements
        )

    if formula:
        search_kwargs["formula"] = formula

    with get_mp_client() as mpr:

        docs = mpr.materials.summary.search(
            **search_kwargs
        )

    records = []

    for doc in docs:

        symmetry = getattr(
            doc,
            "symmetry",
            None
        )

        if symmetry is not None:
            crystal_system = str(
                getattr(
                    symmetry,
                    "crystal_system",
                    ""
                )
            )

            space_group = getattr(
                symmetry,
                "symbol",
                None
            )

        else:
            crystal_system = None
            space_group = None

        records.append(
            {
                "material_id": str(
                    doc.material_id
                ),
                "formula": (
                    doc.formula_pretty
                ),
                "band_gap": doc.band_gap,
                "formation_energy_per_atom":
                    doc.formation_energy_per_atom,
                "energy_above_hull":
                    doc.energy_above_hull,
                "density": doc.density,
                "is_stable": doc.is_stable,
                "elements": ", ".join(
                    str(element)
                    for element in doc.elements
                ),
                "nelements": doc.nelements,
                "volume": doc.volume,
                "crystal_system":
                    crystal_system,
                "space_group":
                    space_group,
            }
        )

    df = pd.DataFrame(records)

    if df.empty:
        return df

    df = df.sort_values(
        [
            "energy_above_hull",
            "band_gap",
        ],
        ascending=[
            True,
            True,
        ],
    )

    result = df.head(
        limit
    ).copy()


    # =====================================================
    # SAVE CACHE
    # =====================================================

    result["cache_status"] = "MISS"

    set_cached_material_search(
        data=result.to_dict(
            orient="records"
        ),
        min_band_gap=min_band_gap,
        max_band_gap=max_band_gap,
        max_energy_above_hull=max_energy_above_hull,
        elements=elements,
        exclude_elements=exclude_elements,
        formula=formula,
        limit=limit,
    )


    return result