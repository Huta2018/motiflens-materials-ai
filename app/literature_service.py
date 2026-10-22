import re
from typing import Optional

import pandas as pd
import requests


from cache_service import (
    get_cached_literature,
    set_cached_literature,
)


OPENALEX_WORKS_URL = "https://api.openalex.org/works"


# =========================================================
# ABSTRACT RECONSTRUCTION
# =========================================================

def reconstruct_abstract(
    inverted_index,
):
    """
    Reconstruct an OpenAlex abstract from its inverted index.
    """

    if not inverted_index:
        return None

    positions = []

    for word, word_positions in inverted_index.items():

        for position in word_positions:

            positions.append(
                (
                    int(position),
                    word,
                )
            )

    positions.sort(
        key=lambda item: item[0]
    )

    abstract = " ".join(
        word
        for _, word in positions
    )

    return abstract.strip() or None


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(
    text,
):
    """
    Normalize whitespace in literature text.
    """

    if not text:
        return None

    text = re.sub(
        r"\s+",
        " ",
        str(text),
    )

    return text.strip()


# =========================================================
# OPENALEX SEARCH
# =========================================================

def search_openalex(
    query,
    limit=10,
    from_year=None,
    to_year=None,
    require_abstract=True,
):
    """
    Search OpenAlex live.

    Parameters
    ----------
    query : str
        Scientific search phrase.

    limit : int
        Maximum number of papers returned.

    from_year : int | None
        Optional minimum publication year.

    to_year : int | None
        Optional maximum publication year.

    require_abstract : bool
        If True, remove papers without reconstructed abstracts.
    """

    if not query or not str(query).strip():

        return pd.DataFrame()


    limit = max(
        1,
        min(
            int(limit),
            50,
        )
    )


    params = {
        "search": str(query).strip(),
        "per-page": limit,
    }


    filters = []


    if from_year is not None:

        filters.append(
            f"from_publication_date:{int(from_year)}-01-01"
        )


    if to_year is not None:

        filters.append(
            f"to_publication_date:{int(to_year)}-12-31"
        )


    if filters:

        params["filter"] = ",".join(
            filters
        )


    headers = {
        "User-Agent": (
            "MotifLens materials-discovery portfolio application"
        )
    }


    response = requests.get(
        OPENALEX_WORKS_URL,
        params=params,
        headers=headers,
        timeout=30,
    )


    response.raise_for_status()


    payload = response.json()

    works = payload.get(
        "results",
        [],
    )


    records = []


    for work in works:

        abstract = reconstruct_abstract(
            work.get(
                "abstract_inverted_index"
            )
        )


        abstract = clean_text(
            abstract
        )


        if (
            require_abstract
            and not abstract
        ):

            continue


        primary_location = (
            work.get("primary_location")
            or {}
        )


        source = (
            primary_location.get("source")
            or {}
        )


        authorships = (
            work.get("authorships")
            or []
        )


        authors = []

        for authorship in authorships[:10]:

            author = (
                authorship.get("author")
                or {}
            )

            name = author.get(
                "display_name"
            )

            if name:
                authors.append(name)


        concepts = []

        for topic in (
            work.get("topics")
            or []
        )[:5]:

            name = topic.get(
                "display_name"
            )

            if name:
                concepts.append(name)


        records.append(
            {
                "openalex_id":
                    work.get("id"),

                "title":
                    work.get("title"),

                "publication_year":
                    work.get(
                        "publication_year"
                    ),

                "publication_date":
                    work.get(
                        "publication_date"
                    ),

                "doi":
                    work.get("doi"),

                "journal":
                    source.get(
                        "display_name"
                    ),

                "authors":
                    "; ".join(authors),

                "cited_by_count":
                    work.get(
                        "cited_by_count"
                    ),

                "is_open_access":
                    (
                        work.get(
                            "open_access"
                        )
                        or {}
                    ).get(
                        "is_oa"
                    ),

                "topics":
                    "; ".join(
                        concepts
                    ),

                "clean_abstract":
                    abstract,

                "data_source":
                    "OpenAlex API (live)",
            }
        )


    df = pd.DataFrame(
        records
    )


    if df.empty:

        return df


    # Prefer recent and well-cited results while retaining
    # OpenAlex search relevance as the initial retrieval step.

    if "cited_by_count" in df.columns:

        df["cited_by_count"] = (
            pd.to_numeric(
                df["cited_by_count"],
                errors="coerce",
            )
            .fillna(0)
        )


    return df.head(limit)


# =========================================================
# MATERIAL-FOCUSED LITERATURE SEARCH
# =========================================================

def search_material_literature(
    formula,
    material_id=None,
    context=None,
    limit=10,
    from_year=None,
    to_year=None,
):
    """
    Search OpenAlex for literature relevant to a material.

    Formula is required. Optional context can describe the
    scientific property or application of interest.
    """

        # =====================================================
    # CACHE CHECK
    # =====================================================

    cached = get_cached_literature(
        formula=formula,
        material_id=material_id,
        context=context,
        from_year=from_year,
        to_year=to_year,
        limit=limit,
    )

    if cached is not None:

        print(
            f"CACHE HIT: literature search for "
            f"{formula} | {context}"
        )

        df = pd.DataFrame(cached)

        if not df.empty:
            df["cache_status"] = "HIT"

        return df


    print(
        f"CACHE MISS: literature search for "
        f"{formula} | {context}"
    )

    formula = str(
        formula
    ).strip()


    if not formula:

        return pd.DataFrame()


    query_parts = [
        f'"{formula}"',
        "material",
    ]


    if context:

        query_parts.append(
            str(context).strip()
        )


    query = " ".join(
        query_parts
    )


    df = search_openalex(
        query=query,
        limit=limit,
        from_year=from_year,
        to_year=to_year,
        require_abstract=True,
    )


    if df.empty:

        return df


    df.insert(
        0,
        "material_id",
        material_id,
    )


    df.insert(
        1,
        "formula",
        formula,
    )


        # =====================================================
    # SAVE CACHE
    # =====================================================

    df["cache_status"] = "MISS"

    set_cached_literature(
        data=df.to_dict(
            orient="records"
        ),
        formula=formula,
        material_id=material_id,
        context=context,
        from_year=from_year,
        to_year=to_year,
        limit=limit,
    )

    return df