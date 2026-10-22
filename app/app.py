from pathlib import Path
import sys

import pandas as pd
import streamlit as st


# =========================================================
# PROJECT PATH
# =========================================================

from data_access import (
    load_candidates,
    get_related_literature,
)

from agent import run_agent

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MotifLens",
    page_icon="🔬",
    layout="wide",
)


# =========================================================
# BASIC STYLING
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f8fafc;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #102a43;
    }

    section[data-testid="stSidebar"] {
        background-color: #eef3f8;
        border-right: 1px solid #d8e1ea;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e1e8ef;
        border-radius: 12px;
        padding: 15px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e1e8ef;
        border-radius: 10px;
    }

    div[data-testid="stExpander"] {
        background-color: white;
        border: 1px solid #e1e8ef;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

BANNER_PATH = APP_DIR / "assets" / "motiflens_banner.png"

if BANNER_PATH.exists():

    left, center, right = st.columns([1, 4, 1])

    with center:
        st.image(
            str(BANNER_PATH),
            width=800,
        )

else:

    st.title("🔬 MotifLens")

    st.caption(
        "AI-Powered Motif-Guided Materials Discovery"
    )


st.caption(
    "Materials data: Materials Project  •  "
    "Scientific literature: OpenAlex"
)

st.markdown(
    "**Portfolio Edition**  •  "
    "Materials Informatics  •  "
    "Structured + Unstructured Scientific Data"
)

st.divider()


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def get_candidate_data():
    return load_candidates()


try:

    candidates = get_candidate_data()

except Exception as exc:

    st.error(
        "Could not load the MotifLens candidate dataset."
    )

    st.exception(exc)

    st.stop()


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("Material Filters")

min_gap = st.sidebar.number_input(
    "Minimum band gap (eV)",
    min_value=0.0,
    max_value=10.0,
    value=1.5,
    step=0.1,
)

max_gap = st.sidebar.number_input(
    "Maximum band gap (eV)",
    min_value=0.0,
    max_value=10.0,
    value=3.0,
    step=0.1,
)

max_hull = st.sidebar.number_input(
    "Maximum energy above hull (eV/atom)",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.01,
    format="%.3f",
)

result_limit = st.sidebar.slider(
    "Maximum results",
    min_value=5,
    max_value=100,
    value=25,
    step=5,
)


if min_gap > max_gap:

    st.sidebar.error(
        "Minimum band gap cannot exceed maximum band gap."
    )

    st.stop()


# =========================================================
# FILTER MATERIALS
# =========================================================

filtered = candidates[
    (candidates["band_gap"] >= min_gap)
    & (candidates["band_gap"] <= max_gap)
    & (candidates["energy_above_hull"] <= max_hull)
].copy()


filtered = filtered.sort_values(
    ["energy_above_hull", "band_gap"],
    ascending=[True, True],
).head(result_limit)


# =========================================================
# MATERIAL DISCOVERY
# =========================================================

st.header("Material Discovery")

st.subheader("Candidate Materials")

st.write(
    f"Found **{len(filtered)}** materials "
    "matching the current filters."
)


if filtered.empty:

    st.info(
        "No candidate materials match the current constraints."
    )

    st.stop()


# =========================================================
# METRICS
# =========================================================

metric1, metric2, metric3 = st.columns(3)


metric1.metric(
    "Candidates",
    len(filtered),
)


metric2.metric(
    "Average band gap",
    f"{filtered['band_gap'].mean():.2f} eV",
)


metric3.metric(
    "Most stable Eₕᵤₗₗ",
    f"{filtered['energy_above_hull'].min():.4f} eV/atom",
)


# =========================================================
# DISPLAY TABLE
# =========================================================

preferred_columns = [
    "material_id",
    "formula",
    "band_gap",
    "formation_energy_per_atom",
    "energy_above_hull",
    "density",
    "motifs",
    "motif_site_count",
    "mean_csm",
]


display_columns = [
    col
    for col in preferred_columns
    if col in filtered.columns
]


st.dataframe(
    filtered[display_columns],
    use_container_width=True,
    hide_index=True,
    column_config={

        "material_id":
            st.column_config.TextColumn(
                "Material ID"
            ),

        "formula":
            st.column_config.TextColumn(
                "Formula"
            ),

        "band_gap":
            st.column_config.NumberColumn(
                "Band Gap (eV)",
                format="%.3f"
            ),

        "formation_energy_per_atom":
            st.column_config.NumberColumn(
                "Formation Energy (eV/atom)",
                format="%.3f"
            ),

        "energy_above_hull":
            st.column_config.NumberColumn(
                "Energy Above Hull (eV/atom)",
                format="%.4f"
            ),

        "density":
            st.column_config.NumberColumn(
                "Density",
                format="%.3f"
            ),

        "motif_site_count":
            st.column_config.NumberColumn(
                "Motif Sites"
            ),

        "mean_csm":
            st.column_config.NumberColumn(
                "Mean CSM",
                format="%.3f"
            ),
    },
)


# =========================================================
# INSPECT CANDIDATE
# =========================================================

st.divider()

st.header("Inspect Candidate")


def material_label(material_id):

    row = filtered[
        filtered["material_id"] == material_id
    ].iloc[0]

    return (
        f"{material_id} — {row['formula']}"
    )


selected_material = st.selectbox(
    "Select a material",
    filtered["material_id"].tolist(),
    format_func=material_label,
)


selected = filtered[
    filtered["material_id"] == selected_material
].iloc[0]


left, right = st.columns(2)


with left:

    st.markdown("#### Material Properties")

    st.write(
        f"**Material ID:** {selected['material_id']}"
    )

    st.write(
        f"**Formula:** {selected['formula']}"
    )

    st.write(
        f"**Band gap:** {selected['band_gap']:.3f} eV"
    )

    st.write(
        "**Energy above hull:** "
        f"{selected['energy_above_hull']:.4f} eV/atom"
    )

    if "formation_energy_per_atom" in selected.index:

        st.write(
            "**Formation energy:** "
            f"{selected['formation_energy_per_atom']:.3f} eV/atom"
        )


with right:

    st.markdown("#### Structural Information")

    if "density" in selected.index:

        st.write(
            f"**Density:** {selected['density']:.3f}"
        )

    if "motif_site_count" in selected.index:

        st.write(
            f"**Motif sites:** {selected['motif_site_count']}"
        )

    if "mean_csm" in selected.index:

        st.write(
            f"**Mean CSM:** {selected['mean_csm']:.3f}"
        )

    if "motifs" in selected.index:

        st.write("**Structural motifs:**")

        st.write(
            selected["motifs"]
        )


# =========================================================
# RELATED LITERATURE
# =========================================================

st.divider()

st.header("📚 Related Scientific Literature")


try:

    literature = get_related_literature(
        selected_material,
        limit=5,
    )


    if literature.empty:

        st.info(
            "No processed literature is currently available "
            "for this candidate."
        )

    else:

        st.success(
            f"{len(literature)} related research "
            f"{'paper' if len(literature) == 1 else 'papers'} available."
        )

        st.caption(
            "Scientific literature originally retrieved from "
            "OpenAlex and processed by the MotifLens literature pipeline."
        )


        for _, paper in literature.iterrows():

            title = paper.get(
                "title",
                "Untitled publication",
            )

            year = paper.get(
                "publication_year",
                None,
            )

            if pd.notna(year):

                try:
                    year_text = str(int(float(year)))
                except Exception:
                    year_text = str(year)

                label = f"{title} ({year_text})"

            else:

                label = title


            with st.expander(label):

                abstract = paper.get(
                    "clean_abstract",
                    None,
                )

                if pd.notna(abstract):

                    st.write(
                        abstract
                    )


                relevance = paper.get(
                    "relevance_score",
                    None,
                )

                if pd.notna(relevance):

                    st.caption(
                        f"Literature relevance score: {relevance}"
                    )


                doi = paper.get(
                    "doi",
                    None,
                )

                if pd.notna(doi) and str(doi).strip():

                    doi_url = str(doi).strip()

                    if not doi_url.startswith("http"):

                        doi_url = (
                            "https://doi.org/"
                            + doi_url.replace(
                                "https://doi.org/",
                                ""
                            )
                        )

                    st.markdown(
                        f"[Open publication via DOI]({doi_url})"
                    )


except Exception as exc:

    st.warning(
        "Related literature could not be loaded."
    )

    st.exception(exc)


# =========================================================
# PORTFOLIO STATUS
# =========================================================

st.divider()

st.caption(
    "MotifLens Portfolio Edition • "
    "Independent deployment in progress • "
    "AI research agent and persistent project workspace coming next"
)

# =========================================================
# AI RESEARCH ASSISTANT
# =========================================================

st.divider()

st.header("🤖 MotifLens AI Research Assistant")

st.write(
    "Ask MotifLens to search candidate materials, investigate "
    "scientific literature, or combine both sources."
)

st.caption(
    "The assistant uses the processed Materials Project and "
    "OpenAlex portfolio datasets shown above."
)


# =========================================================
# EXAMPLE QUESTIONS
# =========================================================

st.markdown("**Example questions:**")

example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    st.info(
        "Find stable oxide candidates with band gaps "
        "between 1.5 and 2.5 eV."
    )

with example_col2:
    st.info(
        "What does the research literature say about FeO?"
    )

with example_col3:
    st.info(
        "Find stable oxide candidates between 1.5 and 2.5 eV, "
        "then summarize the literature for the best candidate "
        "that has research available."
    )


# =========================================================
# CHAT HISTORY
# =========================================================

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []


for message in st.session_state.agent_messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================================================
# CHAT INPUT
# =========================================================

user_prompt = st.chat_input(
    "Ask MotifLens about candidate materials or scientific literature..."
)


# =========================================================
# RUN AGENT
# =========================================================

if user_prompt:

    st.session_state.agent_messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)


    with st.chat_message("assistant"):

        with st.spinner(
            "MotifLens is analyzing materials and literature..."
        ):

            try:

                response = run_agent(
                    user_prompt
                )

                st.markdown(
                    response
                )

                st.session_state.agent_messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                    }
                )

            except Exception as exc:

                st.error(
                    "The MotifLens AI Research Assistant "
                    "encountered an error."
                )

                st.exception(exc)