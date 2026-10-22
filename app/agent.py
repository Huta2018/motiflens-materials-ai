import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from data_access import (
    load_candidates,
    get_related_literature,
)

from materials_service import search_live_materials
from literature_service import search_material_literature
from motif_service import analyze_material_motifs


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not configured. "
        "Add it to your local .env file."
    )

client = OpenAI(
    api_key=OPENAI_API_KEY
)

MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.6",
)


# =========================================================
# TOOL 1: SEARCH MATERIALS
# =========================================================

def search_materials(
    min_band_gap=0.0,
    max_band_gap=10.0,
    max_energy_above_hull=1.0,
    elements=None,
    exclude_elements=None,
    formula=None,
    limit=20,
):
    """
    Search Materials Project live.

    If live retrieval fails or returns no results, fall back
    to the processed MotifLens materials sample.
    """

    limit = max(
        1,
        min(int(limit), 100),
    )

    # -----------------------------------------------------
    # LIVE MATERIALS PROJECT
    # -----------------------------------------------------

    try:

        df = search_live_materials(
            min_band_gap=min_band_gap,
            max_band_gap=max_band_gap,
            max_energy_above_hull=max_energy_above_hull,
            elements=elements,
            exclude_elements=exclude_elements,
            formula=formula,
            limit=limit,
        )

        if not df.empty:

            result = df.copy()

            result["data_source"] = (
                "Materials Project API (live)"
            )

            return result.to_dict(
                orient="records"
            )

    except Exception as exc:

        print(
            "Live Materials Project search failed. "
            "Using local fallback dataset. "
            f"Error: {exc}"
        )

    # -----------------------------------------------------
    # LOCAL MATERIALS FALLBACK
    # -----------------------------------------------------

    df = load_candidates()

    result = df[
        (df["band_gap"] >= float(min_band_gap))
        & (df["band_gap"] <= float(max_band_gap))
        & (
            df["energy_above_hull"]
            <= float(max_energy_above_hull)
        )
    ].copy()

    if formula:

        result = result[
            result["formula"]
            .astype(str)
            .str.lower()
            == str(formula).lower()
        ]

    # Arbitrary element inclusion/exclusion is intentionally
    # not emulated on the small fallback dataset because it
    # does not contain complete composition metadata.

    result = result.sort_values(
        [
            "energy_above_hull",
            "band_gap",
        ],
        ascending=[
            True,
            True,
        ],
    ).head(limit)

    result["data_source"] = (
        "MotifLens processed materials fallback"
    )

    useful_columns = [
        "material_id",
        "formula",
        "band_gap",
        "formation_energy_per_atom",
        "energy_above_hull",
        "density",
        "motifs",
        "motif_site_count",
        "mean_csm",
        "data_source",
    ]

    useful_columns = [
        column
        for column in useful_columns
        if column in result.columns
    ]

    return result[
        useful_columns
    ].to_dict(
        orient="records"
    )


# =========================================================
# TOOL 2: SEARCH LITERATURE
# =========================================================

def search_literature(
    formula,
    material_id=None,
    context=None,
    from_year=None,
    to_year=None,
    limit=5,
):
    """
    Search scientific literature using OpenAlex live.

    If live retrieval fails or returns no records, fall back
    to the processed MotifLens OpenAlex sample when a
    material_id is available.
    """

    limit = max(
        1,
        min(int(limit), 20),
    )

    # -----------------------------------------------------
    # LIVE OPENALEX
    # -----------------------------------------------------

    try:

        df = search_material_literature(
            formula=formula,
            material_id=material_id,
            context=context,
            limit=limit,
            from_year=from_year,
            to_year=to_year,
        )

        if not df.empty:

            result = df.copy()

            result["data_source"] = (
                "OpenAlex API (live)"
            )

            return result.to_dict(
                orient="records"
            )

    except Exception as exc:

        print(
            "Live OpenAlex search failed. "
            "Trying local literature fallback. "
            f"Error: {exc}"
        )

    # -----------------------------------------------------
    # LOCAL LITERATURE FALLBACK
    # -----------------------------------------------------

    if not material_id:
        return []

    try:

        df = get_related_literature(
            material_id,
            limit=limit,
        )

    except Exception as exc:

        print(
            "Local literature fallback failed. "
            f"Error: {exc}"
        )

        return []

    if df.empty:
        return []

    useful_columns = [
        "material_id",
        "formula",
        "title",
        "publication_year",
        "doi",
        "clean_abstract",
        "relevance_score",
    ]

    useful_columns = [
        column
        for column in useful_columns
        if column in df.columns
    ]

    result = df[
        useful_columns
    ].copy()

    result["data_source"] = (
        "MotifLens processed OpenAlex fallback"
    )

    return result.to_dict(
        orient="records"
    )


# =========================================================
# TOOL 3: ANALYZE STRUCTURAL MOTIFS
# =========================================================

def analyze_motifs(
    material_id,
):
    """
    Retrieve a live Materials Project crystal structure and
    analyze local coordination environments with pymatgen
    CrystalNN and ChemEnv.
    """

    try:

        result = analyze_material_motifs(
            material_id
        )

        return result

    except Exception as exc:

        return {
            "material_id":
                material_id,

            "error":
                str(exc),

            "analysis_source":
                "MotifLens motif analysis failed",
        }


# =========================================================
# OPENAI TOOL DEFINITIONS
# =========================================================

TOOLS = [

    # -----------------------------------------------------
    # SEARCH MATERIALS
    # -----------------------------------------------------

    {
        "type": "function",
        "name": "search_materials",

        "description": (
            "Search candidate materials using the live "
            "Materials Project API. Use for materials "
            "discovery, formula searches, chemistry constraints, "
            "band-gap filtering, and thermodynamic-stability "
            "screening. A small local fallback dataset is used "
            "if live retrieval is unavailable."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "min_band_gap": {
                    "type": "number",
                    "description":
                        "Minimum band gap in eV.",
                },

                "max_band_gap": {
                    "type": "number",
                    "description":
                        "Maximum band gap in eV.",
                },

                "max_energy_above_hull": {
                    "type": "number",
                    "description":
                        "Maximum energy above hull in eV/atom.",
                },

                "elements": {
                    "type": [
                        "array",
                        "null",
                    ],
                    "items": {
                        "type": "string"
                    },
                    "description": (
                        "Elements that must be present. "
                        "Example: ['Ti', 'O']."
                    ),
                },

                "exclude_elements": {
                    "type": [
                        "array",
                        "null",
                    ],
                    "items": {
                        "type": "string"
                    },
                    "description":
                        "Elements that must not be present.",
                },

                "formula": {
                    "type": [
                        "string",
                        "null",
                    ],
                    "description": (
                        "Optional chemical formula. "
                        "Example: Fe2O3."
                    ),
                },

                "limit": {
                    "type": "integer",
                    "description":
                        "Maximum number of materials to return.",
                },
            },

            "required": [],

            "additionalProperties": False,
        },
    },


    # -----------------------------------------------------
    # SEARCH LITERATURE
    # -----------------------------------------------------

    {
        "type": "function",
        "name": "search_literature",

        "description": (
            "Search scientific literature using the live "
            "OpenAlex API for a material formula and optional "
            "scientific context. Use this tool for papers, "
            "abstracts, publication years, journals, DOI links, "
            "authors, citation counts, and research topics."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "formula": {
                    "type": "string",
                    "description": (
                        "Chemical formula of the material. "
                        "Example: TiO2 or FeO."
                    ),
                },

                "material_id": {
                    "type": [
                        "string",
                        "null",
                    ],
                    "description": (
                        "Optional Materials Project ID. "
                        "Include it when available."
                    ),
                },

                "context": {
                    "type": [
                        "string",
                        "null",
                    ],
                    "description": (
                        "Scientific topic or application used "
                        "to focus the literature search. Examples: "
                        "photocatalysis, electronic structure, "
                        "spectroscopy, high-pressure behavior, "
                        "battery cathode, dielectric properties."
                    ),
                },

                "from_year": {
                    "type": [
                        "integer",
                        "null",
                    ],
                    "description":
                        "Optional earliest publication year.",
                },

                "to_year": {
                    "type": [
                        "integer",
                        "null",
                    ],
                    "description":
                        "Optional latest publication year.",
                },

                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum number of literature records "
                        "to return."
                    ),
                },
            },

            "required": [
                "formula"
            ],

            "additionalProperties": False,
        },
    },


    # -----------------------------------------------------
    # ANALYZE MOTIFS
    # -----------------------------------------------------

    {
        "type": "function",
        "name": "analyze_motifs",

        "description": (
            "Analyze the live crystal structure of a Materials "
            "Project material using pymatgen CrystalNN and "
            "ChemEnv. Use this tool for structural motifs, "
            "coordination environments, local geometry, "
            "coordination numbers, polyhedral environments, "
            "continuous symmetry measures, or structural "
            "comparisons between candidate materials."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "material_id": {
                    "type": "string",
                    "description": (
                        "Materials Project ID of the material "
                        "whose crystal structure should be "
                        "analyzed. Example: mp-1274279."
                    ),
                },
            },

            "required": [
                "material_id"
            ],

            "additionalProperties": False,
        },
    },
]


# =========================================================
# SYSTEM INSTRUCTIONS
# =========================================================

SYSTEM_PROMPT = """
You are MotifLens, an AI research assistant for scientific
materials discovery.

MotifLens combines live materials properties, live crystal
structures, live structural-motif analysis, and live scientific
literature.

You have three primary scientific tools.


TOOL 1: search_materials

- Uses the live Materials Project API whenever available.
- Supports band-gap constraints.
- Supports energy-above-hull constraints.
- Supports required and excluded chemical elements.
- Supports formula searches.
- Returns Materials Project identifiers and calculated
  materials properties.
- Falls back to a small processed MotifLens materials sample
  if live retrieval is unavailable.


TOOL 2: search_literature

- Uses the live OpenAlex API whenever available.
- Searches literature using material formula.
- Can focus retrieval using a scientific topic or application.
- Can constrain publication years.
- Returns titles, abstracts, DOI information, journals,
  authors, citation counts, topics, and publication years
  when available.
- Falls back to a small processed MotifLens literature sample
  when live retrieval is unavailable and a matching material
  ID is known.


TOOL 3: analyze_motifs

- Retrieves a live Materials Project crystal structure.
- Uses pymatgen CrystalNN for coordination analysis.
- Uses pymatgen ChemEnv for local coordination-geometry
  classification.
- Reports ChemEnv coordination-environment symbols.
- Reports continuous symmetry measures when available.
- Reports site-level and summarized structural environments.
- Use this tool whenever local coordination, structural motifs,
  polyhedra, geometry, or structural comparison is relevant.


IMPORTANT SCIENTIFIC RULES

1. Never invent materials, numerical properties, crystal
   structures, motifs, publications, DOI links, authors,
   measurements, or scientific claims.

2. Use search_materials whenever a question requires candidate
   discovery or property-based screening.

3. Use search_literature whenever a question requires scientific
   literature.

4. Use analyze_motifs whenever a question requires structural
   motifs, local coordination environments, coordination
   geometry, polyhedral environments, or structural comparison.

5. For combined discovery questions, use the tools sequentially
   as needed rather than attempting to answer from prior knowledge.

6. For a discovery + motif + literature workflow:

   a. Search candidate materials.
   b. Inspect the returned calculated properties.
   c. Select a manageable set of candidates using an explicit
      screening criterion.
   d. Analyze motifs for relevant candidate material IDs.
   e. Compare structural environments.
   f. Select a candidate using clearly stated criteria.
   g. Search literature using its formula and material ID.
   h. Pass the user's scientific application as literature
      search context.
   i. Synthesize only after retrieving the relevant evidence.

7. Do not assume the first material returned is automatically
   the scientifically best material.

8. If the user asks for the "best" or "most promising"
   candidate, explicitly state the criterion used.

9. Pay attention to provenance fields such as data_source,
   structure_source, and analysis_source.

10. If data_source is "Materials Project API (live)", describe
    those properties as live Materials Project results.

11. If data_source is "OpenAlex API (live)", describe those
    publications as live OpenAlex search results.

12. If a processed fallback dataset is used, clearly identify
    it as fallback data rather than live retrieval.

13. Materials Project properties such as band gap and energy
    above hull are calculated database properties unless the
    literature independently establishes experimental values.

14. Energy above hull is a calculated thermodynamic-stability
    indicator. It does not by itself prove experimental
    synthesizability or application performance.

15. Clearly distinguish Materials Project calculated properties
    from measurements and claims reported in scientific papers.

16. Do not infer local geometry from coordination number alone.

17. Use the ChemEnv result returned by analyze_motifs when
    describing local coordination geometry.

18. ChemEnv symbols such as O:6 and T:4 are structural
    coordination-environment labels. When useful, provide both
    the exact symbol and a human-readable interpretation.

19. O:6 generally denotes an octahedral six-coordinate
    environment and T:4 denotes a tetrahedral four-coordinate
    environment. Only make such interpretations when supported
    by the returned ChemEnv symbol.

20. Continuous symmetry measure (CSM) quantifies deviation from
    an ideal coordination geometry. Lower CSM generally means
    a closer match to that ideal geometry.

21. Do not describe CSM as an experimental measurement. It is a
    geometry-analysis descriptor calculated from the structure.

22. Distinguish structure-derived motif information from
    electronic and thermodynamic properties.

23. When comparing structures, call analyze_motifs for each
    relevant material. Do not assume compounds with similar
    formulas have identical local environments.

24. Do not claim application suitability solely from a structural
    motif. Treat motifs as one source of evidence.

25. Do not claim that a paper proves application suitability
    unless the retrieved paper actually supports that conclusion.

26. Citation count is not equivalent to scientific relevance.
    Prioritize topical relevance to the user's question.

27. If OpenAlex returns papers that only weakly match the
    requested topic, acknowledge that limitation.

28. If the tools cannot establish a claim, say so rather than
    filling the gap with unsupported information.

29. When presenting candidate materials, prefer a concise table
    containing useful fields such as:

    - formula
    - Materials Project ID
    - band gap
    - energy above hull
    - crystal system
    - data source

30. When presenting motif analysis, include useful information
    such as:

    - element
    - ChemEnv geometry symbol
    - site count
    - CrystalNN coordination number when relevant
    - mean CSM when available
    - structure source

31. When presenting literature evidence, include when available:

    - publication title
    - publication year
    - journal
    - DOI
    - scientific topic
    - relevant finding

32. Clearly separate three evidence categories in complex answers:

    A. Materials Project calculated properties
    B. MotifLens structure-derived motif analysis
    C. Findings reported in scientific literature

33. Keep answers scientifically useful, transparent, and
    reasonably concise.
"""


# =========================================================
# TOOL EXECUTION
# =========================================================

# =========================================================
# TOOL EXECUTION
# =========================================================

def execute_tool(
    tool_name,
    arguments,
):
    """
    Execute a MotifLens tool call.
    """

    if tool_name == "search_materials":

        return search_materials(
            **arguments
        )

    if tool_name == "search_literature":

        return search_literature(
            **arguments
        )

    if tool_name == "analyze_motifs":

        return analyze_motifs(
            **arguments
        )

    return {
        "error": (
            f"Unknown MotifLens tool: {tool_name}"
        )
    }


# =========================================================
# AGENT
# =========================================================

def run_agent(
    user_prompt,
):
    """
    Run the MotifLens AI research assistant.

    The agent can perform multiple sequential scientific
    operations including:

    1. Live Materials Project candidate discovery
    2. Live crystal-structure and motif analysis
    3. Live OpenAlex literature retrieval
    4. Scientific synthesis across those evidence sources
    """

    if not user_prompt:

        return (
            "Please provide a materials-science "
            "question or discovery request."
        )


    # =====================================================
    # INITIAL MODEL REQUEST
    # =====================================================

    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
        tools=TOOLS,
    )


    # =====================================================
    # MULTI-STEP TOOL LOOP
    # =====================================================

    for _ in range(12):

        function_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]


        # =================================================
        # FINAL RESPONSE
        # =================================================

        if not function_calls:

            final_text = response.output_text

            if final_text:

                return final_text

            return (
                "MotifLens completed the analysis but "
                "did not return a textual response."
            )


        # =================================================
        # EXECUTE TOOL CALLS
        # =================================================

        tool_outputs = []


        for call in function_calls:

            # ---------------------------------------------
            # Parse arguments
            # ---------------------------------------------

            try:

                arguments = json.loads(
                    call.arguments
                )

            except (
                json.JSONDecodeError,
                TypeError,
            ):

                arguments = {}


            # ---------------------------------------------
            # Execute tool
            # ---------------------------------------------

            try:

                result = execute_tool(
                    call.name,
                    arguments,
                )

            except Exception as exc:

                result = {
                    "error": str(exc),
                    "tool": call.name,
                }


            # ---------------------------------------------
            # Send tool result back to model
            # ---------------------------------------------

            tool_outputs.append(
                {
                    "type":
                        "function_call_output",

                    "call_id":
                        call.call_id,

                    "output":
                        json.dumps(
                            result,
                            default=str,
                        ),
                }
            )


        # =================================================
        # CONTINUE REASONING WITH TOOL RESULTS
        # =================================================

        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            previous_response_id=response.id,
            input=tool_outputs,
            tools=TOOLS,
        )


    # =====================================================
    # MAXIMUM TOOL-STEP SAFETY STOP
    # =====================================================

    return (
        "MotifLens reached the maximum number of "
        "scientific tool-use steps for this request. "
        "Try narrowing the materials search or asking "
        "about fewer candidates."
    )


# =========================================================
# OPTIONAL DIRECT TEST
# =========================================================

if __name__ == "__main__":

    test_question = """
    Analyze the structural motifs of FeO with Materials Project
    ID mp-1274279.

    Describe the Fe and O coordination environments and explain
    the ChemEnv geometry results.
    """

    print(
        run_agent(
            test_question
        )
    )