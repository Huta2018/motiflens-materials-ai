import os
from collections import Counter, defaultdict

from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.analysis.local_env import CrystalNN

from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
    LocalGeometryFinder,
)
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
    SimplestChemenvStrategy,
)
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
    LightStructureEnvironments,
)

from cache_service import (
    get_cached_motif,
    set_cached_motif,
)


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# MATERIALS PROJECT CLIENT
# =========================================================

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


# =========================================================
# GET LIVE STRUCTURE
# =========================================================

def get_material_structure(material_id):
    """
    Retrieve the final crystal structure from Materials Project.
    """

    with get_mp_client() as mpr:

        structure = (
            mpr.get_structure_by_material_id(
                material_id,
                final=True,
            )
        )

    return structure


# =========================================================
# ADD OXIDATION STATES WHEN POSSIBLE
# =========================================================

def prepare_structure(structure):
    """
    Attempt to assign oxidation states.

    CrystalNN and ChemEnv can benefit from oxidation-state
    information. If assignment fails, preserve the original
    structure and continue.
    """

    structure = structure.copy()

    try:

        structure.add_oxidation_state_by_guess()

        oxidation_states_assigned = True

    except Exception:

        oxidation_states_assigned = False

    return (
        structure,
        oxidation_states_assigned,
    )


# =========================================================
# CRYSTALNN COORDINATION
# =========================================================

def get_crystalnn_coordination(structure):
    """
    Calculate site-level coordination numbers using CrystalNN.
    """

    cnn = CrystalNN()

    results = {}

    for site_index, _ in enumerate(
        structure
    ):

        try:

            cn = cnn.get_cn(
                structure,
                site_index,
                use_weights=False,
            )

            results[
                site_index
            ] = int(
                round(cn)
            )

        except Exception:

            results[
                site_index
            ] = None

    return results


# =========================================================
# CHEMENV ANALYSIS
# =========================================================

def get_chemenv_environments(structure):
    """
    Determine local coordination geometries using pymatgen
    ChemEnv.
    """

    lgf = LocalGeometryFinder()

    lgf.setup_structure(
        structure=structure
    )

    structure_environments = (
        lgf.compute_structure_environments(
            maximum_distance_factor=1.41,
            only_cations=False,
        )
    )

    strategy = (
        SimplestChemenvStrategy()
    )

    light_environments = (
        LightStructureEnvironments
        .from_structure_environments(
            strategy=strategy,
            structure_environments=(
                structure_environments
            ),
        )
    )

    return light_environments


# =========================================================
# SITE ELEMENT LABEL
# =========================================================

def get_element_label(site):
    """
    Return a clean element symbol even when oxidation states
    have been assigned.
    """

    try:

        return (
            site.specie
            .element
            .symbol
        )

    except Exception:

        try:

            return site.specie.symbol

        except Exception:

            return str(
                site.specie
            )


# =========================================================
# MAIN MOTIF ANALYSIS
# =========================================================

def analyze_material_motifs(material_id):
    """
    Analyze a live Materials Project crystal structure.

    The analysis combines:

    1. CrystalNN coordination numbers
    2. ChemEnv coordination geometry classification
    3. Continuous symmetry measure (CSM), when available

    Results are cached by Materials Project ID. Repeated
    requests for the same material can therefore avoid both
    another Materials Project request and another ChemEnv
    calculation until the cache expires.
    """

    # =====================================================
    # CACHE CHECK
    # =====================================================

    cached = get_cached_motif(
        material_id
    )

    if cached is not None:

        print(
            f"CACHE HIT: motif analysis for {material_id}"
        )

        cached[
            "cache_status"
        ] = "HIT"

        return cached


    print(
        f"CACHE MISS: motif analysis for {material_id}"
    )


    # =====================================================
    # RETRIEVE LIVE STRUCTURE
    # =====================================================

    raw_structure = (
        get_material_structure(
            material_id
        )
    )


    # =====================================================
    # PREPARE STRUCTURE
    # =====================================================

    (
        structure,
        oxidation_states_assigned,
    ) = prepare_structure(
        raw_structure
    )


    # =====================================================
    # CRYSTALNN
    # =====================================================

    crystalnn_results = (
        get_crystalnn_coordination(
            structure
        )
    )


    # =====================================================
    # CHEMENV
    # =====================================================

    light_environments = (
        get_chemenv_environments(
            structure
        )
    )


    # =====================================================
    # RESULT CONTAINERS
    # =====================================================

    site_results = []

    environment_counter = (
        Counter()
    )

    csm_by_environment = (
        defaultdict(list)
    )

    unresolved_sites = 0


    # =====================================================
    # PROCESS EACH CRYSTALLOGRAPHIC SITE
    # =====================================================

    for site_index, site in enumerate(
        structure
    ):

        element = (
            get_element_label(
                site
            )
        )

        crystalnn_cn = (
            crystalnn_results.get(
                site_index
            )
        )


        try:

            site_chemenv = (
                light_environments
                .coordination_environments[
                    site_index
                ]
            )

        except Exception:

            site_chemenv = None


        # -------------------------------------------------
        # NO CHEMENV ASSIGNMENT
        # -------------------------------------------------

        if not site_chemenv:

            unresolved_sites += 1

            site_results.append(
                {
                    "site_index":
                        site_index,

                    "element":
                        element,

                    "crystalnn_cn":
                        crystalnn_cn,

                    "ce_symbol":
                        None,

                    "geometry":
                        "unresolved",

                    "csm":
                        None,
                }
            )

            continue


        # -------------------------------------------------
        # PRIMARY CHEMENV ENVIRONMENT
        # -------------------------------------------------

        primary = (
            site_chemenv[0]
        )

        ce_symbol = (
            primary.get(
                "ce_symbol"
            )
        )

        csm = (
            primary.get(
                "csm"
            )
        )


        # -------------------------------------------------
        # ALTERNATE CSM LOCATION
        # -------------------------------------------------

        if csm is None:

            ce_dict = (
                primary.get(
                    "ce_dict"
                )
            )

            if isinstance(
                ce_dict,
                dict,
            ):

                csm = (
                    ce_dict.get(
                        "csm"
                    )
                )


        # -------------------------------------------------
        # NORMALIZE CSM
        # -------------------------------------------------

        if csm is not None:

            try:

                csm = float(
                    csm
                )

            except Exception:

                csm = None


        # -------------------------------------------------
        # GEOMETRY LABEL
        # -------------------------------------------------

        if ce_symbol:

            geometry = (
                ce_symbol
            )

        else:

            geometry = (
                "unresolved"
            )


        # -------------------------------------------------
        # SUMMARY KEY
        # -------------------------------------------------

        environment_key = (
            element,
            geometry,
        )


        environment_counter[
            environment_key
        ] += 1


        if csm is not None:

            csm_by_environment[
                environment_key
            ].append(
                csm
            )


        # -------------------------------------------------
        # SITE RESULT
        # -------------------------------------------------

        site_results.append(
            {
                "site_index":
                    site_index,

                "element":
                    element,

                "crystalnn_cn":
                    crystalnn_cn,

                "ce_symbol":
                    ce_symbol,

                "geometry":
                    geometry,

                "csm":
                    csm,
            }
        )


    # =====================================================
    # BUILD MOTIF SUMMARY
    # =====================================================

    motif_summary = []


    for (
        element,
        geometry,
    ), count in (
        environment_counter
        .most_common()
    ):

        csm_values = (
            csm_by_environment.get(
                (
                    element,
                    geometry,
                ),
                [],
            )
        )


        if csm_values:

            mean_csm = (
                sum(csm_values)
                / len(csm_values)
            )

            min_csm = min(
                csm_values
            )

            max_csm = max(
                csm_values
            )

        else:

            mean_csm = None

            min_csm = None

            max_csm = None


        motif_summary.append(
            {
                "element":
                    element,

                "geometry":
                    geometry,

                "site_count":
                    count,

                "mean_csm":
                    mean_csm,

                "min_csm":
                    min_csm,

                "max_csm":
                    max_csm,
            }
        )


    # =====================================================
    # LATTICE
    # =====================================================

    lattice = (
        raw_structure.lattice
    )


    # =====================================================
    # FINAL RESULT
    # =====================================================

    result = {
        "material_id":
            str(material_id),

        "formula":
            raw_structure
            .composition
            .reduced_formula,

        "num_sites":
            len(raw_structure),

        "lattice_parameters": {

            "a":
                float(
                    lattice.a
                ),

            "b":
                float(
                    lattice.b
                ),

            "c":
                float(
                    lattice.c
                ),

            "alpha":
                float(
                    lattice.alpha
                ),

            "beta":
                float(
                    lattice.beta
                ),

            "gamma":
                float(
                    lattice.gamma
                ),
        },

        "oxidation_states_assigned":
            oxidation_states_assigned,

        "motif_summary":
            motif_summary,

        "site_environments":
            site_results,

        "unresolved_sites":
            unresolved_sites,

        "methods": [
            "pymatgen CrystalNN",
            "pymatgen ChemEnv",
        ],

        "structure_source":
            "Materials Project API (live)",

        "analysis_source":
            "MotifLens live motif analysis",

        "cache_status":
            "MISS",
    }


    # =====================================================
    # SAVE CACHE
    # =====================================================

    set_cached_motif(
        material_id,
        result,
    )


    # =====================================================
    # RETURN RESULT
    # =====================================================

    return result