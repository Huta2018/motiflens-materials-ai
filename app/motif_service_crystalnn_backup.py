import os
from collections import Counter

from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.analysis.local_env import CrystalNN


load_dotenv()


# =========================================================
# MATERIALS PROJECT CLIENT
# =========================================================

def get_mp_client():
    api_key = os.getenv("MP_API_KEY")

    if not api_key:
        raise RuntimeError(
            "MP_API_KEY is not configured."
        )

    return MPRester(api_key)


# =========================================================
# GET STRUCTURE
# =========================================================

def get_material_structure(material_id):
    """
    Retrieve the final crystal structure for a Materials
    Project material ID.
    """

    with get_mp_client() as mpr:
        structure = mpr.get_structure_by_material_id(
            material_id,
            final=True,
        )

    return structure


# =========================================================
# COORDINATION LABEL
# =========================================================

def coordination_label(coordination_number):
    """
    Convert common coordination numbers into simple
    human-readable local-environment labels.

    These are coordination-based descriptors, not rigorous
    geometry assignments.
    """

    labels = {
        1: "single-coordinate",
        2: "two-coordinate",
        3: "three-coordinate",
        4: "four-coordinate",
        5: "five-coordinate",
        6: "six-coordinate",
        7: "seven-coordinate",
        8: "eight-coordinate",
        9: "nine-coordinate",
        10: "ten-coordinate",
        11: "eleven-coordinate",
        12: "twelve-coordinate",
    }

    return labels.get(
        coordination_number,
        f"{coordination_number}-coordinate",
    )


# =========================================================
# ANALYZE LOCAL ENVIRONMENTS
# =========================================================

def analyze_material_motifs(material_id):
    """
    Retrieve a Materials Project crystal structure and analyze
    local coordination environments using CrystalNN.

    Returns site-level coordination information plus a summary.

    Important:
    The first implementation reports coordination environments.
    It does not claim exact polyhedral geometry solely from
    coordination number.
    """

    structure = get_material_structure(
        material_id
    )

    cnn = CrystalNN()

    site_results = []

    motif_counter = Counter()

    failed_sites = 0


    for site_index, site in enumerate(structure):

        element = str(
            site.specie
        )

        try:

            cn = cnn.get_cn(
                structure,
                site_index,
                use_weights=False,
            )

            cn = int(
                round(cn)
            )

            label = coordination_label(
                cn
            )

            motif_key = (
                f"{element}:{label}"
            )

            motif_counter[
                motif_key
            ] += 1


            site_results.append(
                {
                    "site_index":
                        site_index,

                    "element":
                        element,

                    "coordination_number":
                        cn,

                    "coordination_environment":
                        label,
                }
            )


        except Exception as exc:

            failed_sites += 1

            site_results.append(
                {
                    "site_index":
                        site_index,

                    "element":
                        element,

                    "coordination_number":
                        None,

                    "coordination_environment":
                        "unresolved",

                    "error":
                        str(exc),
                }
            )


    motif_summary = []

    for motif, count in motif_counter.most_common():

        element, environment = motif.split(
            ":",
            1,
        )

        motif_summary.append(
            {
                "element":
                    element,

                "coordination_environment":
                    environment,

                "site_count":
                    count,
            }
        )


    result = {
        "material_id":
            str(material_id),

        "formula":
            structure.composition.reduced_formula,

        "num_sites":
            len(structure),

        "lattice_parameters": {
            "a":
                float(structure.lattice.a),

            "b":
                float(structure.lattice.b),

            "c":
                float(structure.lattice.c),

            "alpha":
                float(structure.lattice.alpha),

            "beta":
                float(structure.lattice.beta),

            "gamma":
                float(structure.lattice.gamma),
        },

        "motif_summary":
            motif_summary,

        "site_environments":
            site_results,

        "failed_sites":
            failed_sites,

        "method":
            "pymatgen CrystalNN",

        "data_source":
            "Materials Project crystal structure (live)",
    }

    return result