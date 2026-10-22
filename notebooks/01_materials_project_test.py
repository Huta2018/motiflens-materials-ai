# Databricks notebook source
# DBTITLE 1,Install materials project packages
#installing dependencies
#%pip install mp-api pymatgen

# COMMAND ----------

import getpass

MP_API_KEY = getpass.getpass("Materials Project API key: ")

# COMMAND ----------

#test MP connection
from mp_api.client import MPRester

with MPRester(MP_API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        formula="BiVO4",
        fields=[
            "material_id",
            "formula_pretty",
            "band_gap",
            "energy_above_hull",
            "structure"
        ]
    )

print(f"Found {len(docs)} structures")

for d in docs[:5]:
    print(
        d.material_id,
        d.formula_pretty,
        d.band_gap,
        d.energy_above_hull
    )

# COMMAND ----------

#now extract all local environments
doc = next(d for d in docs if str(d.material_id) == "mp-23044")
structure = doc.structure

print(structure)

# COMMAND ----------

#run ChemEnv
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
    LocalGeometryFinder,
)

lgf = LocalGeometryFinder()

coord_envs = lgf.compute_coordination_environments(
    structure,
    only_cations=True
)

print(type(coord_envs))
print(coord_envs)

# COMMAND ----------

#To get the deterministic motif we will inspect the cation site and choose the candidate with the lowest SAM
motif_rows = []

for site_index, environments in enumerate(coord_envs):

    # Skip sites ChemEnv didn't process
    if not environments:
        continue

    # Lowest CSM = closest ideal coordination geometry
    best_env = min(
        environments,
        key=lambda env: env["csm"]
    )

    site = structure[site_index]

    motif_rows.append({
        "material_id": "mp-23044",
        "site_index": site_index,
        "element": site.species_string,
        "motif": best_env["ce_symbol"],
        "csm": float(best_env["csm"]),
    })

motif_rows

# COMMAND ----------

import pandas as pd

motif_df = pd.DataFrame(motif_rows)

display(motif_df)

# COMMAND ----------

#function to generate motifs
def extract_motifs(material_id, structure, coord_envs):
    rows = []

    for site_index, environments in enumerate(coord_envs):
        if not environments:
            continue

        best_env = min(environments, key=lambda env: env["csm"])
        site = structure[site_index]

        rows.append({
            "material_id": material_id,
            "site_index": site_index,
            "element": site.species_string,
            "motif": best_env["ce_symbol"],
            "csm": float(best_env["csm"]),
        })

    return rows

# COMMAND ----------

from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
    LocalGeometryFinder,
)

all_motif_rows = []

for d in docs:
    structure = d.structure

    lgf = LocalGeometryFinder()

    coord_envs = lgf.compute_coordination_environments(
        structure,
        only_cations=True
    )

    rows = extract_motifs(
        str(d.material_id),
        structure,
        coord_envs
    )

    all_motif_rows.extend(rows)

# COMMAND ----------

motifs_sdf = spark.createDataFrame(all_motif_rows)

display(motifs_sdf)

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW SCHEMAS IN bootcamp_students;

# COMMAND ----------

# Save the first motif dataset in my bootcamp Unity Catalog schema.
# This is the raw ChemEnv output that later Spark transformations will clean and aggregate.
motifs_sdf.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(
        "bootcamp_students.banjadehuta.bronze_material_motifs"
    )

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT *
# MAGIC FROM bootcamp_students.banjadehuta.bronze_material_motifs;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC DESCRIBE DETAIL
# MAGIC bootcamp_students.banjadehuta.bronze_material_motifs;

# COMMAND ----------

#build real oxide dataset
from mp_api.client import MPRester

# Start with a small oxide sample so we can benchmark the complete
# structure-to-motif pipeline before increasing the dataset size.
with MPRester(MP_API_KEY) as mpr:
    oxide_docs = mpr.materials.summary.search(
        elements=["O"],
        fields=[
            "material_id",
            "formula_pretty",
            "structure",
            "band_gap",
            "formation_energy_per_atom",
            "energy_above_hull",
            "density",
            "symmetry",
        ],
        num_chunks=1,
        chunk_size=100,
    )

print(f"Retrieved {len(oxide_docs)} materials")

# COMMAND ----------

# Quick sanity check on the materials returned by the API.
for material in oxide_docs[:10]:
    print(
        material.material_id,
        material.formula_pretty,
        material.band_gap,
        material.energy_above_hull,
        len(material.structure),
    )

# COMMAND ----------

import time
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
    LocalGeometryFinder,
)

# Benchmark ChemEnv on the 100-material sample before scaling the pipeline.
# We keep failures instead of stopping the batch so problematic structures
# can be inspected separately.
all_motif_rows = []
failed_materials = []
total_sites = 0

start_time = time.time()

for material in oxide_docs:
    material_id = str(material.material_id)
    structure = material.structure
    total_sites += len(structure)

    try:
        lgf = LocalGeometryFinder()

        coord_envs = lgf.compute_coordination_environments(
            structure,
            only_cations=True
        )

        motif_rows = extract_motifs(
            material_id,
            structure,
            coord_envs
        )

        all_motif_rows.extend(motif_rows)

    except Exception as exc:
        # Keep enough information to investigate failures after the batch.
        failed_materials.append({
            "material_id": material_id,
            "error": str(exc)
        })

elapsed = time.time() - start_time

# COMMAND ----------

# These numbers will tell us whether ChemEnv is practical to scale
# and how quickly the structural records grow relative to materials.
successful = len(oxide_docs) - len(failed_materials)

print(f"Materials processed : {len(oxide_docs)}")
print(f"Successful          : {successful}")
print(f"Failed              : {len(failed_materials)}")
print(f"Atomic sites        : {total_sites}")
print(f"Motif records       : {len(all_motif_rows)}")
print(f"Total runtime       : {elapsed:.2f} sec")
print(f"Runtime/material    : {elapsed / len(oxide_docs):.3f} sec")

# COMMAND ----------

# Look for obviously suspicious assignments before scaling further.
motif_100_df = spark.createDataFrame(all_motif_rows)

display(
    motif_100_df
        .groupBy("element", "motif")
        .count()
        .orderBy("count", ascending=False)
)

# COMMAND ----------

# Build atom-neighbor relationships for each crystal.
# These records preserve local structural connectivity and will also
# provide the high-volume dataset used by the Spark pipeline.
def extract_neighbors(material_id, structure, cutoff=5.0):
    neighbor_rows = []

    for center_index, center_site in enumerate(structure):

        # pymatgen handles periodic boundary conditions when finding neighbors.
        neighbors = structure.get_neighbors(
            center_site,
            r=cutoff
        )

        for neighbor in neighbors:
            neighbor_rows.append({
                "material_id": material_id,
                "center_site": center_index,
                "center_element": center_site.species_string,
                "neighbor_element": neighbor.species_string,
                "distance": float(neighbor.nn_distance),
            })

    return neighbor_rows

# COMMAND ----------

import time

# Benchmark the neighbor representation on the same sample used for ChemEnv
# so we can estimate how quickly the structural dataset will scale.
all_neighbor_rows = []

start_time = time.time()

for material in oxide_docs:
    rows = extract_neighbors(
        str(material.material_id),
        material.structure,
        cutoff=5.0
    )

    all_neighbor_rows.extend(rows)

neighbor_elapsed = time.time() - start_time

print(f"Materials processed : {len(oxide_docs)}")
print(f"Neighbor records    : {len(all_neighbor_rows)}")
print(f"Total runtime       : {neighbor_elapsed:.2f} sec")
print(
    f"Neighbors/material  : "
    f"{len(all_neighbor_rows) / len(oxide_docs):.1f}"
)

# COMMAND ----------

# Move the periodic atom-neighbor relationships into Spark.
# Later we can derive coordination statistics and structural features from this table.
neighbors_sdf = spark.createDataFrame(all_neighbor_rows)

display(neighbors_sdf)

# COMMAND ----------

# Sanity-check the structural relationships before persisting them.
display(
    neighbors_sdf
        .groupBy("center_element", "neighbor_element")
        .count()
        .orderBy("count", ascending=False)
)

# COMMAND ----------

# Keep the MVP focused on inorganic oxides rather than every
# oxygen-containing compound returned by the API.
excluded_elements = {"F", "Cl", "Br", "I", "C", "N", "H"}

oxide_docs_clean = []

for material in oxide_docs:
    elements = {str(el) for el in material.structure.composition.elements}

    if "O" in elements and not elements.intersection(excluded_elements):
        oxide_docs_clean.append(material)

print(f"Original sample : {len(oxide_docs)}")
print(f"Oxide subset    : {len(oxide_docs_clean)}")

# COMMAND ----------

# Inspect the filtered materials before rebuilding the structural tables.
for material in oxide_docs_clean[:20]:
    print(material.material_id, material.formula_pretty)

# COMMAND ----------

from pymatgen.analysis.local_env import CrystalNN

# CrystalNN gives us a chemically informed local-neighbor definition
# instead of treating every atom inside a fixed radius as connected.
cnn = CrystalNN()

# COMMAND ----------

def extract_crystalnn_neighbors(material_id, structure):
    neighbor_rows = []

    for center_index, center_site in enumerate(structure):
        try:
            neighbors = cnn.get_nn_info(structure, center_index)

            for neighbor in neighbors:
                neighbor_site = neighbor["site"]

                neighbor_rows.append({
                    "material_id": material_id,
                    "center_site": center_index,
                    "center_element": center_site.species_string,
                    "neighbor_element": neighbor_site.species_string,
                    "distance": float(
                        center_site.distance(neighbor_site)
                    ),
                    "weight": float(neighbor["weight"]),
                })

        except Exception:
            # A single unusual site should not stop the entire batch.
            continue

    return neighbor_rows

# COMMAND ----------

# Build the first chemically informed neighbor dataset.
clean_neighbor_rows = []

for material in oxide_docs_clean:
    rows = extract_crystalnn_neighbors(
        str(material.material_id),
        material.structure
    )

    clean_neighbor_rows.extend(rows)

print(f"Materials        : {len(oxide_docs_clean)}")
print(f"Neighbor records : {len(clean_neighbor_rows)}")
print(
    f"Records/material : "
    f"{len(clean_neighbor_rows) / len(oxide_docs_clean):.1f}"
)

# COMMAND ----------

# Move the neighbor relationships into Spark so we can profile
# coordination patterns across the oxide dataset.
clean_neighbors_sdf = spark.createDataFrame(clean_neighbor_rows)

display(
    clean_neighbors_sdf
        .groupBy("center_element", "neighbor_element")
        .count()
        .orderBy("count", ascending=False)
)

# COMMAND ----------

from pymatgen.core import Element

# Restrict the MVP to metal oxides:
# oxygen must be present and every other element must be metallic.
# This removes borates, sulfates, halides, organics, and similar systems.
metal_oxide_docs = []

for material in oxide_docs:
    elements = [
        str(el)
        for el in material.structure.composition.elements
    ]

    non_oxygen = [el for el in elements if el != "O"]

    if (
        "O" in elements
        and non_oxygen
        and all(Element(el).is_metal for el in non_oxygen)
    ):
        metal_oxide_docs.append(material)

print(f"Original sample      : {len(oxide_docs)}")
print(f"Metal oxide subset   : {len(metal_oxide_docs)}")

# COMMAND ----------

# Make sure the operational definition produces the material family
# we actually want to use in the discovery application.
for material in metal_oxide_docs[:30]:
    print(material.material_id, material.formula_pretty)

# COMMAND ----------

# Rebuild the neighbor dataset using only the metal-oxide subset.
metal_oxide_neighbor_rows = []

for material in metal_oxide_docs:
    rows = extract_crystalnn_neighbors(
        str(material.material_id),
        material.structure
    )

    metal_oxide_neighbor_rows.extend(rows)

print(f"Materials        : {len(metal_oxide_docs)}")
print(f"Neighbor records : {len(metal_oxide_neighbor_rows)}")
print(
    f"Records/material : "
    f"{len(metal_oxide_neighbor_rows) / len(metal_oxide_docs):.1f}"
)

# COMMAND ----------

# Profile the dominant local element pairs after tightening
# the chemistry scope.
metal_oxide_neighbors_sdf = spark.createDataFrame(
    metal_oxide_neighbor_rows
)

display(
    metal_oxide_neighbors_sdf
        .groupBy("center_element", "neighbor_element")
        .count()
        .orderBy("count", ascending=False)
)

# COMMAND ----------

from pymatgen.core import Element

# Final MVP definition:
# - oxygen-containing
# - all non-O elements are metals
# - binary or ternary chemistry
# - enough oxygen to avoid strongly metal-rich/intermetallic compounds
metal_oxide_docs_final = []

for material in oxide_docs:
    composition = material.structure.composition
    elements = [str(el) for el in composition.elements]

    non_oxygen = [el for el in elements if el != "O"]

    # Atomic fraction gives a simple way to remove oxygen-poor compounds
    # such as metal-rich suboxides that are outside the initial project scope.
    oxygen_fraction = composition.get_atomic_fraction(Element("O"))

    if (
        "O" in elements
        and non_oxygen
        and all(Element(el).is_metal for el in non_oxygen)
        and 2 <= len(elements) <= 3
        and oxygen_fraction >= 0.30
    ):
        metal_oxide_docs_final.append(material)

print(f"Original API sample  : {len(oxide_docs)}")
print(f"Final oxide subset   : {len(metal_oxide_docs_final)}")

# COMMAND ----------

# Final sanity check before we freeze the chemistry definition
# and start building persistent Bronze tables.
for material in metal_oxide_docs_final[:30]:
    composition = material.structure.composition

    print(
        material.material_id,
        material.formula_pretty,
        f"O fraction = {composition.get_atomic_fraction(Element('O')):.2f}"
    )

# COMMAND ----------

import json
from datetime import datetime, timezone

# Build the raw material-level dataset that all later processing
# will use as the canonical source for this batch.
bronze_material_rows = []

ingested_at = datetime.now(timezone.utc)

for material in metal_oxide_docs_final:
    structure = material.structure

    bronze_material_rows.append({
        "material_id": str(material.material_id),
        "formula": material.formula_pretty,
        "num_sites": len(structure),

        # Keep the physical properties we plan to expose in the discovery UI.
        "band_gap": (
            float(material.band_gap)
            if material.band_gap is not None else None
        ),
        "formation_energy_per_atom": (
            float(material.formation_energy_per_atom)
            if material.formation_energy_per_atom is not None else None
        ),
        "energy_above_hull": (
            float(material.energy_above_hull)
            if material.energy_above_hull is not None else None
        ),
        "density": (
            float(material.density)
            if material.density is not None else None
        ),

        # Preserve the full crystal structure so we can reconstruct it
        # later without calling the Materials Project API again.
        "structure_json": json.dumps(structure.as_dict()),

        # Basic lineage fields for reproducibility and future incremental loads.
        "source": "Materials Project",
        "ingested_at": ingested_at,
    })

# COMMAND ----------

# Confirm that the source list still contains the filtered materials
# and that the Bronze record list was actually populated.
print("Filtered materials:", len(metal_oxide_docs_final))
print("Bronze rows:", len(bronze_material_rows))

# COMMAND ----------

# Reload a small source batch from Materials Project.
with MPRester(MP_API_KEY) as mpr:
    oxide_docs = mpr.materials.summary.search(
        elements=["O"],
        fields=[
            "material_id",
            "formula_pretty",
            "structure",
            "band_gap",
            "formation_energy_per_atom",
            "energy_above_hull",
            "density",
            "symmetry",
        ],
        num_chunks=1,
        chunk_size=100,
    )

print(f"Retrieved {len(oxide_docs)} materials")

# COMMAND ----------

from pymatgen.core import Element

# Keep the MVP focused on binary/ternary metal oxides
# with a reasonable oxygen fraction.
metal_oxide_docs_final = []

for material in oxide_docs:
    composition = material.structure.composition
    elements = [str(el) for el in composition.elements]
    non_oxygen = [el for el in elements if el != "O"]

    oxygen_fraction = composition.get_atomic_fraction(Element("O"))

    if (
        "O" in elements
        and non_oxygen
        and all(Element(el).is_metal for el in non_oxygen)
        and 2 <= len(elements) <= 3
        and oxygen_fraction >= 0.30
    ):
        metal_oxide_docs_final.append(material)

print(f"Final oxide subset: {len(metal_oxide_docs_final)}")

# COMMAND ----------

print(len(bronze_material_rows))

# COMMAND ----------

import json
from datetime import datetime, timezone

# Rebuild the Bronze material records from the final filtered oxide set.
bronze_material_rows = []

ingested_at = datetime.now(timezone.utc)

for material in metal_oxide_docs_final:
    structure = material.structure

    bronze_material_rows.append({
        "material_id": str(material.material_id),
        "formula": material.formula_pretty,
        "num_sites": len(structure),
        "band_gap": float(material.band_gap) if material.band_gap is not None else None,
        "formation_energy_per_atom": (
            float(material.formation_energy_per_atom)
            if material.formation_energy_per_atom is not None else None
        ),
        "energy_above_hull": (
            float(material.energy_above_hull)
            if material.energy_above_hull is not None else None
        ),
        "density": float(material.density) if material.density is not None else None,

        # Keep the full structure so later jobs can reconstruct it without
        # making another Materials Project API call.
        "structure_json": json.dumps(structure.as_dict()),

        # Simple provenance fields for reproducibility.
        "source": "Materials Project",
        "ingested_at": ingested_at,
    })

print(f"Bronze rows built: {len(bronze_material_rows)}")

# COMMAND ----------

# Convert the material-level records into a Spark DataFrame.
bronze_materials_sdf = spark.createDataFrame(bronze_material_rows)

display(bronze_materials_sdf)

# COMMAND ----------

# Persist the raw material-level dataset in Unity Catalog.
# This becomes the stable source for downstream motif and neighbor processing.
bronze_materials_sdf.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(
        "bootcamp_students.banjadehuta.bronze_materials"
    )

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     material_id,
# MAGIC     formula,
# MAGIC     num_sites,
# MAGIC     band_gap,
# MAGIC     formation_energy_per_atom,
# MAGIC     energy_above_hull,
# MAGIC     density
# MAGIC FROM bootcamp_students.banjadehuta.bronze_materials
# MAGIC ORDER BY formula;

# COMMAND ----------

from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
    LocalGeometryFinder,
)

# Build site-level motif records for the final oxide subset.
# For the MVP, each cation site is assigned the ChemEnv geometry
# with the lowest continuous symmetry measure (CSM).
bronze_motif_rows = []

for material in metal_oxide_docs_final:
    material_id = str(material.material_id)
    structure = material.structure

    lgf = LocalGeometryFinder()

    coord_envs = lgf.compute_coordination_environments(
        structure,
        only_cations=True
    )

    for site_index, environments in enumerate(coord_envs):

        # Some sites may not return a valid coordination environment.
        if not environments:
            continue

        best_env = min(
            environments,
            key=lambda env: env["csm"]
        )

        site = structure[site_index]

        bronze_motif_rows.append({
            "material_id": material_id,
            "site_index": site_index,
            "element": site.species_string,
            "motif": best_env["ce_symbol"],
            "csm": float(best_env["csm"]),
        })

print(f"Motif records built: {len(bronze_motif_rows)}")

# COMMAND ----------

# Move the site-level motif assignments into Spark.
bronze_motifs_sdf = spark.createDataFrame(bronze_motif_rows)

display(bronze_motifs_sdf)

# COMMAND ----------

# Persist the motif-level Bronze dataset for downstream aggregation
# and material-similarity calculations.
bronze_motifs_sdf.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(
        "bootcamp_students.banjadehuta.bronze_material_motifs"
    )

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     material_id,
# MAGIC     site_index,
# MAGIC     element,
# MAGIC     motif,
# MAGIC     csm
# MAGIC FROM bootcamp_students.banjadehuta.bronze_material_motifs
# MAGIC ORDER BY material_id, site_index;

# COMMAND ----------

from pymatgen.analysis.local_env import CrystalNN

cnn = CrystalNN()

def extract_crystalnn_neighbors(material_id, structure):
    neighbor_rows = []

    for center_index, center_site in enumerate(structure):
        try:
            neighbors = cnn.get_nn_info(structure, center_index)

            for neighbor in neighbors:
                neighbor_site = neighbor["site"]

                neighbor_rows.append({
                    "material_id": material_id,
                    "center_site": center_index,
                    "center_element": center_site.species_string,
                    "neighbor_element": neighbor_site.species_string,
                    "distance": float(center_site.distance(neighbor_site)),
                    "weight": float(neighbor["weight"]),
                })

        except Exception:
            # Keep unusual sites from stopping the whole batch.
            continue

    return neighbor_rows

# COMMAND ----------

# Build local neighbor relationships for the same filtered oxide set
# used in the materials and motif Bronze tables.
bronze_neighbor_rows = []

for material in metal_oxide_docs_final:
    rows = extract_crystalnn_neighbors(
        str(material.material_id),
        material.structure
    )

    bronze_neighbor_rows.extend(rows)

print(f"Neighbor records built: {len(bronze_neighbor_rows)}")

# COMMAND ----------

# Move the neighbor-level records into Spark.
bronze_neighbors_sdf = spark.createDataFrame(bronze_neighbor_rows)

display(bronze_neighbors_sdf)

# COMMAND ----------

# Persist the local structural relationships as the third Bronze Delta table.
bronze_neighbors_sdf.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(
        "bootcamp_students.banjadehuta.bronze_neighbor_relationships"
    )

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT *
# MAGIC FROM bootcamp_students.banjadehuta.bronze_neighbor_relationships
# MAGIC ORDER BY material_id, center_site
# MAGIC LIMIT 100;

# COMMAND ----------

from pyspark.sql import functions as F

# Read from Delta rather than relying on notebook variables.
# From this point onward, Bronze is our persistent source of truth.
bronze_motifs = spark.table(
    "bootcamp_students.banjadehuta.bronze_material_motifs"
)

display(bronze_motifs)

# COMMAND ----------

# Collapse repeated site-level motif assignments into one
# material-motif record that is easier to use for discovery.
silver_material_motifs = (
    bronze_motifs
    .groupBy(
        "material_id",
        "element",
        "motif"
    )
    .agg(
        F.count("*").alias("motif_count"),
        F.avg("csm").alias("mean_csm"),
        F.min("csm").alias("min_csm"),
        F.max("csm").alias("max_csm")
    )
)

display(silver_material_motifs)

# COMMAND ----------

# Silver contains cleaned and aggregated structural information
# derived from the raw site-level Bronze records.
silver_material_motifs.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(
        "bootcamp_students.banjadehuta.silver_material_motifs"
    )

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT *
# MAGIC FROM bootcamp_students.banjadehuta.silver_material_motifs
# MAGIC ORDER BY material_id, motif;

# COMMAND ----------

from pyspark.sql import functions as F

# Load persisted tables rather than relying on notebook variables.
materials = spark.table(
    "bootcamp_students.banjadehuta.bronze_materials"
)

motifs = spark.table(
    "bootcamp_students.banjadehuta.silver_material_motifs"
)

# COMMAND ----------

# Build one structural profile per material.
# Each motif is kept as a compact string so the profile remains
# easy to inspect and can later be exposed through the application.
motif_profiles = (
    motifs
    .groupBy("material_id")
    .agg(
        F.collect_set(
            F.concat_ws(
                ":",
                F.col("element"),
                F.col("motif")
            )
        ).alias("motifs"),

        F.sum("motif_count").alias("total_motif_sites"),

        F.avg("mean_csm").alias("mean_motif_csm")
    )
)

display(motif_profiles)

# COMMAND ----------

# Combine physical properties with the structural motif representation.
# This becomes the main material profile used by the discovery layer.
silver_material_profiles = (
    materials
    .join(
        motif_profiles,
        on="material_id",
        how="left"
    )
    .select(
        "material_id",
        "formula",
        "num_sites",
        "band_gap",
        "formation_energy_per_atom",
        "energy_above_hull",
        "density",
        "motifs",
        "total_motif_sites",
        "mean_motif_csm"
    )
)

display(silver_material_profiles)

# COMMAND ----------

# Persist the application-ready material profile.
silver_material_profiles.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(
        "bootcamp_students.banjadehuta.silver_material_profiles"
    )

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     material_id,
# MAGIC     formula,
# MAGIC     band_gap,
# MAGIC     energy_above_hull,
# MAGIC     motifs,
# MAGIC     total_motif_sites
# MAGIC FROM bootcamp_students.banjadehuta.silver_material_profiles
# MAGIC ORDER BY formula;

# COMMAND ----------

from pyspark.sql import functions as F

# Use the normalized motif table for similarity calculations.
# Keeping motifs as rows makes shared-environment joins straightforward.
motifs = spark.table(
    "bootcamp_students.banjadehuta.silver_material_motifs"
)

materials = spark.table(
    "bootcamp_students.banjadehuta.silver_material_profiles"
)

# COMMAND ----------

# A motif key combines the coordinating element and ChemEnv geometry.
# This prevents, for example, Fe:O:6 and another element's O:6
# environment from being treated as identical.
motif_index = (
    motifs
    .withColumn(
        "motif_key",
        F.concat_ws("|", "element", "motif")
    )
    .select(
        "material_id",
        "motif_key",
        "motif_count",
        "mean_csm"
    )
)

display(motif_index)

# COMMAND ----------

# Connect pairs of materials that share at least one structural motif.
# We keep source_id < candidate_id so the same pair is not counted twice.
left = motif_index.alias("a")
right = motif_index.alias("b")

shared_motifs = (
    left.join(
        right,
        (F.col("a.motif_key") == F.col("b.motif_key"))
        & (F.col("a.material_id") < F.col("b.material_id"))
    )
    .select(
        F.col("a.material_id").alias("source_id"),
        F.col("b.material_id").alias("candidate_id"),
        F.col("a.motif_key").alias("shared_motif"),
        F.col("a.mean_csm").alias("source_csm"),
        F.col("b.mean_csm").alias("candidate_csm")
    )
)

display(shared_motifs)

# COMMAND ----------

# MVP similarity is the number of distinct structural motifs
# shared by two materials. We will add rarity and CSM weighting later.
material_similarity = (
    shared_motifs
    .groupBy("source_id", "candidate_id")
    .agg(
        F.countDistinct("shared_motif").alias("shared_motif_count"),
        F.collect_set("shared_motif").alias("shared_motifs"),
        F.avg(
            F.abs(F.col("source_csm") - F.col("candidate_csm"))
        ).alias("mean_csm_difference")
    )
)

display(
    material_similarity
    .orderBy(
        F.desc("shared_motif_count"),
        F.asc("mean_csm_difference")
    )
)

# COMMAND ----------

# Persist the first material-to-material similarity results.
material_similarity.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(
        "bootcamp_students.banjadehuta.gold_material_similarity"
    )

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT *
# MAGIC FROM bootcamp_students.banjadehuta.gold_material_similarity
# MAGIC ORDER BY shared_motif_count DESC, mean_csm_difference ASC;

# COMMAND ----------

import getpass
from mp_api.client import MPRester

# Prompt for the API key without showing or storing it in the notebook.
MP_API_KEY = getpass.getpass("Materials Project API key: ")

# Pull the data in smaller API pages to avoid oversized requests.
with MPRester(MP_API_KEY) as mpr:
    oxide_docs_large = mpr.materials.summary.search(
        elements=["O"],
        fields=[
            "material_id",
            "formula_pretty",
            "structure",
            "band_gap",
            "formation_energy_per_atom",
            "energy_above_hull",
            "density",
            "symmetry",
        ],
        num_chunks=10,
        chunk_size=200,
    )

print(f"Retrieved {len(oxide_docs_large)} materials")

# COMMAND ----------

import json
from datetime import datetime, timezone

# Convert the Materials Project response into a Spark-friendly format.
# This raw Bronze table becomes our persistent source instead of the API.
raw_material_rows = []

ingested_at = datetime.now(timezone.utc)

for material in oxide_docs_large:
    structure = material.structure

    raw_material_rows.append({
        "material_id": str(material.material_id),
        "formula": material.formula_pretty,
        "num_sites": len(structure),

        "band_gap": (
            float(material.band_gap)
            if material.band_gap is not None else None
        ),

        "formation_energy_per_atom": (
            float(material.formation_energy_per_atom)
            if material.formation_energy_per_atom is not None else None
        ),

        "energy_above_hull": (
            float(material.energy_above_hull)
            if material.energy_above_hull is not None else None
        ),

        "density": (
            float(material.density)
            if material.density is not None else None
        ),

        # Preserve the full structure so later processing can be done
        # entirely from Delta without calling Materials Project again.
        "structure_json": json.dumps(structure.as_dict()),

        "source": "Materials Project",
        "ingested_at": ingested_at,
    })

print(f"Raw records prepared: {len(raw_material_rows)}")

# COMMAND ----------

# Move the API response into Spark before persisting it.
raw_materials_sdf = spark.createDataFrame(raw_material_rows)

print(f"Spark rows: {raw_materials_sdf.count()}")
display(raw_materials_sdf.limit(10))

# COMMAND ----------

# Persist the larger unfiltered API batch.
# Future filtering and processing should start from this table.
raw_materials_sdf.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(
        "bootcamp_students.banjadehuta.bronze_materials_raw"
    )

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT COUNT(*) AS total_records
# MAGIC FROM bootcamp_students.banjadehuta.bronze_materials_raw;

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType
from pymatgen.core import Structure, Element
import json

# Apply the MVP chemistry definition directly to persisted Bronze records.
# This keeps filtering reproducible even after the notebook session restarts.
def is_target_metal_oxide(structure_json):
    try:
        structure = Structure.from_dict(json.loads(structure_json))
        composition = structure.composition

        elements = [str(el) for el in composition.elements]
        non_oxygen = [el for el in elements if el != "O"]

        if "O" not in elements or not non_oxygen:
            return False

        oxygen_fraction = composition.get_atomic_fraction(Element("O"))

        return (
            all(Element(el).is_metal for el in non_oxygen)
            and 2 <= len(elements) <= 3
            and oxygen_fraction >= 0.30
        )

    except Exception:
        return False


is_target_metal_oxide_udf = F.udf(
    is_target_metal_oxide,
    BooleanType()
)

# COMMAND ----------

# Start downstream processing from the persisted raw Bronze table,
# not from in-memory API objects.
bronze_raw = spark.table(
    "bootcamp_students.banjadehuta.bronze_materials_raw"
)

silver_filtered_materials = (
    bronze_raw
    .withColumn(
        "is_target_oxide",
        is_target_metal_oxide_udf("structure_json")
    )
    .filter(F.col("is_target_oxide"))
    .drop("is_target_oxide")
)

print("Raw materials:", bronze_raw.count())
print("Filtered oxides:", silver_filtered_materials.count())

display(
    silver_filtered_materials.select(
        "material_id",
        "formula",
        "num_sites",
        "band_gap",
        "energy_above_hull"
    ).limit(30)
)

# COMMAND ----------

# Persist the chemistry-filtered material set for motif and neighbor extraction.
silver_filtered_materials.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(
        "bootcamp_students.banjadehuta.silver_filtered_materials"
    )

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT COUNT(*) AS filtered_materials
# MAGIC FROM bootcamp_students.banjadehuta.silver_filtered_materials;

# COMMAND ----------

import json
from pymatgen.core import Structure
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
    LocalGeometryFinder,
)

# Read the filtered materials from Delta so this step is independent
# of the original Materials Project API session.
filtered_rows = (
    spark.table(
        "bootcamp_students.banjadehuta.silver_filtered_materials"
    )
    .select("material_id", "structure_json")
    .collect()
)

print(f"Materials to process: {len(filtered_rows)}")

# COMMAND ----------

from pyspark.sql import functions as F

# Start from the persisted filtered oxide dataset and narrow it
# to relatively stable semiconducting candidates before expensive motif analysis.
candidate_df = (
    spark.table(
        "bootcamp_students.banjadehuta.silver_filtered_materials"
    )
    .filter(
        (F.col("band_gap") >= 1.5) &
        (F.col("band_gap") <= 3.0) &
        (F.col("energy_above_hull") <= 0.05)
    )
)

print("Candidate materials:", candidate_df.count())

display(
    candidate_df.select(
        "material_id",
        "formula",
        "band_gap",
        "energy_above_hull",
        "num_sites"
    )
    .orderBy(
        F.asc("energy_above_hull"),
        F.asc("band_gap")
    )
)

# COMMAND ----------

import json
import time

from pymatgen.core import Structure
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
    LocalGeometryFinder,
)

# Pull only the screened candidates into Python for the more expensive
# coordination-environment calculation.
candidate_rows = (
    candidate_df
    .select(
        "material_id",
        "formula",
        "num_sites",
        "structure_json"
    )
    .collect()
)

print(f"Candidates to analyze: {len(candidate_rows)}")

# COMMAND ----------

# Run motif extraction only on scientifically relevant candidates.
# Progress is printed per material so slow structures are easy to identify.
candidate_motif_rows = []
failed_candidates = []

for i, row in enumerate(candidate_rows, start=1):
    start = time.time()

    material_id = row["material_id"]
    formula = row["formula"]

    try:
        structure = Structure.from_dict(
            json.loads(row["structure_json"])
        )

        lgf = LocalGeometryFinder()

        coord_envs = lgf.compute_coordination_environments(
            structure,
            only_cations=True
        )

        for site_index, environments in enumerate(coord_envs):
            if not environments:
                continue

            # MVP assignment: choose the coordination geometry
            # with the lowest continuous symmetry measure.
            best_env = min(
                environments,
                key=lambda env: env["csm"]
            )

            candidate_motif_rows.append({
                "material_id": material_id,
                "formula": formula,
                "site_index": site_index,
                "element": structure[site_index].species_string,
                "motif": best_env["ce_symbol"],
                "csm": float(best_env["csm"]),
            })

    except Exception as exc:
        failed_candidates.append({
            "material_id": material_id,
            "formula": formula,
            "error": str(exc)
        })

    elapsed = time.time() - start

    print(
        f"[{i}/{len(candidate_rows)}] "
        f"{material_id} | {formula} | "
        f"{row['num_sites']} sites | "
        f"{elapsed:.1f}s"
    )

print(f"\nMotif rows: {len(candidate_motif_rows)}")
print(f"Failures: {len(failed_candidates)}")

# COMMAND ----------

# Save the expensive motif extraction results immediately.
# This table becomes the persistent structural representation
# for the screened semiconductor candidates.
candidate_motifs_sdf = spark.createDataFrame(candidate_motif_rows)

candidate_motifs_sdf.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(
        "bootcamp_students.banjadehuta.gold_candidate_motifs"
    )

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     material_id,
# MAGIC     formula,
# MAGIC     site_index,
# MAGIC     element,
# MAGIC     motif,
# MAGIC     csm
# MAGIC FROM bootcamp_students.banjadehuta.gold_candidate_motifs
# MAGIC ORDER BY material_id, site_index;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT COUNT(*) AS motif_records
# MAGIC FROM bootcamp_students.banjadehuta.gold_candidate_motifs;

# COMMAND ----------

from pyspark.sql import functions as F

# Aggregate site-level motif assignments into one structural profile per candidate.
candidate_motif_profiles = (
    spark.table(
        "bootcamp_students.banjadehuta.gold_candidate_motifs"
    )
    .groupBy("material_id", "formula")
    .agg(
        F.collect_set(
            F.concat_ws("|", "element", "motif")
        ).alias("motifs"),

        F.count("*").alias("motif_site_count"),

        F.avg("csm").alias("mean_csm"),

        F.min("csm").alias("min_csm"),

        F.max("csm").alias("max_csm")
    )
)

display(candidate_motif_profiles)

# COMMAND ----------

# Combine electronic/stability properties with the motif profile.
# This is the table the discovery layer and AI agent can query directly.
candidate_discovery = (
    candidate_df.alias("m")
    .join(
        candidate_motif_profiles.alias("s"),
        on=["material_id", "formula"],
        how="left"
    )
    .select(
        "material_id",
        "formula",
        "band_gap",
        "formation_energy_per_atom",
        "energy_above_hull",
        "density",
        "num_sites",
        "motifs",
        "motif_site_count",
        "mean_csm",
        "min_csm",
        "max_csm"
    )
)

display(
    candidate_discovery
    .orderBy(
        F.asc("energy_above_hull"),
        F.asc("band_gap")
    )
)

# COMMAND ----------

# Persist the application-ready candidate dataset.
candidate_discovery.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(
        "bootcamp_students.banjadehuta.gold_candidate_discovery"
    )

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     material_id,
# MAGIC     formula,
# MAGIC     band_gap,
# MAGIC     energy_above_hull,
# MAGIC     motifs,
# MAGIC     mean_csm
# MAGIC FROM bootcamp_students.banjadehuta.gold_candidate_discovery
# MAGIC ORDER BY energy_above_hull, band_gap;

# COMMAND ----------

from pyspark.sql import functions as F

# Reusable search function for the discovery layer.
# The AI agent will eventually call this instead of writing SQL directly.
def search_candidates(
    min_band_gap=None,
    max_band_gap=None,
    max_energy_above_hull=None,
    limit=20
):
    df = spark.table(
        "bootcamp_students.banjadehuta.gold_candidate_discovery"
    )

    # Apply only the filters supplied by the user/agent.
    if min_band_gap is not None:
        df = df.filter(F.col("band_gap") >= min_band_gap)

    if max_band_gap is not None:
        df = df.filter(F.col("band_gap") <= max_band_gap)

    if max_energy_above_hull is not None:
        df = df.filter(
            F.col("energy_above_hull") <= max_energy_above_hull
        )

    # Prefer lower energy above hull, then lower motif distortion.
    return (
        df
        .orderBy(
            F.asc("energy_above_hull"),
            F.asc("mean_csm")
        )
        .limit(limit)
    )

# COMMAND ----------

# Example screening query for relatively stable oxide semiconductors.
results = search_candidates(
    min_band_gap=1.8,
    max_band_gap=2.5,
    max_energy_above_hull=0.05,
    limit=10
)

display(results)

# COMMAND ----------

# Broader search to make sure the function behaves correctly
# under different property constraints.
results = search_candidates(
    min_band_gap=1.5,
    max_band_gap=3.0,
    limit=20
)

display(results)

# COMMAND ----------

from pyspark.sql import functions as F

# Find materials that share local coordination motifs with a reference material.
# The score is based first on how many motifs are shared, then on CSM similarity.
def find_similar_materials(material_id, limit=20):

    motifs = spark.table(
        "bootcamp_students.banjadehuta.gold_candidate_motifs"
    )

    candidates = spark.table(
        "bootcamp_students.banjadehuta.gold_candidate_discovery"
    )

    # Build a normalized motif key so element + geometry define the local environment.
    motif_index = (
        motifs
        .withColumn(
            "motif_key",
            F.concat_ws("|", "element", "motif")
        )
        .select(
            "material_id",
            "motif_key",
            "csm"
        )
    )

    # Motifs belonging to the reference material.
    reference = (
        motif_index
        .filter(F.col("material_id") == material_id)
        .alias("ref")
    )

    # All motifs from other candidate materials.
    others = (
        motif_index
        .filter(F.col("material_id") != material_id)
        .alias("cand")
    )

    # Join on shared motif identity.
    shared = (
        reference
        .join(
            others,
            F.col("ref.motif_key") == F.col("cand.motif_key")
        )
        .select(
            F.col("cand.material_id").alias("candidate_id"),
            F.col("ref.motif_key").alias("shared_motif"),
            F.col("ref.csm").alias("reference_csm"),
            F.col("cand.csm").alias("candidate_csm")
        )
    )

    # Aggregate similarity at the material level.
    similarity = (
        shared
        .groupBy("candidate_id")
        .agg(
            F.countDistinct("shared_motif").alias("shared_motif_count"),
            F.collect_set("shared_motif").alias("shared_motifs"),
            F.avg(
                F.abs(
                    F.col("reference_csm") -
                    F.col("candidate_csm")
                )
            ).alias("mean_csm_difference")
        )
    )

    # Add the physical properties used in screening.
    result = (
        similarity.alias("s")
        .join(
            candidates.alias("m"),
            F.col("s.candidate_id") == F.col("m.material_id"),
            how="left"
        )
        .select(
            F.col("candidate_id").alias("material_id"),
            "formula",
            "band_gap",
            "energy_above_hull",
            "shared_motif_count",
            "shared_motifs",
            "mean_csm_difference"
        )
        .orderBy(
            F.desc("shared_motif_count"),
            F.asc("mean_csm_difference"),
            F.asc("energy_above_hull")
        )
        .limit(limit)
    )

    return result

# COMMAND ----------

# Inspect available candidate IDs first.
display(
    spark.table(
        "bootcamp_students.banjadehuta.gold_candidate_discovery"
    )
    .select("material_id", "formula", "motifs")
)

# COMMAND ----------

results = find_similar_materials(
    material_id="YOUR_MP_ID",
    limit=10
)

display(results)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- Pick a real material that we can save into the Lakebase research project.
# MAGIC SELECT
# MAGIC     material_id,
# MAGIC     formula,
# MAGIC     band_gap,
# MAGIC     energy_above_hull,
# MAGIC     motifs
# MAGIC FROM bootcamp_students.banjadehuta.gold_candidate_discovery
# MAGIC ORDER BY energy_above_hull
# MAGIC LIMIT 10;

# COMMAND ----------

# PostgreSQL driver used by the application to communicate with Lakebase.
%pip install psycopg[binary]

# COMMAND ----------

import getpass

# Use the temporary Lakebase OAuth token without storing it in notebook source.
LAKEBASE_TOKEN = getpass.getpass("Lakebase OAuth token: ")

# COMMAND ----------

import psycopg

# Use the exact host and role shown under
# Lakebase -> Connect to your database.
LAKEBASE_HOST = "PASTE_THE_HOST_FROM_LAKEBASE_HERE"
LAKEBASE_USER = "PASTE_THE_ROLE_FROM_LAKEBASE_HERE"

conn = psycopg.connect(
    host=LAKEBASE_HOST,
    dbname="databricks_postgres",
    user=LAKEBASE_USER,
    password=LAKEBASE_TOKEN,
    sslmode="require"
)

print("Connected to Lakebase")

# COMMAND ----------

# MAGIC %pip install psycopg[binary]

# COMMAND ----------

import getpass
import psycopg

# Full Lakebase connection string copied from the Lakebase Connect dialog.
# The OAuth token is supplied separately as the temporary password.
LAKEBASE_URL = (
    "postgresql://banjadehuta%40gmail.com"
    "@ep-odd-sound-d1dox0kv.database.us-west-2.cloud.databricks.com"
    "/databricks_postgres?sslmode=require"
)

# Copy the OAuth token from:
# Lakebase -> Connect to your database -> Copy OAuth token
# The token is temporary and expires after about 1 hour.
LAKEBASE_TOKEN = getpass.getpass("Paste Lakebase OAuth token: ")

try:
    # Open the PostgreSQL connection.
    conn = psycopg.connect(
        LAKEBASE_URL,
        password=LAKEBASE_TOKEN
    )

    print("Connected to Lakebase successfully.")

    # Verify that the MotifLens schema and tables are accessible.
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'motiflens'
            ORDER BY table_name;
        """)

        tables = cur.fetchall()

    print("\nMotifLens tables:")
    for table in tables:
        print(" -", table[0])

    # Verify the research project created earlier.
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                project_id,
                project_name,
                description,
                created_at
            FROM motiflens.research_projects
            ORDER BY created_at;
        """)

        projects = cur.fetchall()

    print("\nResearch projects:")
    for project in projects:
        print(project)

    # Verify previously saved materials.
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                c.candidate_id,
                p.project_name,
                c.material_id,
                c.status,
                c.similarity_score,
                c.created_at
            FROM motiflens.candidate_materials AS c
            JOIN motiflens.research_projects AS p
                ON c.project_id = p.project_id
            ORDER BY c.created_at;
        """)

        candidates = cur.fetchall()

    print("\nSaved candidates:")
    for candidate in candidates:
        print(candidate)

except Exception as exc:
    print("Lakebase connection failed:")
    print(exc)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SHOW TABLES IN bootcamp_students.banjadehuta;

# COMMAND ----------

