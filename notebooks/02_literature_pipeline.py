# Databricks notebook source
import requests

# Start with a simple literature search before building
# the full Spark ingestion pipeline.
search_term = "lead oxide PbO"

url = "https://api.openalex.org/works"

params = {
    "search": search_term,
    "per-page": 10
}

response = requests.get(
    url,
    params=params,
    timeout=30
)

response.raise_for_status()

data = response.json()

print("Results returned:", len(data["results"]))

# COMMAND ----------

# Inspect basic publication metadata returned by OpenAlex.
for work in data["results"][:5]:
    print("Title:", work.get("title"))
    print("Year:", work.get("publication_year"))
    print("DOI:", work.get("doi"))
    print("OpenAlex ID:", work.get("id"))
    print("-" * 80)

# COMMAND ----------

#check for abstracts
for work in data["results"][:5]:
    print("Title:", work.get("title"))

    abstract_index = work.get("abstract_inverted_index")

    if abstract_index:
        print("Abstract available: YES")
        print("Number of unique words:", len(abstract_index))
    else:
        print("Abstract available: NO")

    print("-" * 80)

# COMMAND ----------

def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return None

    word_positions = []

    for word, positions in inverted_index.items():
        for position in positions:
            word_positions.append((position, word))

    word_positions.sort(key=lambda x: x[0])

    return " ".join(
        word for _, word in word_positions
    )


# Test reconstruction on the first few papers
for work in data["results"][:5]:

    abstract_text = reconstruct_abstract(
        work.get("abstract_inverted_index")
    )

    print("TITLE:")
    print(work.get("title"))

    print("\nABSTRACT:")
    print(
        abstract_text
        if abstract_text
        else "No abstract available"
    )

    print("\n" + "=" * 100 + "\n")

# COMMAND ----------

from datetime import datetime, timezone

literature_records = []

for work in data["results"]:

    abstract_text = reconstruct_abstract(
        work.get("abstract_inverted_index")
    )

    literature_records.append({
        "work_id": work.get("id"),
        "doi": work.get("doi"),
        "title": work.get("title"),
        "abstract_text": abstract_text,
        "publication_year": work.get("publication_year"),
        "search_term": search_term,
        "retrieved_at": datetime.now(timezone.utc).isoformat()
    })

print(
    f"Prepared {len(literature_records)} "
    "literature records"
)

# COMMAND ----------

literature_records[0]

# COMMAND ----------

from pyspark.sql import functions as F

literature_df = spark.createDataFrame(literature_records)

display(literature_df)

# COMMAND ----------

bronze_literature_df = (
    literature_df
    .withColumn(
        "retrieved_at",
        F.to_timestamp("retrieved_at")
    )
    .dropDuplicates(["work_id"])
)

(
    bronze_literature_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        "bootcamp_students.banjadehuta.bronze_literature_raw"
    )
)

print("bronze_literature_raw created successfully.")

# COMMAND ----------

bronze_check = spark.table(
    "bootcamp_students.banjadehuta.bronze_literature_raw"
)

print("Total literature records:", bronze_check.count())

display(
    bronze_check.select(
        "title",
        "abstract_text",
        "publication_year",
        "search_term"
    )
)

# COMMAND ----------

from pyspark.sql import functions as F

bronze_df = spark.table(
    "bootcamp_students.banjadehuta.bronze_literature_raw"
)

silver_literature_df = (
    bronze_df

    # Keep publications that actually contain an abstract
    .filter(
        F.col("abstract_text").isNotNull()
        & (F.length(F.trim(F.col("abstract_text"))) > 0)
    )

    # Normalize whitespace in the free-form text
    .withColumn(
        "clean_abstract",
        F.regexp_replace(
            F.trim(F.col("abstract_text")),
            r"\s+",
            " "
        )
    )

    # Normalize title as well
    .withColumn(
        "clean_title",
        F.regexp_replace(
            F.trim(F.col("title")),
            r"\s+",
            " "
        )
    )

    # Derive features from the unstructured text
    .withColumn(
        "abstract_word_count",
        F.size(
            F.split(
                F.col("clean_abstract"),
                r"\s+"
            )
        )
    )

    .withColumn(
        "abstract_char_count",
        F.length("clean_abstract")
    )

    # Lowercase representation useful for matching/search
    .withColumn(
        "abstract_search_text",
        F.lower("clean_abstract")
    )

    # Remove duplicate publications
    .dropDuplicates(["work_id"])
)

display(
    silver_literature_df.select(
        "title",
        "publication_year",
        "abstract_word_count",
        "abstract_char_count",
        "clean_abstract"
    )
)

# COMMAND ----------

(
    silver_literature_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        "bootcamp_students.banjadehuta."
        "silver_literature"
    )
)

print("silver_literature created successfully.")

# COMMAND ----------

silver_check = spark.table(
    "bootcamp_students.banjadehuta.silver_literature"
)

print("Bronze records:", bronze_df.count())
print("Silver records:", silver_check.count())

display(
    silver_check.select(
        "title",
        "publication_year",
        "search_term",
        "abstract_word_count",
        "clean_abstract"
    )
)

# COMMAND ----------

candidate_df = (
    spark.table(
        "bootcamp_students.banjadehuta.gold_candidate_discovery"
    )
    .select(
        "material_id",
        "formula",
        "band_gap",
        "energy_above_hull"
    )
    .orderBy(
        "energy_above_hull",
        "band_gap"
    )
    .limit(10)
)

display(candidate_df)

# COMMAND ----------

candidate_materials = [
    row.asDict()
    for row in candidate_df.collect()
]

candidate_materials

# COMMAND ----------

import requests
import time
from datetime import datetime, timezone

literature_records_multi = []

for material in candidate_materials:

    material_id = material["material_id"]
    formula = material["formula"]

    # Search using the formula plus materials-related context.
    search_term = f'"{formula}" material oxide'

    response = requests.get(
        "https://api.openalex.org/works",
        params={
            "search": search_term,
            "per-page": 10
        },
        timeout=30
    )

    response.raise_for_status()

    works = response.json().get("results", [])

    for work in works:

        abstract_text = reconstruct_abstract(
            work.get("abstract_inverted_index")
        )

        literature_records_multi.append({
            "material_id": material_id,
            "formula": formula,
            "work_id": work.get("id"),
            "doi": work.get("doi"),
            "title": work.get("title"),
            "abstract_text": abstract_text,
            "publication_year": work.get("publication_year"),
            "search_term": search_term,
            "retrieved_at": datetime.now(
                timezone.utc
            ).isoformat()
        })

    print(
        f"{material_id} ({formula}): "
        f"{len(works)} publications"
    )

    # Be polite to the external API.
    time.sleep(0.2)

print(
    f"\nTotal literature associations: "
    f"{len(literature_records_multi)}"
)

# COMMAND ----------

for record in literature_records_multi[:20]:
    print(
        record["material_id"],
        "|",
        record["formula"],
        "|",
        record["title"]
    )

# COMMAND ----------

from pyspark.sql import functions as F

literature_multi_df = spark.createDataFrame(
    literature_records_multi
)

bronze_multi_df = (
    literature_multi_df
    .withColumn(
        "retrieved_at",
        F.to_timestamp("retrieved_at")
    )
    .dropDuplicates(
        ["material_id", "work_id"]
    )
)

(
    bronze_multi_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        "bootcamp_students.banjadehuta."
        "bronze_material_literature"
    )
)

print(
    "Bronze literature associations:",
    bronze_multi_df.count()
)

# COMMAND ----------

bronze_multi_df = spark.table(
    "bootcamp_students.banjadehuta."
    "bronze_material_literature"
)

silver_material_literature = (
    bronze_multi_df

    # Require actual unstructured abstract text
    .filter(
        F.col("abstract_text").isNotNull()
        & (F.length(F.trim("abstract_text")) > 0)
    )

    # Clean title
    .withColumn(
        "clean_title",
        F.regexp_replace(
            F.trim("title"),
            r"\s+",
            " "
        )
    )

    # Clean free-form abstract
    .withColumn(
        "clean_abstract",
        F.regexp_replace(
            F.trim("abstract_text"),
            r"\s+",
            " "
        )
    )

    # Build searchable text
    .withColumn(
        "search_text",
        F.lower(
            F.concat_ws(
                " ",
                "clean_title",
                "clean_abstract"
            )
        )
    )

    # Text-derived features
    .withColumn(
        "abstract_word_count",
        F.size(
            F.split(
                "clean_abstract",
                r"\s+"
            )
        )
    )

    .withColumn(
        "abstract_char_count",
        F.length("clean_abstract")
    )

    # Basic relevance signal:
    # does the title/abstract actually mention
    # the candidate formula?
    .withColumn(
        "formula_mentioned",
        F.instr(
            F.lower("search_text"),
            F.lower("formula")
        ) > 0
    )

    .dropDuplicates(
        ["material_id", "work_id"]
    )
)

display(
    silver_material_literature.select(
        "material_id",
        "formula",
        "title",
        "abstract_word_count",
        "formula_mentioned"
    )
)

# COMMAND ----------

silver_material_literature.groupBy(
    "formula_mentioned"
).count().show()

# COMMAND ----------

display(
    silver_material_literature
    .filter(F.col("formula_mentioned") == True)
    .select(
        "material_id",
        "formula",
        "title",
        "clean_abstract"
    )
)

# COMMAND ----------

from pyspark.sql import functions as F

scored_literature_df = (
    silver_material_literature

    .withColumn(
        "formula_in_title",
        F.instr(
            F.lower(F.col("clean_title")),
            F.lower(F.col("formula"))
        ) > 0
    )

    .withColumn(
        "formula_in_abstract",
        F.instr(
            F.lower(F.col("clean_abstract")),
            F.lower(F.col("formula"))
        ) > 0
    )

    # Simple interpretable relevance score
    .withColumn(
        "relevance_score",
        F.when(F.col("formula_in_title"), 2).otherwise(0)
        + F.when(F.col("formula_in_abstract"), 1).otherwise(0)
    )
)

# COMMAND ----------

display(
    scored_literature_df
    .select(
        "material_id",
        "formula",
        "title",
        "formula_in_title",
        "formula_in_abstract",
        "relevance_score",
        "abstract_word_count"
    )
    .orderBy(
        F.desc("relevance_score"),
        F.desc("abstract_word_count")
    )
)

# COMMAND ----------

relevant_literature_df = (
    scored_literature_df
    .filter(F.col("relevance_score") >= 2)
    .dropDuplicates(["material_id", "work_id"])
)

print(
    "Raw associations:",
    bronze_multi_df.count()
)

print(
    "Relevant literature:",
    relevant_literature_df.count()
)

display(
    relevant_literature_df.select(
        "material_id",
        "formula",
        "title",
        "relevance_score",
        "clean_abstract"
    )
)

# COMMAND ----------

(
    relevant_literature_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        "bootcamp_students.banjadehuta."
        "silver_material_literature"
    )
)

print("silver_material_literature created successfully.")

# COMMAND ----------

silver_check = spark.table(
    "bootcamp_students.banjadehuta."
    "silver_material_literature"
)

print("Silver literature records:", silver_check.count())

display(
    silver_check.select(
        "material_id",
        "formula",
        "title",
        "publication_year",
        "relevance_score",
        "abstract_word_count",
        "clean_abstract"
    )
)

# COMMAND ----------

(
    relevant_literature_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        "bootcamp_students.banjadehuta."
        "silver_material_literature"
    )
)

print(
    "Saved records:",
    spark.table(
        "bootcamp_students.banjadehuta."
        "silver_material_literature"
    ).count()
)

# COMMAND ----------

display(
    spark.table(
        "bootcamp_students.banjadehuta."
        "silver_material_literature"
    ).select(
        "material_id",
        "formula",
        "title",
        "publication_year",
        "relevance_score",
        "clean_abstract"
    )
)