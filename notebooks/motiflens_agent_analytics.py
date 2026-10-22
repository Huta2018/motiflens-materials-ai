# Databricks notebook source
from pyspark.sql import functions as F

source_table = (
    "bootcamp_students.banjadehuta."
    "lb_agent_actions_history"
)

target_table = (
    "bootcamp_students.banjadehuta."
    "gold_agent_activity"
)

df = (
    spark.table(source_table)
    .filter(F.col("_pg_change_type") == "insert")
    .select(
        "_pg_change_type",
        "_pg_lsn",
        "_pg_xid",
        F.col("_timestamp").alias("cdf_timestamp"),
        "material_id",
        "action_type",
        "action_details",
        "created_at",
        F.get_json_object(
            "action_details",
            "$.source"
        ).alias("source"),
        F.get_json_object(
            "action_details",
            "$.status"
        ).alias("status"),
        F.to_date("created_at").alias("event_date")
    )
)

(
    df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(target_table)
)

print("MotifLens agent analytics refreshed.")

# COMMAND ----------

display(
    spark.sql("""
        SELECT
            event_date,
            source,
            action_type,
            COUNT(*) AS action_count
        FROM bootcamp_students.banjadehuta.gold_agent_activity
        GROUP BY
            event_date,
            source,
            action_type
        ORDER BY
            event_date DESC,
            action_count DESC
    """)
)