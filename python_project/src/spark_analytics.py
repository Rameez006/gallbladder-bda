"""
Module 3: Big Data Analytics with Apache Spark (PySpark)
-------------------------------------------------------
Demonstrates how distributed DataFrames load, partition, filter, and
aggregate medical records at scale using PySpark SQL and transformations.
"""

from typing import Any
import pandas as pd


def run_pyspark_analytics(csv_path: str, partitions: int = 4) -> dict[str, Any]:
    """
    Initialize a local SparkSession, load the medical CSV across partitions,
    run distributed filtering and groupBy aggregations, and return summary dicts.
    """
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = (
        SparkSession.builder.appName("GallbladderCancerBigDataAnalytics")
        .master(f"local[{partitions}]")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )

    try:
        sdf = (
            spark.read.option("header", "true")
            .option("inferSchema", "true")
            .csv(csv_path)
            .repartition(partitions)
        )

        # Clean nulls and duplicates in distributed Spark mode
        sdf_clean = sdf.dropDuplicates().na.drop(subset=["Cancer_Risk"])

        total_records = sdf_clean.count()
        num_partitions = sdf_clean.rdd.getNumPartitions()

        # Distributed aggregation by Gallstones presence
        gallstone_summary = (
            sdf_clean.groupBy("Gallstones")
            .agg(
                F.count("*").alias("patient_count"),
                F.round(F.avg("Wall_Thickness_mm"), 2).alias("avg_wall_thickness_mm"),
                F.round(F.avg("Age"), 1).alias("avg_age"),
                F.round(F.mean("Cancer_Risk") * 100, 2).alias("elevated_risk_pct"),
            )
            .orderBy("Gallstones")
            .toPandas()
        )

        # Distributed filtering: High-risk clinical cohort
        high_risk_cohort = sdf_clean.filter(
            (F.col("Age") >= 50)
            & (F.col("Gallstones") == 1)
            & (F.col("Wall_Thickness_mm") >= 3.5)
        )

        # Age-bracket aggregation using Spark SQL expressions
        age_bracket_summary = (
            sdf_clean.withColumn(
                "Age_Group",
                F.when(F.col("Age") < 40, "Under 40")
                .when(F.col("Age") < 55, "40 - 54")
                .when(F.col("Age") < 70, "55 - 69")
                .otherwise("70+"),
            )
            .groupBy("Age_Group")
            .agg(
                F.count("*").alias("records"),
                F.round(F.mean("Cancer_Risk") * 100, 2).alias("risk_rate_pct"),
            )
            .orderBy("Age_Group")
            .toPandas()
        )

        return {
            "total_records": total_records,
            "num_partitions": num_partitions,
            "high_risk_filter_count": high_risk_cohort.count(),
            "gallstone_summary": gallstone_summary,
            "age_bracket_summary": age_bracket_summary,
            "schema_fields": [(f.name, str(f.dataType)) for f in sdf_clean.schema.fields],
        }
    finally:
        spark.stop()
