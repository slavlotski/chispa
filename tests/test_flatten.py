from __future__ import annotations

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
)

from chispa import flatten_dataframe


def describe_flatten_dataframe():
    def it_flattens_struct_columns(spark: SparkSession):
        schema = StructType([
            StructField(
                "name",
                StructType([
                    StructField("first", StringType()),
                    StructField("last", StringType()),
                ]),
            ),
            StructField("state", StringType()),
        ])
        df = spark.createDataFrame([(("James", "Smith"), "OH")], schema)

        flat = flatten_dataframe(df)

        assert sorted(flat.columns) == ["name_first", "name_last", "state"]
        row = flat.collect()[0]
        assert row["name_first"] == "James"
        assert row["name_last"] == "Smith"
        assert row["state"] == "OH"

    def it_flattens_nested_struct_columns(spark: SparkSession):
        schema = StructType([
            StructField(
                "person",
                StructType([
                    StructField(
                        "name",
                        StructType([
                            StructField("first", StringType()),
                        ]),
                    ),
                    StructField("age", IntegerType()),
                ]),
            ),
        ])
        df = spark.createDataFrame([((("Anna",), 30),)], schema)  # type: ignore[arg-type]

        flat = flatten_dataframe(df)

        assert sorted(flat.columns) == ["person_age", "person_name_first"]

    def it_explodes_array_columns(spark: SparkSession):
        schema = StructType([
            StructField("id", IntegerType()),
            StructField("tags", ArrayType(StringType())),
        ])
        df = spark.createDataFrame([(1, ["a", "b"]), (2, ["c"])], schema)

        flat = flatten_dataframe(df)

        assert flat.columns == ["id", "tags"]
        rows = sorted((r["id"], r["tags"]) for r in flat.collect())
        assert rows == [(1, "a"), (1, "b"), (2, "c")]

    def it_flattens_map_columns_into_one_column_per_key(spark: SparkSession):
        schema = StructType([
            StructField("id", IntegerType()),
            StructField("attrs", MapType(StringType(), StringType())),
        ])
        df = spark.createDataFrame([(1, {"color": "red", "size": "L"})], schema)

        flat = flatten_dataframe(df)

        assert sorted(flat.columns) == ["attrs_color", "attrs_size", "id"]
        row = flat.collect()[0]
        assert row["attrs_color"] == "red"
        assert row["attrs_size"] == "L"

    def it_handles_mixed_struct_array_and_map(spark: SparkSession):
        schema = StructType([
            StructField("state", StringType()),
            StructField("info", MapType(StringType(), StringType())),
            StructField(
                "counties",
                ArrayType(
                    StructType([
                        StructField("name", StringType()),
                        StructField("population", LongType()),
                    ])
                ),
            ),
        ])
        df = spark.createDataFrame(
            [("FL", {"governor": "Rick"}, [("Dade", 12345), ("Broward", 40000)])],
            schema,
        )

        flat = flatten_dataframe(df)

        assert sorted(flat.columns) == ["counties_name", "counties_population", "info_governor", "state"]
        rows = sorted((r["state"], r["counties_name"], r["counties_population"]) for r in flat.collect())
        assert rows == [("FL", "Broward", 40000), ("FL", "Dade", 12345)]

    def it_uses_a_custom_separator(spark: SparkSession):
        schema = StructType([
            StructField(
                "name",
                StructType([
                    StructField("first", StringType()),
                ]),
            ),
        ])
        df = spark.createDataFrame([(("Anna",),)], schema)  # type: ignore[arg-type]

        flat = flatten_dataframe(df, sep=":")

        assert flat.columns == ["name:first"]

    def it_rejects_dot_separator(spark: SparkSession):
        df = spark.createDataFrame([(1,)], ["x"])
        with pytest.raises(ValueError, match=r"must not be '\.'"):
            flatten_dataframe(df, sep=".")

    def it_returns_unchanged_df_when_no_nested_columns(spark: SparkSession):
        df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "name"])

        flat = flatten_dataframe(df)

        assert flat.columns == ["id", "name"]
        assert flat.collect() == df.collect()
