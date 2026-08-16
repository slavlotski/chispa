from __future__ import annotations

import pytest
from pyspark.sql import SparkSession

from chispa import Chispa, DataFramesNotEqualError
from chispa.default_formats import DefaultFormats
from chispa.formatting import Color, FormattingConfig, Style


def describe_chispa():
    def it_falls_back_to_the_default_formats():
        chispa = Chispa()
        assert chispa.formats.mismatched_rows.color == Color.RED
        assert chispa.formats.matched_rows.color == Color.BLUE
        assert chispa.formats.mismatched_cells.style == [Style.UNDERLINE]

    def it_keeps_a_formatting_config_as_is():
        formats = FormattingConfig(mismatched_rows={"color": "green"})
        chispa = Chispa(formats)
        assert chispa.formats is formats

    def it_converts_an_arbitrary_dataclass_to_a_formatting_config():
        chispa = Chispa(DefaultFormats())
        assert isinstance(chispa.formats, FormattingConfig)
        assert chispa.formats.mismatched_rows.color == Color.RED

    def it_does_not_throw_when_dfs_are_equal(spark: SparkSession):
        data = [(1, "jose"), (2, "li")]
        df1 = spark.createDataFrame(data, ["num", "name"])
        df2 = spark.createDataFrame(data, ["num", "name"])
        Chispa().assert_df_equality(df1, df2)

    def it_throws_when_dfs_are_not_equal(spark: SparkSession):
        df1 = spark.createDataFrame([(1, "jose"), (2, "li")], ["num", "name"])
        df2 = spark.createDataFrame([(1, "jose"), (2, "laura")], ["num", "name"])
        with pytest.raises(DataFramesNotEqualError):
            Chispa().assert_df_equality(df1, df2)
