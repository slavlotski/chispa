from __future__ import annotations

import pytest
from pyspark.sql.types import (
    ArrayType,
    DecimalType,
    DoubleType,
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType,
)

from chispa.bcolors import bcolors
from chispa.common_enums import OutputFormat
from chispa.schema_comparer import (
    SchemasNotEqualError,
    _are_fields_shallow_equal,
    _compare_array_types,
    _compare_map_types,
    are_schemas_equal,
    are_structfields_equal,
    assert_schema_equality,
    assert_schema_equality_ignore_nullable,
    create_schema_comparison_tree,
    print_schema_diff,
)


def _tree_line(tree: str, needle: str) -> str:
    """Return the single line of a comparison tree containing `needle`.

    Both schemas are printed side by side on one line, which is colored red when they
    differ and blue when they match, so the color of the line is the actual assertion.
    """
    matches = [line for line in tree.split("\n") if needle in line]
    assert len(matches) == 1, f"expected exactly one line containing {needle!r}, found {len(matches)}"
    return matches[0]


def describe_assert_schema_equality():
    def it_does_nothing_when_equal():
        s1 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
        ])
        s2 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
        ])
        assert_schema_equality(s1, s2)

    def it_throws_when_column_names_differ():
        s1 = StructType([
            StructField("HAHA", StringType(), True),
            StructField("age", IntegerType(), True),
        ])
        s2 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
        ])
        with pytest.raises(SchemasNotEqualError):
            assert_schema_equality(s1, s2)

    def it_throws_when_schema_lengths_differ():
        s1 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
        ])
        s2 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("fav_number", IntegerType(), True),
        ])
        with pytest.raises(SchemasNotEqualError):
            assert_schema_equality(s1, s2)

    def it_throws_when_data_types_differ():
        s1 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("height", DecimalType(10, 2), True),
        ])
        s2 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("height", DecimalType(10, 3), True),
        ])
        with pytest.raises(SchemasNotEqualError):
            assert_schema_equality(s1, s2)

    def it_throws_when_data_types_differ_with_enabled_ignore_nullability():
        s1 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("height", DecimalType(10, 2), True),
        ])
        s2 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("height", DecimalType(10, 3), True),
        ])
        with pytest.raises(SchemasNotEqualError):
            assert_schema_equality(s1, s2, ignore_nullable=True)

    def it_throws_when_data_types_differ_with_enabled_ignore_metadata():
        s1 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("height", DecimalType(10, 2), True),
        ])
        s2 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("height", DecimalType(10, 3), True),
        ])
        with pytest.raises(SchemasNotEqualError):
            assert_schema_equality(s1, s2, ignore_metadata=True)


def describe_tree_string():
    def it_prints_correctly_for_wide_schemas():
        with open("tests/data/tree_string/it_prints_correctly_for_wide_schemas.txt") as f:
            expected = f.read()

        s1 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("fav_number", IntegerType(), True),
            StructField("fav_numbers", ArrayType(IntegerType(), True), True),
            StructField(
                "fav_colors",
                StructType([
                    StructField("red", IntegerType(), True),
                    StructField("green", IntegerType(), True),
                    StructField("blue", IntegerType(), True),
                ]),
            ),
        ])

        s2 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("fav_number", IntegerType(), True),
            StructField("fav_numbers", ArrayType(IntegerType(), True), True),
            StructField(
                "fav_colors",
                StructType([
                    StructField("orange", IntegerType(), True),
                    StructField("green", IntegerType(), True),
                    StructField("yellow", IntegerType(), True),
                ]),
            ),
        ])

        result = create_schema_comparison_tree(s1, s2, ignore_nullable=False, ignore_metadata=False)

        assert repr(result) + "\n" == expected

    def it_prints_correctly_for_wide_schemas_multiple_nested_structs():
        with open("tests/data/tree_string/it_prints_correctly_for_wide_schemas_multiple_nested_structs.txt") as f:
            expected = f.read()

        s1 = StructType([
            StructField("name", StringType(), True),
            StructField(
                "fav_genres",
                StructType([
                    StructField(
                        "rock",
                        StructType([
                            StructField("metal", IntegerType(), True),
                            StructField("punk", IntegerType(), True),
                        ]),
                        True,
                    ),
                    StructField(
                        "electronic",
                        StructType([
                            StructField("house", IntegerType(), True),
                            StructField("dubstep", IntegerType(), True),
                        ]),
                        True,
                    ),
                ]),
            ),
        ])

        s2 = StructType([
            StructField("name", StringType(), True),
            StructField(
                "fav_genres",
                StructType([
                    StructField(
                        "rock",
                        StructType([
                            StructField("metal", IntegerType(), True),
                            StructField("classic", IntegerType(), True),
                        ]),
                        True,
                    ),
                    StructField(
                        "electronic",
                        StructType([
                            StructField("house", IntegerType(), True),
                            StructField("dubstep", IntegerType(), True),
                        ]),
                        True,
                    ),
                    StructField(
                        "pop",
                        StructType([
                            StructField("pop", IntegerType(), True),
                        ]),
                        True,
                    ),
                ]),
            ),
        ])

        result = create_schema_comparison_tree(s1, s2, ignore_nullable=False, ignore_metadata=False)
        assert repr(result) + "\n" == expected

    def it_prints_correctly_for_wide_schemas_ignore_nullable():
        with open("tests/data/tree_string/it_prints_correctly_for_wide_schemas_ignore_nullable.txt") as f:
            expected = f.read()

        s1 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("fav_number", IntegerType(), True),
            StructField("fav_numbers", ArrayType(IntegerType(), True), True),
            StructField(
                "fav_colors",
                StructType([
                    StructField("red", IntegerType(), True),
                    StructField("green", IntegerType(), True),
                    StructField("blue", IntegerType(), True),
                ]),
            ),
        ])

        s2 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), False),
            StructField("fav_number", IntegerType(), True),
            StructField("fav_numbers", ArrayType(IntegerType(), True), False),
            StructField(
                "fav_colors",
                StructType([
                    StructField("orange", IntegerType(), True),
                    StructField("green", IntegerType(), False),
                    StructField("yellow", IntegerType(), True),
                ]),
            ),
        ])

        result = create_schema_comparison_tree(s1, s2, ignore_nullable=True, ignore_metadata=False)

        assert repr(result) + "\n" == expected

    def it_prints_correctly_for_wide_schemas_different_lengths():
        with open("tests/data/tree_string/it_prints_correctly_for_wide_schemas_different_lengths.txt") as f:
            expected = f.read()

        s1 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("fav_number", IntegerType(), True),
            StructField("fav_numbers", ArrayType(IntegerType(), True), True),
            StructField(
                "fav_colors",
                StructType([
                    StructField("red", IntegerType(), True),
                    StructField("green", IntegerType(), True),
                    StructField("blue", IntegerType(), True),
                ]),
            ),
        ])

        s2 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("fav_number", IntegerType(), True),
            StructField("fav_numbers", ArrayType(IntegerType(), True), True),
            StructField(
                "fav_colors",
                StructType([
                    StructField("orange", IntegerType(), True),
                    StructField("green", IntegerType(), True),
                    StructField("yellow", IntegerType(), True),
                    StructField("purple", IntegerType(), True),
                ]),
            ),
            StructField("phone_number", StringType(), True),
        ])

        result = create_schema_comparison_tree(s1, s2, ignore_nullable=False, ignore_metadata=False)
        assert repr(result) + "\n" == expected

    def it_prints_correctly_for_wide_schemas_ignore_metadata():
        with open("tests/data/tree_string/it_prints_correctly_for_wide_schemas_ignore_metadata.txt") as f:
            expected = f.read()

        s1 = StructType([
            StructField("name", StringType(), True, {"foo": "bar"}),
            StructField("age", IntegerType(), True),
            StructField("fav_number", IntegerType(), True),
            StructField("fav_numbers", ArrayType(IntegerType(), True), True),
            StructField(
                "fav_colors",
                StructType([
                    StructField("red", IntegerType(), True),
                    StructField("green", IntegerType(), True),
                    StructField("blue", IntegerType(), True),
                ]),
            ),
        ])

        s2 = StructType([
            StructField("name", StringType(), True, {"foo": "baz"}),
            StructField("age", IntegerType(), True),
            StructField("fav_number", IntegerType(), True),
            StructField("fav_numbers", ArrayType(IntegerType(), True), True),
            StructField(
                "fav_colors",
                StructType([
                    StructField("orange", IntegerType(), True),
                    StructField("green", IntegerType(), True),
                    StructField("yellow", IntegerType(), True),
                ]),
            ),
        ])
        result = create_schema_comparison_tree(s1, s2, ignore_nullable=False, ignore_metadata=True)
        assert repr(result) + "\n" == expected

    def it_prints_correctly_for_wide_schemas_with_metadata():
        with open("tests/data/tree_string/it_prints_correctly_for_wide_schemas_with_metadata.txt") as f:
            expected = f.read()

        s1 = StructType([
            StructField("name", StringType(), True, {"foo": "bar"}),
            StructField("age", IntegerType(), True),
            StructField("fav_number", IntegerType(), True),
            StructField("fav_numbers", ArrayType(IntegerType(), True), True),
            StructField(
                "fav_colors",
                StructType([
                    StructField("red", IntegerType(), True),
                    StructField("green", IntegerType(), True),
                    StructField("blue", IntegerType(), True),
                ]),
            ),
        ])

        s2 = StructType([
            StructField("name", StringType(), True, {"foo": "baz"}),
            StructField("age", IntegerType(), True),
            StructField("fav_number", IntegerType(), True),
            StructField("fav_numbers", ArrayType(IntegerType(), True), True),
            StructField(
                "fav_colors",
                StructType([
                    StructField("orange", IntegerType(), True),
                    StructField("green", IntegerType(), True),
                    StructField("yellow", IntegerType(), True),
                ]),
            ),
        ])

        result = create_schema_comparison_tree(s1, s2, ignore_nullable=False, ignore_metadata=False)
        assert repr(result) + "\n" == expected

    def it_shows_diff_for_array_element_type_mismatch():
        with open("tests/data/tree_string/array_element_type_mismatch.txt") as f:
            expected = f.read()

        s1 = StructType([
            StructField("id", IntegerType(), True),
            StructField("scores", ArrayType(IntegerType(), True), True),
        ])
        s2 = StructType([
            StructField("id", IntegerType(), True),
            StructField("scores", ArrayType(DoubleType(), True), True),
        ])
        result = create_schema_comparison_tree(s1, s2, ignore_nullable=False, ignore_metadata=False)
        assert repr(result) + "\n" == expected

    def it_shows_diff_for_array_containsNull_mismatch():
        with open("tests/data/tree_string/array_containsNull_mismatch.txt") as f:
            expected = f.read()

        s1 = StructType([
            StructField("id", IntegerType(), True),
            StructField("tags", ArrayType(StringType(), True), True),
        ])
        s2 = StructType([
            StructField("id", IntegerType(), True),
            StructField("tags", ArrayType(StringType(), False), True),
        ])
        result = create_schema_comparison_tree(s1, s2, ignore_nullable=False, ignore_metadata=False)
        assert repr(result) + "\n" == expected

    def it_shows_diff_for_array_of_structs():
        with open("tests/data/tree_string/array_of_structs_diff.txt") as f:
            expected = f.read()

        s1 = StructType([
            StructField("id", IntegerType(), True),
            StructField(
                "people",
                ArrayType(
                    StructType([
                        StructField("name", StringType(), True),
                        StructField("age", IntegerType(), True),
                    ]),
                    True,
                ),
                True,
            ),
        ])
        s2 = StructType([
            StructField("id", IntegerType(), True),
            StructField(
                "people",
                ArrayType(
                    StructType([
                        StructField("name", StringType(), True),
                        StructField("height", DoubleType(), True),
                    ]),
                    True,
                ),
                True,
            ),
        ])
        result = create_schema_comparison_tree(s1, s2, ignore_nullable=False, ignore_metadata=False)
        assert repr(result) + "\n" == expected

    def it_shows_diff_for_map_value_type_mismatch():
        with open("tests/data/tree_string/map_value_type_mismatch.txt") as f:
            expected = f.read()

        s1 = StructType([
            StructField("id", IntegerType(), True),
            StructField("props", MapType(StringType(), IntegerType(), True), True),
        ])
        s2 = StructType([
            StructField("id", IntegerType(), True),
            StructField("props", MapType(StringType(), DoubleType(), True), True),
        ])
        result = create_schema_comparison_tree(s1, s2, ignore_nullable=False, ignore_metadata=False)
        assert repr(result) + "\n" == expected

    def it_shows_diff_for_map_valueContainsNull_mismatch():
        with open("tests/data/tree_string/map_valueContainsNull_mismatch.txt") as f:
            expected = f.read()

        s1 = StructType([
            StructField("id", IntegerType(), True),
            StructField("props", MapType(StringType(), IntegerType(), True), True),
        ])
        s2 = StructType([
            StructField("id", IntegerType(), True),
            StructField("props", MapType(StringType(), IntegerType(), False), True),
        ])
        result = create_schema_comparison_tree(s1, s2, ignore_nullable=False, ignore_metadata=False)
        assert repr(result) + "\n" == expected

    def it_shows_diff_for_map_with_struct_values():
        with open("tests/data/tree_string/map_with_struct_values_diff.txt") as f:
            expected = f.read()

        s1 = StructType([
            StructField("id", IntegerType(), True),
            StructField(
                "data",
                MapType(
                    StringType(),
                    StructType([
                        StructField("x", IntegerType(), True),
                        StructField("y", IntegerType(), True),
                    ]),
                    True,
                ),
                True,
            ),
        ])
        s2 = StructType([
            StructField("id", IntegerType(), True),
            StructField(
                "data",
                MapType(
                    StringType(),
                    StructType([
                        StructField("x", IntegerType(), True),
                        StructField("z", DoubleType(), True),
                    ]),
                    True,
                ),
                True,
            ),
        ])
        result = create_schema_comparison_tree(s1, s2, ignore_nullable=False, ignore_metadata=False)
        assert repr(result) + "\n" == expected

    def it_shows_diff_for_nested_array_of_arrays():
        with open("tests/data/tree_string/nested_array_of_arrays_diff.txt") as f:
            expected = f.read()

        s1 = StructType([
            StructField("id", IntegerType(), True),
            StructField("matrix", ArrayType(ArrayType(IntegerType(), True), True), True),
        ])
        s2 = StructType([
            StructField("id", IntegerType(), True),
            StructField("matrix", ArrayType(ArrayType(DoubleType(), False), True), True),
        ])
        result = create_schema_comparison_tree(s1, s2, ignore_nullable=False, ignore_metadata=False)
        assert repr(result) + "\n" == expected

    def it_shows_diff_for_decimal_precision_in_array_and_map_of_structs():
        with open("tests/data/tree_string/decimal_precision_in_array_and_map.txt") as f:
            expected = f.read()

        s1 = StructType([
            StructField("id", IntegerType(), True),
            StructField(
                "items",
                ArrayType(
                    StructType([
                        StructField("name", StringType(), True),
                        StructField("price", DecimalType(10, 2), True),
                    ]),
                    True,
                ),
                True,
            ),
            StructField(
                "totals",
                MapType(
                    StringType(),
                    StructType([
                        StructField("amount", DecimalType(10, 2), True),
                        StructField("tax", DecimalType(8, 2), True),
                    ]),
                    True,
                ),
                True,
            ),
        ])
        s2 = StructType([
            StructField("id", IntegerType(), True),
            StructField(
                "items",
                ArrayType(
                    StructType([
                        StructField("name", StringType(), True),
                        StructField("price", DecimalType(12, 4), True),
                    ]),
                    True,
                ),
                True,
            ),
            StructField(
                "totals",
                MapType(
                    StringType(),
                    StructType([
                        StructField("amount", DecimalType(10, 2), True),
                        StructField("tax", DecimalType(10, 4), True),
                    ]),
                    True,
                ),
                True,
            ),
        ])
        result = create_schema_comparison_tree(s1, s2, ignore_nullable=False, ignore_metadata=False)
        assert repr(result) + "\n" == expected

    def it_handles_empty_vs_nonempty_schema():
        with open("tests/data/tree_string/empty_vs_nonempty_schema.txt") as f:
            expected = f.read()

        s1 = StructType([])
        s2 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
        ])
        result = create_schema_comparison_tree(s1, s2, ignore_nullable=False, ignore_metadata=False)
        assert repr(result) + "\n" == expected

    def it_shows_diff_for_mismatched_complex_kinds():
        with open("tests/data/tree_string/mismatched_complex_kinds.txt") as f:
            expected = f.read()

        s1 = StructType([
            StructField("id", IntegerType(), True),
            StructField(
                "data",
                ArrayType(
                    StructType([
                        StructField("x", IntegerType(), True),
                        StructField("y", IntegerType(), True),
                    ]),
                    True,
                ),
                True,
            ),
        ])
        s2 = StructType([
            StructField("id", IntegerType(), True),
            StructField(
                "data",
                MapType(
                    StringType(),
                    StructType([
                        StructField("a", DoubleType(), True),
                        StructField("b", DoubleType(), True),
                    ]),
                    True,
                ),
                True,
            ),
        ])
        result = create_schema_comparison_tree(s1, s2, ignore_nullable=False, ignore_metadata=False)
        assert repr(result) + "\n" == expected


def describe_assert_schema_equality_ignore_nullable():
    def it_has_good_error_messages_for_different_sized_schemas():
        s1 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
        ])
        s2 = StructType([
            StructField("name", StringType(), False),
            StructField("age", IntegerType(), True),
            StructField("something", IntegerType(), True),
            StructField("else", IntegerType(), True),
        ])
        with pytest.raises(SchemasNotEqualError):
            assert_schema_equality_ignore_nullable(s1, s2)

    def it_does_nothing_when_equal():
        s1 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
        ])
        s2 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
        ])
        assert_schema_equality_ignore_nullable(s1, s2)

    def it_does_nothing_when_only_nullable_flag_is_different():
        s1 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
        ])
        s2 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), False),
        ])
        assert_schema_equality_ignore_nullable(s1, s2)


def describe_are_schemas_equal_ignore_nullable():
    def it_returns_true_when_only_nullable_flag_is_different():
        s1 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("coords", ArrayType(DoubleType(), True), True),
        ])
        s2 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), False),
            StructField("coords", ArrayType(DoubleType(), True), False),
        ])
        assert are_schemas_equal(s1, s2, ignore_nullable=True) is True

    def it_returns_true_when_only_nullable_flag_is_different_within_array_element():
        s1 = StructType([StructField("coords", ArrayType(DoubleType(), True), True)])
        s2 = StructType([StructField("coords", ArrayType(DoubleType(), False), True)])
        assert are_schemas_equal(s1, s2, ignore_nullable=True) is True

    def it_returns_true_when_only_nullable_flag_is_different_within_nested_array_element():
        s1 = StructType([StructField("coords", ArrayType(ArrayType(DoubleType(), True), True), True)])
        s2 = StructType([StructField("coords", ArrayType(ArrayType(DoubleType(), False), True), True)])
        assert are_schemas_equal(s1, s2, ignore_nullable=True) is True

    def it_returns_false_when_the_element_type_is_different_within_array():
        s1 = StructType([StructField("coords", ArrayType(DoubleType(), True), True)])
        s2 = StructType([StructField("coords", ArrayType(IntegerType(), True), True)])
        assert are_schemas_equal(s1, s2, ignore_nullable=True) is False

    def it_returns_false_when_column_names_differ():
        s1 = StructType([
            StructField("blah", StringType(), True),
            StructField("age", IntegerType(), True),
        ])
        s2 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), False),
        ])
        assert are_schemas_equal(s1, s2, ignore_nullable=True) is False

    def it_returns_false_when_columns_have_different_order():
        s1 = StructType([
            StructField("blah", StringType(), True),
            StructField("age", IntegerType(), True),
        ])
        s2 = StructType([
            StructField("age", IntegerType(), False),
            StructField("blah", StringType(), True),
        ])
        assert are_schemas_equal(s1, s2, ignore_nullable=True) is False


def describe_are_schemas_equal_ignore_nullable_and_ignore_metadata():
    def it_returns_true_when_nullable_flag_is_different_within_array_element_with_both_flags():
        s1 = StructType([StructField("coords", ArrayType(DoubleType(), True), True)])
        s2 = StructType([StructField("coords", ArrayType(DoubleType(), False), True)])
        assert are_schemas_equal(s1, s2, ignore_nullable=True, ignore_metadata=True) is True

    def it_returns_true_when_metadata_is_different_on_field_containing_array_with_both_flags():
        s1 = StructType([StructField("coords", ArrayType(DoubleType(), True), True, {"foo": "bar"})])
        s2 = StructType([StructField("coords", ArrayType(DoubleType(), True), True, {"foo": "baz"})])
        assert are_schemas_equal(s1, s2, ignore_nullable=True, ignore_metadata=True) is True

    def it_returns_true_when_metadata_is_different_on_struct_fields_within_array_with_both_flags():
        s1 = StructType([
            StructField(
                "people",
                ArrayType(
                    StructType([
                        StructField("name", StringType(), True, {"comment": "first"}),
                        StructField("age", IntegerType(), True),
                    ]),
                    True,
                ),
                True,
            )
        ])
        s2 = StructType([
            StructField(
                "people",
                ArrayType(
                    StructType([
                        StructField("name", StringType(), True, {"comment": "second"}),
                        StructField("age", IntegerType(), True),
                    ]),
                    True,
                ),
                True,
            )
        ])
        assert are_schemas_equal(s1, s2, ignore_nullable=True, ignore_metadata=True) is True

    def it_returns_true_when_metadata_is_different_on_field_containing_nested_array_with_both_flags():
        s1 = StructType([StructField("matrix", ArrayType(ArrayType(IntegerType(), True), True), True, {"foo": "bar"})])
        s2 = StructType([StructField("matrix", ArrayType(ArrayType(IntegerType(), True), True), True, {"foo": "baz"})])
        assert are_schemas_equal(s1, s2, ignore_nullable=True, ignore_metadata=True) is True

    def it_returns_true_when_nullable_is_different_on_struct_fields_within_array_with_both_flags():
        s1 = StructType([
            StructField(
                "people",
                ArrayType(
                    StructType([
                        StructField("name", StringType(), True),
                        StructField("age", IntegerType(), True),
                    ]),
                    True,
                ),
                True,
            )
        ])
        s2 = StructType([
            StructField(
                "people",
                ArrayType(
                    StructType([
                        StructField("name", StringType(), False),
                        StructField("age", IntegerType(), True),
                    ]),
                    True,
                ),
                True,
            )
        ])
        assert are_schemas_equal(s1, s2, ignore_nullable=True, ignore_metadata=True) is True


def describe_are_structfields_equal():
    def it_returns_true_when_only_nullable_flag_is_different_within_array_element():
        s1 = StructField("coords", ArrayType(DoubleType(), True), True)
        s2 = StructField("coords", ArrayType(DoubleType(), False), True)
        assert are_structfields_equal(s1, s2, True) is True

    def it_returns_false_when_the_element_type_is_different_within_array():
        s1 = StructField("coords", ArrayType(DoubleType(), True), True)
        s2 = StructField("coords", ArrayType(IntegerType(), True), True)
        assert are_structfields_equal(s1, s2, True) is False

    def it_returns_true_when_the_element_type_is_same_within_struct():
        s1 = StructField("coords", StructType([StructField("hello", DoubleType(), True)]), True)
        s2 = StructField("coords", StructType([StructField("hello", DoubleType(), True)]), True)
        assert are_structfields_equal(s1, s2, True) is True

    def it_returns_false_when_the_element_type_is_different_within_struct():
        s1 = StructField("coords", StructType([StructField("hello", DoubleType(), True)]), True)
        s2 = StructField("coords", StructType([StructField("hello", IntegerType(), True)]), True)
        assert are_structfields_equal(s1, s2, True) is False

    def it_returns_false_when_the_element_name_is_different_within_struct():
        s1 = StructField("coords", StructType([StructField("hello", DoubleType(), True)]), True)
        s2 = StructField("coords", StructType([StructField("world", DoubleType(), True)]), True)
        assert are_structfields_equal(s1, s2, True) is False

    def it_returns_true_when_different_nullability_within_struct():
        s1 = StructField("coords", StructType([StructField("hello", DoubleType(), True)]), True)
        s2 = StructField("coords", StructType([StructField("hello", DoubleType(), False)]), True)
        assert are_structfields_equal(s1, s2, True) is True

    def it_returns_false_when_metadata_differs():
        s1 = StructField("coords", StringType(), True, {"hi": "whatever"})
        s2 = StructField("coords", StringType(), True, {"hi": "no"})
        assert are_structfields_equal(s1, s2, ignore_nullability=True, ignore_metadata=False) is False

    def it_allows_metadata_to_be_ignored():
        s1 = StructField("coords", StringType(), True, {"hi": "whatever"})
        s2 = StructField("coords", StringType(), True, {"hi": "no"})
        assert are_structfields_equal(s1, s2, ignore_nullability=False, ignore_metadata=True) is True

    def it_allows_nullability_and_metadata_to_be_ignored():
        s1 = StructField("coords", StringType(), True, {"hi": "whatever"})
        s2 = StructField("coords", StringType(), False, {"hi": "no"})
        assert are_structfields_equal(s1, s2, ignore_nullability=True, ignore_metadata=True) is True

    def it_returns_true_when_metadata_is_the_same():
        s1 = StructField("coords", StringType(), True, {"hi": "whatever"})
        s2 = StructField("coords", StringType(), True, {"hi": "whatever"})
        assert are_structfields_equal(s1, s2, ignore_nullability=True, ignore_metadata=False) is True

    def it_returns_true_when_only_nullable_flag_is_different_for_map_values():
        s1 = StructField("coords", MapType(StringType(), DoubleType(), True), True)
        s2 = StructField("coords", MapType(StringType(), DoubleType(), False), True)
        assert are_structfields_equal(s1, s2, ignore_nullability=True) is True

    def it_returns_false_when_only_nullable_flag_is_different_for_map_values():
        s1 = StructField("coords", MapType(StringType(), DoubleType(), True), True)
        s2 = StructField("coords", MapType(StringType(), DoubleType(), False), True)
        assert are_structfields_equal(s1, s2) is False

    def it_returns_false_when_nullable_differs_and_only_ignore_metadata_is_true():
        """Bug fix: ignore_metadata=True should not cause nullable differences to be ignored."""
        s1 = StructField("x", IntegerType(), True)
        s2 = StructField("x", IntegerType(), False)
        assert are_structfields_equal(s1, s2, ignore_nullability=False, ignore_metadata=True) is False

    def it_returns_false_when_array_containsNull_differs_and_ignore_nullable_is_false():
        """Bug fix: ArrayType.containsNull should be checked when ignore_nullable=False."""
        s1 = StructField("coords", ArrayType(DoubleType(), True), True)
        s2 = StructField("coords", ArrayType(DoubleType(), False), True)
        assert are_structfields_equal(s1, s2, ignore_nullability=False, ignore_metadata=True) is False

    def it_returns_true_when_array_containsNull_differs_and_ignore_nullable_is_true():
        s1 = StructField("coords", ArrayType(DoubleType(), True), True)
        s2 = StructField("coords", ArrayType(DoubleType(), False), True)
        assert are_structfields_equal(s1, s2, ignore_nullability=True, ignore_metadata=False) is True

    def it_returns_false_when_map_valueContainsNull_differs_and_only_ignore_metadata_is_true():
        """Bug fix: MapType.valueContainsNull is a nullable property, not metadata."""
        s1 = StructField("m", MapType(StringType(), DoubleType(), True), True)
        s2 = StructField("m", MapType(StringType(), DoubleType(), False), True)
        assert are_structfields_equal(s1, s2, ignore_nullability=False, ignore_metadata=True) is False

    def it_returns_false_when_nested_struct_nullable_differs_and_only_ignore_metadata_is_true():
        """Bug fix: nested StructField nullable should be checked when ignore_nullable=False."""
        s1 = StructField("outer", StructType([StructField("inner", IntegerType(), True)]), True)
        s2 = StructField("outer", StructType([StructField("inner", IntegerType(), False)]), True)
        assert are_structfields_equal(s1, s2, ignore_nullability=False, ignore_metadata=True) is False

    def it_returns_true_when_nested_struct_nullable_differs_and_ignore_nullable_is_true():
        s1 = StructField("outer", StructType([StructField("inner", IntegerType(), True)]), True)
        s2 = StructField("outer", StructType([StructField("inner", IntegerType(), False)]), True)
        assert are_structfields_equal(s1, s2, ignore_nullability=True, ignore_metadata=False) is True


def describe_are_schemas_equal():
    def it_returns_true_for_identical_schemas():
        s1 = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
        ])
        assert are_schemas_equal(s1, s1) is True

    def it_returns_false_when_nullable_differs_with_no_flags():
        s1 = StructType([StructField("name", StringType(), True)])
        s2 = StructType([StructField("name", StringType(), False)])
        assert are_schemas_equal(s1, s2) is False

    def it_returns_true_when_nullable_differs_with_ignore_nullable():
        s1 = StructType([StructField("name", StringType(), True)])
        s2 = StructType([StructField("name", StringType(), False)])
        assert are_schemas_equal(s1, s2, ignore_nullable=True) is True

    def it_returns_false_when_nullable_differs_with_only_ignore_metadata():
        s1 = StructType([StructField("name", StringType(), True)])
        s2 = StructType([StructField("name", StringType(), False)])
        assert are_schemas_equal(s1, s2, ignore_metadata=True) is False

    def it_returns_true_when_metadata_differs_with_ignore_metadata():
        s1 = StructType([StructField("name", StringType(), True, {"a": "1"})])
        s2 = StructType([StructField("name", StringType(), True, {"a": "2"})])
        assert are_schemas_equal(s1, s2, ignore_metadata=True) is True

    def it_returns_false_when_array_containsNull_differs_with_only_ignore_metadata():
        s1 = StructType([StructField("arr", ArrayType(IntegerType(), True), True)])
        s2 = StructType([StructField("arr", ArrayType(IntegerType(), False), True)])
        assert are_schemas_equal(s1, s2, ignore_metadata=True) is False

    def it_returns_true_when_array_containsNull_differs_with_ignore_nullable():
        s1 = StructType([StructField("arr", ArrayType(IntegerType(), True), True)])
        s2 = StructType([StructField("arr", ArrayType(IntegerType(), False), True)])
        assert are_schemas_equal(s1, s2, ignore_nullable=True) is True

    def it_returns_false_when_nested_struct_nullable_differs_with_only_ignore_metadata():
        s1 = StructType([StructField("s", StructType([StructField("x", IntegerType(), True)]), True)])
        s2 = StructType([StructField("s", StructType([StructField("x", IntegerType(), False)]), True)])
        assert are_schemas_equal(s1, s2, ignore_metadata=True) is False

    def it_returns_false_when_map_valueContainsNull_differs_with_only_ignore_metadata():
        s1 = StructType([StructField("m", MapType(StringType(), IntegerType(), True), True)])
        s2 = StructType([StructField("m", MapType(StringType(), IntegerType(), False), True)])
        assert are_schemas_equal(s1, s2, ignore_metadata=True) is False

    def it_returns_true_when_map_valueContainsNull_differs_with_ignore_nullable():
        s1 = StructType([StructField("m", MapType(StringType(), IntegerType(), True), True)])
        s2 = StructType([StructField("m", MapType(StringType(), IntegerType(), False), True)])
        assert are_schemas_equal(s1, s2, ignore_nullable=True) is True


def describe_print_schema_diff():
    def it_prints_a_table(capsys):
        s1 = StructType([StructField("name", StringType(), True)])
        s2 = StructType([StructField("nombre", StringType(), True)])
        print_schema_diff(s1, s2, ignore_nullable=False, ignore_metadata=False)
        captured = capsys.readouterr()
        assert "schema1" in captured.out
        assert "schema2" in captured.out

    def it_prints_a_tree(capsys):
        s1 = StructType([StructField("name", StringType(), True)])
        s2 = StructType([StructField("nombre", StringType(), True)])
        print_schema_diff(s1, s2, ignore_nullable=False, ignore_metadata=False, output_format=OutputFormat.TREE)
        captured = capsys.readouterr()
        assert "|-- name: string" in captured.out
        assert "|-- nombre: string" in captured.out

    def it_throws_with_an_unknown_output_format():
        s1 = StructType([StructField("name", StringType(), True)])
        s2 = StructType([StructField("name", StringType(), True)])
        with pytest.raises(ValueError):
            print_schema_diff(s1, s2, ignore_nullable=False, ignore_metadata=False, output_format="invalid")


def describe_create_schema_comparison_tree():
    def it_handles_empty_schemas():
        result = create_schema_comparison_tree(StructType([]), StructType([]), False, False)
        assert "schema1" in result
        assert "schema2" in result

    def it_marks_leaf_types_with_different_type_names_as_unequal():
        s1 = StructType([StructField("num", IntegerType(), True)])
        s2 = StructType([StructField("num", StringType(), True)])
        line = _tree_line(create_schema_comparison_tree(s1, s2, False, False), "|-- num: int")
        assert "|-- num: string" in line
        assert line.startswith(bcolors.LightRed)

    def it_marks_fields_with_different_nullability_as_unequal():
        s1 = StructType([StructField("name", StringType(), True)])
        s2 = StructType([StructField("name", StringType(), False)])
        line = _tree_line(create_schema_comparison_tree(s1, s2, False, False), "|-- name: string")
        assert "(nullable = true)" in line
        assert "(nullable = false)" in line
        assert line.startswith(bcolors.LightRed)

    def it_marks_maps_with_different_key_types_as_unequal():
        s1 = StructType([StructField("m", MapType(StringType(), IntegerType(), True), True)])
        s2 = StructType([StructField("m", MapType(IntegerType(), IntegerType(), True), True)])
        result = create_schema_comparison_tree(s1, s2, False, False)
        key_line = _tree_line(result, "|-- key:")
        assert "|-- key: string" in key_line
        assert "|-- key: int" in key_line
        assert key_line.startswith(bcolors.LightRed)
        # only the key type differs, so the value line is still marked as equal
        assert _tree_line(result, "|-- value:").startswith(bcolors.LightBlue)


def describe_are_fields_shallow_equal():
    def it_returns_true_when_both_fields_are_missing():
        assert _are_fields_shallow_equal(None, None, False, False) is True

    def it_returns_false_when_only_one_field_is_missing():
        sf = StructField("name", StringType(), True)
        assert _are_fields_shallow_equal(sf, None, False, False) is False


def describe_compare_array_types():
    def it_marks_the_element_line_as_unequal_when_one_side_is_missing():
        lines: list[tuple[str, str, bool]] = []
        _compare_array_types(ArrayType(StringType(), True), None, False, False, 0, lines)
        assert len(lines) == 1
        left, right, is_equal = lines[0]
        assert "|-- element: string" in left
        assert right == ""
        assert is_equal is False


def describe_compare_map_types():
    def it_marks_the_key_and_value_lines_as_unequal_when_one_side_is_missing():
        lines: list[tuple[str, str, bool]] = []
        _compare_map_types(None, MapType(StringType(), IntegerType(), True), False, False, 0, lines)
        assert len(lines) == 2
        key_line, value_line = lines
        assert key_line[0] == ""
        assert "|-- key: string" in key_line[1]
        assert key_line[2] is False
        assert value_line[0] == ""
        assert "|-- value: int" in value_line[1]
        assert value_line[2] is False


def describe_handle_schemas_not_equal():
    def it_uses_a_tree_for_wide_schemas():
        s1 = StructType([StructField("nums", ArrayType(IntegerType(), True), True)])
        s2 = StructType([StructField("nums", ArrayType(StringType(), True), True)])
        with pytest.raises(SchemasNotEqualError) as exc_info:
            assert_schema_equality(s1, s2)
        assert "|-- element: int" in str(exc_info.value)


def describe_assert_schema_equality_full():
    def it_throws_when_schema_lengths_differ_and_nullability_is_ignored():
        s1 = StructType([StructField("name", StringType(), True)])
        s2 = StructType([
            StructField("name", StringType(), False),
            StructField("age", IntegerType(), True),
        ])
        with pytest.raises(SchemasNotEqualError):
            assert_schema_equality(s1, s2, ignore_nullable=True)


def describe_are_structfields_equal_with_missing_fields():
    def it_returns_true_when_both_are_none_and_nullability_is_ignored():
        assert are_structfields_equal(None, None, ignore_nullability=True) is True

    def it_returns_false_when_only_one_is_none_and_nullability_is_ignored():
        sf = StructField("name", StringType(), True)
        assert are_structfields_equal(sf, None, ignore_nullability=True) is False
