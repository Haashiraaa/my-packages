from __future__ import annotations

import sys
import pandas as pd
import logging
from pandas import DataFrame, Series, Index
from haashi_pkg.utility.utils import Utility
from typing import (
    List,
    Optional,
    Sequence,
    Union,
    Iterable,
    Any,
    Tuple,
    Dict,
    Literal,
)


# -------------------------
# Global config
# -------------------------

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

ut = Utility(level=logging.INFO)

# -------------------------
# Type aliases
# -------------------------

Column = Union[str, Sequence[str]]
AggOp = Union[str, Sequence[str]]
MissingStats = Tuple[int, int, float]


class DataEngine:
    """
    DataEngine provides reusable utilities for loading, inspecting,
    cleaning, transforming, and summarizing tabular data using pandas.

    Philosophy:
    - Inspection does NOT mutate data
    - Cleaning returns new objects
    - Conversions are explicit
    """

    def __init__(self, *file_paths: str, save_path: Optional[str] = None) -> None:
        self.file_paths: List[str] = list(file_paths)
        self.save_path: Optional[str] = save_path
        self.dropped_row_count: int = 0
        self.cummulative_missing: int = 0

    # =====================================================
    # Loading
    # =====================================================

    """
    Load one or more CSV files.
                                                                                - Multiple files → list of DataFrames
    - Single file + chunksize → iterator
    """

    def load_csv_single(
        self,
        skip_rows: int = 0,
        header_row: int = 0,
    ) -> DataFrame:

        try:

            return pd.read_csv(  # type: ignore
                self.file_paths[0],
                sep=None,
                engine="python",
                skiprows=skip_rows,
                header=header_row,
            )

        except FileNotFoundError as fnf:
            print(ut.text["MISSING_FILE"])
            ut.debug(fnf)
            sys.exit(1)

        except Exception as exc:
            print(ut.text["ERROR"])
            ut.debug(exc)
            sys.exit(1)

    def load_csv_many(
            self, skip_rows: int = 0, header_row: int = 0
    ) -> List[DataFrame]:
        try:
            return [
                pd.read_csv(  # type: ignore
                    path,
                    sep=None,
                    engine="python",
                    skiprows=skip_rows,
                    header=header_row,
                ) for path in self.file_paths
            ]
        except FileNotFoundError as fnf:
            print(ut.text["MISSING_FILE"])
            ut.debug(fnf)
            sys.exit(1)

        except Exception as exc:
            print(ut.text["ERROR"])
            ut.debug(exc)
            sys.exit(1)

    def load_csv_chunk(
        self,
        skip_rows: int = 0,
        header_row: int = 0,
        chunk_size: int = 1000,
    ) -> Iterable[DataFrame]:
        try:
            return pd.read_csv(  # type: ignore
                self.file_paths[0],
                sep=None,
                engine="python",
                skiprows=skip_rows,
                header=header_row,
                chunksize=chunk_size,
            )
        except FileNotFoundError as fnf:
            print(ut.text["MISSING_FILE"])
            ut.debug(fnf)
            sys.exit(1)

        except Exception as exc:
            print(ut.text["ERROR"])
            ut.debug(exc)
            sys.exit(1)

    # =====================================================
    # Inspection (NO mutation)
    # =====================================================

    def inspect_dataframe(
        self,
        df: DataFrame,
        rows: int = 5,
        verbose: bool = True,
    ) -> None:
        """Display basic structural information about a DataFrame."""
        if verbose:
            print(df.head(rows))
            print(df.dtypes)  # type: ignore
            print(df.shape)

    def count_missing(
        self,
        df: DataFrame,
        columns: Column,
    ) -> Union[int, List[int]]:
        """Count missing values per column."""
        if isinstance(columns, (list, tuple)):
            return [int(df[col].isna().sum()) for col in columns]
        return int(df[columns].isna().sum())

    def count_duplicates(
        self,
        df: DataFrame,
        columns: Column,
    ) -> Union[int, List[int]]:
        """Count duplicate values per column."""
        if isinstance(columns, (list, tuple)):
            return [int(df[col].duplicated().sum()) for col in columns]
        return int(df[columns].duplicated().sum())

    def inspect_text_formatting(
        self,
        df: DataFrame,
        columns: str,
    ) -> Dict[str, Any]:
        """
        Inspect a text column for formatting inconsistencies
        WITHOUT modifying the data.

        Intended for:
        - categorical text
        - labels
        - regions
        """

        s = df[columns].astype(str)

        lowered = s.str.lower()

        return {
            "total_values_checked": len(s),
            "has_leading_trailing_whitespace": s.str.match(r"^\s|\s$").any(),
            "has_multiple_internal_spaces": s.str.contains(r"\s{2,}").any(),
            "has_tabs_or_newlines": s.str.contains(r"[\t\n\r]").any(),
            "has_case_inconsistency": lowered.nunique() < s.nunique(),
            "sample_values": {
                "whitespace": s[s.str.match(r"^\s|\s$")].head(5).tolist(),
                "multiple_spaces": s[s.str.contains(r"\s{2,}")].head(5).tolist(),
                "tabs_newlines": s[s.str.contains(r"[\t\n\r]")].head(5).tolist(),
                "case_variants": s[lowered.duplicated()].head(5).tolist(),
            },
        }

    # =====================================================
    # Type Conversion
    # =====================================================

    def convert_numeric(
        self,
        series: Series,
        integer: bool = False,
    ) -> Series:
        """
        Convert a Series to numeric.

        - Non-numeric characters are stripped
        - Errors coerced to NaN
        """
        cleaned = series.astype(str).str.replace(r"[^0-9.]", "", regex=True)

        if integer:
            return pd.to_numeric(cleaned, errors="coerce").astype("Int64")

        return pd.to_numeric(cleaned, errors="coerce")

    def convert_datetime(
        self,
        series: Series,
    ) -> Series:
        """Convert a Series to datetime using mixed formats."""
        return pd.to_datetime(series, errors="coerce", format="mixed")  # type: ignore

    # =====================================================
    # Normalization (mutation helpers)
    # =====================================================

    def normalize_column_names(self, df: DataFrame) -> DataFrame:
        """Lowercase and strip column names."""
        df = df.copy()
        df.columns = df.columns.str.lower().str.strip()
        return df

    def normalize_text_values(
        self,
        series: Series,
        method: str = "lower",
    ) -> Series:
        """
        Normalize string values (case + whitespace).
        """
        s = series.astype(str).str.strip()

        method = method.lower()
        if method == "lower":
            return s.str.lower()
        if method == "upper":
            return s.str.upper()
        if method == "title":
            return s.str.title()

        raise ValueError("Invalid normalization method")

    # =====================================================
    # Missing Data
    # =====================================================

    def missing_summary(
        self,
        df: DataFrame,
        column: str,
    ) -> MissingStats:
        """
        Return total rows, missing count, and missing percentage.
        """
        total = len(df)
        missing = int(df[column].isna().sum())
        percent = (missing / total) * 100
        self.cummulative_missing += missing
        return total, missing, percent

    def drop_rows_with_missing(
        self,
        df: DataFrame,
        columns: Sequence[str],
    ) -> DataFrame:
        """
        Drop rows with missing values in specified columns.
        """
        mask = df[columns].isna().any(axis=1)
        self.dropped_row_count += int(mask.sum())
        return df.loc[~mask].copy()

    def fill_missing_forward(
        self,
        series: Series,
    ) -> Series:
        """Forward-fill missing values."""
        return series.ffill()  # type: ignore

    def fill_missing_backward(
        self,
        series: Series,
    ) -> Series:
        """Backward-fill missing values."""
        return series.bfill()  # type: ignore

    # =====================================================
    # Aggregation
    # =====================================================

    def aggregate(
        self,
        df: DataFrame,
        value_col: str,
        group_cols: Union[str, List[str]],
        op: AggOp = "sum",
    ) -> Union[Series, DataFrame]:
        """
        Aggregate values using groupby.
        """
        gb = df.groupby(group_cols, observed=True)[value_col]  # type: ignore

        if isinstance(op, (list, tuple)):
            return gb.agg([o.lower() for o in op])  # type: ignore

        return getattr(gb, op.lower())()  # type: ignore

    # =====================================================
    # Merging
    # =====================================================

    def merge_dataframes(
        self,
        df1: DataFrame,
        df2: DataFrame,
        on: str,
        how: Literal["inner", "left", "right", "outer", "cross"] = "inner",
        validate: str = "1:1",
        **kwargs: Any,
    ) -> DataFrame:

        return pd.merge(
            df1, df2, on=on, how=how, validate=validate, **kwargs
        )

    # =====================================================
    # Saving
    # =====================================================

    def save_csv(
        self,
        df: DataFrame,
        path: Optional[str] = None,
    ) -> None:
        """Save DataFrame to CSV."""
        path = path or self.save_path
        if not path:
            raise ValueError("No save path provided")

        df.to_csv(path, index=False)  # type: ignore
        print(f"File saved → {path}")
