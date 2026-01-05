# dataloader.py

import logging
import pandas as pd

from haashi_pkg.utility.utils import Utility

from pandas import DataFrame
from typing import Iterable, List, Optional


class DataLoader:
    """
    Lightweight data ingestion utility for loading tabular data formats.
    Focused strictly on I/O, not validation or transformation.
    """

    def __init__(self, *file_paths: str) -> None:
        """Initialize loader with one or more file paths."""
        self.file_paths: List[str] = list(file_paths)
        self.ut = Utility(level=logging.WARNING)

    # ==========================
    # Load CSV
    # ==========================

    def load_csv_single(
        self,
        skip_rows: int = 0,
        header_row: int = 0,
    ) -> Optional[DataFrame]:
        """Load a single CSV file into a DataFrame."""
        try:
            return pd.read_csv(
                self.file_paths[0],
                sep=None,
                engine="python",
                skiprows=skip_rows,
                header=header_row,
            )
        except FileNotFoundError as fnf:
            self.ut.handle_file_not_found(fnf)
        except Exception as exc:
            self.ut.handle_error(exc)

    def load_csv_many(
        self,
        skip_rows: int = 0,
        header_row: int = 0,
    ) -> Optional[List[DataFrame]]:
        """Load multiple CSV files into a list of DataFrames."""
        try:
            return [
                pd.read_csv(
                    path,
                    sep=None,
                    engine="python",
                    skiprows=skip_rows,
                    header=header_row,
                )
                for path in self.file_paths
            ]
        except FileNotFoundError as fnf:
            self.ut.handle_file_not_found(fnf)
        except Exception as exc:
            self.ut.handle_error(exc)

    def load_csv_chunk(
        self,
        skip_rows: int = 0,
        header_row: int = 0,
        chunk_size: int = 1000,
    ) -> Optional[Iterable[DataFrame]]:
        """Stream a large CSV file in iterable chunks."""
        try:
            return pd.read_csv(
                self.file_paths[0],
                sep=None,
                engine="python",
                skiprows=skip_rows,
                header=header_row,
                chunksize=chunk_size,
            )
        except FileNotFoundError as fnf:
            self.ut.handle_file_not_found(fnf)
        except Exception as exc:
            self.ut.handle_error(exc)

    # ==========================
    # Load Excel
    # ==========================
    # COMING SOON

    # ==========================
    # Load JSON
    # ==========================
    # COMING SOON

    # ==========================
    # Load Parquet
    # ==========================

    def load_parquet(self) -> Optional[DataFrame]:
        """Load a single Parquet file into a DataFrame."""
        try:
            return pd.read_parquet(
                self.file_paths[0], engine="pyarrow"
            )
        except FileNotFoundError as fnf:
            self.ut.handle_file_not_found(fnf)
        except Exception as exc:
            self.ut.handle_error(exc)

    def load_parquet_many(self) -> Optional[List[DataFrame]]:
        """Load multiple Parquet files into a list of DataFrames."""
        try:
            return [
                pd.read_parquet(
                    path, engine="pyarrow"
                )
                for path in self.file_paths
            ]
        except FileNotFoundError as fnf:
            self.ut.handle_file_not_found(fnf)
        except Exception as exc:
            self.ut.handle_error(exc)