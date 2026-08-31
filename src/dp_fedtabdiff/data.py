"""Minimal mixed-type preprocessing and deterministic client partitioning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler


@dataclass
class CategoryEncoder:
    """Per-column category vocabulary with an explicit unknown token."""

    vocabularies: dict[str, dict[str, int]]

    @classmethod
    def fit(cls, rows: list[dict[str, object]], columns: list[str]) -> CategoryEncoder:
        vocabularies = {}
        for column in columns:
            values = sorted({str(row[column]) for row in rows})
            vocabularies[column] = {value: index for index, value in enumerate(values)}
            vocabularies[column]["<unknown>"] = len(values)
        return cls(vocabularies)

    def transform(self, rows: list[dict[str, object]], columns: list[str]) -> np.ndarray:
        return np.asarray(
            [
                [
                    self.vocabularies[column].get(
                        str(row[column]), self.vocabularies[column]["<unknown>"]
                    )
                    for column in columns
                ]
                for row in rows
            ],
            dtype=np.int64,
        )


def iid_partition(n_rows: int, n_clients: int, seed: int = 0) -> list[np.ndarray]:
    """Split row indices into similarly sized, deterministic IID partitions."""
    if n_rows < 1 or n_clients < 1 or n_clients > n_rows:
        raise ValueError("n_rows and n_clients must define non-empty partitions")
    indices = np.arange(n_rows)
    np.random.default_rng(seed).shuffle(indices)
    return [part.copy() for part in np.array_split(indices, n_clients)]


@dataclass
class DataTransformer:
    """Fit mixed-type preprocessing on training data and reuse it safely."""

    categorical_columns: list[str]
    numerical_columns: list[str]
    numerical_scaler: str = "standard"
    category_encoder: CategoryEncoder | None = None
    numerical_transformer: object | None = None

    @property
    def categorical_cols(self) -> list[str]:
        """Compatibility name used by the canonical FinDiff API."""
        return self.categorical_columns

    @property
    def numerical_cols(self) -> list[str]:
        """Compatibility name used by the canonical FinDiff API."""
        return self.numerical_columns

    @property
    def categorical_mapping_(self) -> dict[str, list[int]]:
        """Return global token indices grouped by categorical column."""
        if self.category_encoder is None:
            raise RuntimeError("fit must be called before reading categorical_mapping_")
        offset = 0
        mapping = {}
        for column in self.categorical_columns:
            size = len(self.category_encoder.vocabularies[column])
            mapping[column] = list(range(offset, offset + size))
            offset += size
        return mapping

    @property
    def embedding_mapping_(self) -> dict[str, np.ndarray]:
        """Alias matching the reference FinDiff transformer."""
        return {column: np.asarray(indices) for column, indices in self.categorical_mapping_.items()}

    @property
    def label_cardinality_(self) -> int | None:
        """Optional label cardinality placeholder for FinDiff compatibility."""
        return None

    def fit(self, frame: pd.DataFrame) -> DataTransformer:
        missing = set(self.categorical_columns + self.numerical_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        rows = frame[self.categorical_columns].astype(str).to_dict("records")
        self.category_encoder = CategoryEncoder.fit(rows, self.categorical_columns)
        if self.numerical_scaler == "standard":
            self.numerical_transformer = StandardScaler()
        elif self.numerical_scaler == "quantile":
            self.numerical_transformer = QuantileTransformer(
                output_distribution="normal", random_state=0
            )
        elif self.numerical_scaler == "none":
            self.numerical_transformer = None
        else:
            raise ValueError("numerical_scaler must be standard, quantile, or none")
        if self.numerical_transformer is not None:
            self.numerical_transformer.fit(frame[self.numerical_columns].astype(float))
        return self

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        if self.category_encoder is None:
            raise RuntimeError("fit must be called before transform")
        categorical = self.category_encoder.transform(
            frame[self.categorical_columns].astype(str).to_dict("records"),
            self.categorical_columns,
        )
        numerical_frame = frame[self.numerical_columns].astype(float)
        if self.numerical_transformer is not None:
            numerical = self.numerical_transformer.transform(numerical_frame).astype(np.float32)
        else:
            numerical = numerical_frame.to_numpy(dtype=np.float32)
        return {"cat": categorical, "num": numerical}

    def fit_transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return self.fit(frame).transform(frame)

    def inverse_transform(self, categorical: np.ndarray, numerical: np.ndarray) -> pd.DataFrame:
        if self.category_encoder is None:
            raise RuntimeError("fit must be called before inverse_transform")
        decoded = {}
        for index, column in enumerate(self.categorical_columns):
            inverse = {value: key for key, value in self.category_encoder.vocabularies[column].items()}
            decoded[column] = [inverse.get(int(value), "<unknown>") for value in categorical[:, index]]
        if self.numerical_transformer is not None:
            numerical = self.numerical_transformer.inverse_transform(
                pd.DataFrame(numerical, columns=self.numerical_columns)
            )
        for index, column in enumerate(self.numerical_columns):
            decoded[column] = numerical[:, index]
        return pd.DataFrame(decoded)
