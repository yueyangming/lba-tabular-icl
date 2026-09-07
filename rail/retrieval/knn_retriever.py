from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class RetrievalResult:
    indices: list[int]
    distances: list[float]


class TabularKNNRetriever:
    def __init__(self) -> None:
        self.preprocessor: ColumnTransformer | None = None
        self.index: NearestNeighbors | None = None
        self._train_frame: pd.DataFrame | None = None
        self._train_matrix = None

    @staticmethod
    def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
        numeric_columns = list(X.select_dtypes(include=["number", "bool"]).columns)
        categorical_columns = [col for col in X.columns if col not in numeric_columns]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_columns),
                ("categorical", categorical_pipeline, categorical_columns),
            ],
            remainder="drop",
        )

    def fit(self, X_train: pd.DataFrame) -> "TabularKNNRetriever":
        self._train_frame = X_train.reset_index(drop=True)
        self.preprocessor = self._build_preprocessor(self._train_frame)
        transformed = self.preprocessor.fit_transform(self._train_frame)
        self._train_matrix = transformed
        # Candidate generation uses Euclidean distance on the standardized /
        # one-hot-expanded feature space. Later selector stages may rerank this
        # candidate pool with cosine similarity; that is a deliberate two-stage
        # design rather than an accidental metric mismatch.
        self.index = NearestNeighbors(metric="euclidean")
        self.index.fit(transformed)
        return self

    def retrieve(self, query: pd.Series | pd.DataFrame, *, k: int = 16) -> RetrievalResult:
        if self.preprocessor is None or self.index is None:
            raise ValueError("Retriever has not been fitted.")
        if self._train_frame is None:
            raise ValueError("Retriever has not been fitted.")

        if isinstance(query, pd.Series):
            query_frame = query.to_frame().T
        else:
            query_frame = query

        k = min(k, len(self._train_frame))
        transformed = self.preprocessor.transform(query_frame)
        distances, indices = self.index.kneighbors(transformed, n_neighbors=k)
        return RetrievalResult(
            indices=indices[0].tolist(),
            distances=[float(value) for value in distances[0]],
        )

    def transform_query(self, query: pd.Series | pd.DataFrame):
        if self.preprocessor is None:
            raise ValueError("Retriever has not been fitted.")
        if isinstance(query, pd.Series):
            query_frame = query.to_frame().T
        else:
            query_frame = query
        return self.preprocessor.transform(query_frame)

    def get_candidate_matrix(self, indices: list[int]):
        if self._train_matrix is None:
            raise ValueError("Retriever has not been fitted.")
        return self._train_matrix[indices]

    def retrieve_examples(
        self,
        query: pd.Series | pd.DataFrame,
        *,
        k: int = 16,
    ) -> tuple[pd.DataFrame, list[float]]:
        if self._train_frame is None:
            raise ValueError("Retriever has not been fitted.")
        result = self.retrieve(query, k=k)
        examples = self._train_frame.iloc[result.indices].reset_index(drop=True)
        return examples, result.distances
