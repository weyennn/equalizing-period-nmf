"""Core utilities for the global NMF period-weighting experiment.

The module keeps mathematical transformations separate from filesystem output so
that the fixed settings can be tested without running the full corpus.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import norm as sparse_norm
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class CoherenceResult:
    coherence_cv: float
    covered_features: int
    total_features: int
    coverage: float


def build_locked_nmf(
    n_topics: int = 20,
    max_iter: int = 400,
    random_state: int = 42,
) -> NMF:
    """Construct the one allowed Global NMF configuration."""
    return NMF(
        n_components=n_topics,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        max_iter=max_iter,
        random_state=random_state,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
    )


def split_train_validation(
    df: pd.DataFrame,
    validation_fraction: float = 0.20,
    random_state: int = 42,
    stratify_column: str = "year_month",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one reproducible month-stratified train/validation split."""
    if stratify_column not in df.columns:
        raise ValueError(f"Missing stratification column: {stratify_column}")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if df[stratify_column].isna().any():
        raise ValueError(f"{stratify_column} contains missing values")
    counts = df[stratify_column].value_counts()
    if (counts < 2).any():
        bad = counts[counts < 2].index.tolist()
        raise ValueError(f"Every stratum needs at least two rows; invalid strata: {bad}")

    train_idx, valid_idx = train_test_split(
        np.arange(len(df)),
        test_size=validation_fraction,
        random_state=random_state,
        stratify=df[stratify_column].to_numpy(),
        shuffle=True,
    )
    return df.iloc[train_idx].copy(), df.iloc[valid_idx].copy()


def compute_document_weights(months: pd.Series | Sequence[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute sqrt(n_bar / n_p) row weights and their energy audit table."""
    month_series = pd.Series(months, copy=False).astype(str)
    counts = month_series.value_counts().sort_index()
    if counts.empty or (counts <= 0).any():
        raise ValueError("Month counts must be positive")
    n_bar = float(counts.mean())
    by_month = np.sqrt(n_bar / counts.astype(float))
    weights = month_series.map(by_month).to_numpy(dtype=float)
    audit = pd.DataFrame(
        {
            "year_month": counts.index,
            "n_documents": counts.to_numpy(dtype=int),
            "n_bar": n_bar,
            "document_weight": by_month.to_numpy(dtype=float),
        }
    )
    audit["weighted_energy"] = audit["n_documents"] * audit["document_weight"] ** 2
    return weights, audit


def compute_temporal_idf(x_presence: sparse.spmatrix, months: pd.Series | Sequence[str]) -> np.ndarray:
    """Compute smooth inverse period frequency from train rows only."""
    matrix = sparse.csr_matrix(x_presence, copy=True)
    matrix.data = np.ones_like(matrix.data, dtype=float)
    month_values = pd.Series(months, copy=False).astype(str).to_numpy()
    if matrix.shape[0] != len(month_values):
        raise ValueError("Matrix rows and month labels differ")
    unique_months = np.sort(np.unique(month_values))
    period_presence = []
    for month in unique_months:
        rows = np.flatnonzero(month_values == month)
        period_presence.append(np.asarray(matrix[rows].sum(axis=0)).ravel() > 0)
    pf = np.vstack(period_presence).sum(axis=0)
    n_periods = len(unique_months)
    return np.log((1.0 + n_periods) / (1.0 + pf)) + 1.0


def apply_temporal_tfidf(x_base: sparse.spmatrix, temporal_idf: np.ndarray) -> sparse.csr_matrix:
    """Apply feature-wise temporal IDF and re-normalize every nonzero row to L2=1."""
    matrix = sparse.csr_matrix(x_base)
    temporal_idf = np.asarray(temporal_idf, dtype=float)
    if matrix.shape[1] != temporal_idf.size:
        raise ValueError("Temporal IDF length differs from matrix vocabulary")
    if not np.isfinite(temporal_idf).all() or (temporal_idf <= 0).any():
        raise ValueError("Temporal IDF must contain finite positive values")
    weighted = matrix.multiply(temporal_idf).tocsr()
    return normalize(weighted, norm="l2", axis=1, copy=False).tocsr()


def build_scenario_matrices(
    x_base_train: sparse.spmatrix,
    x_temporal_train: sparse.spmatrix,
    document_weights: np.ndarray,
) -> dict[str, sparse.csr_matrix]:
    """Build the four training matrices S1-S4."""
    x_base = sparse.csr_matrix(x_base_train)
    x_temporal = sparse.csr_matrix(x_temporal_train)
    weights = np.asarray(document_weights, dtype=float)
    if x_base.shape != x_temporal.shape:
        raise ValueError("Base and Temporal TF-IDF matrices must have equal shapes")
    if x_base.shape[0] != weights.size:
        raise ValueError("Document weight length differs from train rows")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Document weights must be finite and positive")
    diagonal = sparse.diags(weights)
    return {
        "S1_baseline": x_base,
        "S2_document_weighting": diagonal.dot(x_base).tocsr(),
        "S3_temporal_tfidf": x_temporal,
        "S4_dual_level": diagonal.dot(x_temporal).tocsr(),
    }


def compute_period_energy_distribution(
    matrix: sparse.spmatrix,
    months: pd.Series | Sequence[str],
) -> pd.DataFrame:
    """Return monthly squared Frobenius energy shares for a train matrix."""
    x = sparse.csr_matrix(matrix)
    month_values = pd.Series(months, copy=False).astype(str).to_numpy()
    if x.shape[0] != len(month_values):
        raise ValueError("Matrix rows and month labels differ")
    row_energy = np.asarray(x.multiply(x).sum(axis=1)).ravel()
    frame = pd.DataFrame({"year_month": month_values, "row_energy": row_energy})
    out = frame.groupby("year_month", sort=True, as_index=False)["row_energy"].sum()
    out = out.rename(columns={"row_energy": "frobenius_energy"})
    total = float(out["frobenius_energy"].sum())
    if total <= 0:
        raise ValueError("Total matrix energy must be positive")
    out["energy_share"] = out["frobenius_energy"] / total
    out["uniform_target"] = 1.0 / len(out)
    out["residual"] = out["energy_share"] - out["uniform_target"]
    return out


def compute_tvd(shares: Sequence[float], target: Sequence[float] | None = None) -> float:
    """Compute Total Variation Distance between probability vectors."""
    q = np.asarray(shares, dtype=float)
    if target is None:
        r = np.full(q.size, 1.0 / q.size)
    else:
        r = np.asarray(target, dtype=float)
    if q.shape != r.shape or q.size == 0:
        raise ValueError("TVD vectors must be nonempty and have equal shapes")
    if not np.isclose(q.sum(), 1.0) or not np.isclose(r.sum(), 1.0):
        raise ValueError("TVD inputs must sum to one")
    return float(0.5 * np.abs(q - r).sum())


def _squared_residual_without_dense_reconstruction(
    x_reference: sparse.spmatrix,
    w: np.ndarray,
    h: np.ndarray,
) -> float:
    """Compute ||X-WH||_F^2 without materializing the N x vocabulary product."""
    x = sparse.csr_matrix(x_reference)
    w = np.asarray(w, dtype=float)
    h = np.asarray(h, dtype=float)
    if w.shape[0] != x.shape[0] or h.shape[1] != x.shape[1] or w.shape[1] != h.shape[0]:
        raise ValueError("Incompatible X, W, and H shapes")
    x_sq = float(sparse_norm(x) ** 2)
    wh_sq = float(np.sum((w.T @ w) * (h @ h.T)))
    xht = np.asarray(x @ h.T)
    cross = float(np.sum(w * xht))
    return max(0.0, x_sq + wh_sq - 2.0 * cross)


def compute_rre_valid(model, x_base_valid: sparse.spmatrix) -> tuple[float, np.ndarray]:
    """Project the common unweighted validation matrix and compute relative error."""
    x = sparse.csr_matrix(x_base_valid)
    denominator = float(sparse_norm(x))
    if denominator <= 0:
        raise ValueError("Validation reference matrix must have positive norm")
    w_valid = np.asarray(model.transform(x), dtype=float)
    residual_sq = _squared_residual_without_dense_reconstruction(x, w_valid, model.components_)
    return float(np.sqrt(residual_sq) / denominator), w_valid


def compute_soft_topic_prevalence(
    w_eval: np.ndarray,
    months: pd.Series | Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Row-normalize projected W and average topic proportions within each month."""
    w = np.asarray(w_eval, dtype=float)
    month_values = pd.Series(months, copy=False).astype(str).to_numpy()
    if w.ndim != 2 or w.shape[0] != len(month_values):
        raise ValueError("W rows and month labels differ")
    row_sums = w.sum(axis=1, keepdims=True)
    zero_rows = np.flatnonzero(row_sums.ravel() <= 0)
    if zero_rows.size:
        raise ValueError(
            f"Projected W contains {zero_rows.size} zero-sum rows; soft prevalence would be undefined"
        )
    theta = w / row_sums
    columns = [f"topic_{i}" for i in range(1, w.shape[1] + 1)]
    frame = pd.DataFrame(theta, columns=columns)
    frame.insert(0, "year_month", month_values)
    prevalence = frame.groupby("year_month", sort=True, as_index=False)[columns].mean()
    return prevalence, theta


def encode_features_for_coherence(analyzer_tokens: Iterable[Iterable[str]]) -> list[list[str]]:
    """Encode n-grams as one dictionary token while retaining unigrams."""
    return [[str(feature).replace(" ", "_") for feature in document] for document in analyzer_tokens]


def encode_topic_features(topics: Iterable[Iterable[str]]) -> list[list[str]]:
    return [[str(feature).replace(" ", "_") for feature in topic] for topic in topics]


def coherence_feature_coverage(
    encoded_topics: Sequence[Sequence[str]],
    dictionary_token2id: dict[str, int],
) -> tuple[int, int, float, list[str]]:
    """Audit whether every top-word position is represented in the dictionary."""
    flattened = [feature for topic in encoded_topics for feature in topic]
    missing = sorted({feature for feature in flattened if feature not in dictionary_token2id})
    covered = sum(feature in dictionary_token2id for feature in flattened)
    total = len(flattened)
    coverage = covered / total if total else 0.0
    return covered, total, coverage, missing


def compute_cv_coherence(
    topics: Sequence[Sequence[str]],
    coherence_corpus: list[list[str]],
    processes: int = 1,
) -> CoherenceResult:
    """Compute c_v on the full train corpus with a no-filter dictionary and 100% coverage."""
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel
    except Exception as exc:  # pragma: no cover - depends on external binary environment
        raise RuntimeError(
            "Gensim could not be imported. Install the versions in requirements.txt."
        ) from exc

    dictionary = Dictionary(coherence_corpus)
    encoded_topics = encode_topic_features(topics)
    covered, total, coverage, missing = coherence_feature_coverage(encoded_topics, dictionary.token2id)
    if not np.isclose(coverage, 1.0):
        raise ValueError(
            f"Coherence dictionary coverage is {coverage:.6f}, expected 1.0; missing={missing[:20]}"
        )
    model = CoherenceModel(
        topics=encoded_topics,
        texts=coherence_corpus,
        dictionary=dictionary,
        coherence="c_v",
        processes=processes,
    )
    return CoherenceResult(float(model.get_coherence()), covered, total, coverage)


def validate_nonnegative(name: str, matrix: sparse.spmatrix) -> None:
    x = sparse.csr_matrix(matrix)
    if x.data.size and (x.data < 0).any():
        raise ValueError(f"{name} contains negative values")
    if x.data.size and not np.isfinite(x.data).all():
        raise ValueError(f"{name} contains non-finite values")


def row_l2_norms(matrix: sparse.spmatrix) -> np.ndarray:
    x = sparse.csr_matrix(matrix)
    return np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())


def select_representative_documents(
    metadata: pd.DataFrame,
    theta: np.ndarray,
    top_n: int = 5,
) -> pd.DataFrame:
    """Select top documents per topic from common-space projected proportions."""
    if len(metadata) != theta.shape[0]:
        raise ValueError("Metadata rows and theta rows differ")
    rows: list[dict] = []
    for topic_idx in range(theta.shape[1]):
        order = np.argsort(theta[:, topic_idx])[::-1][:top_n]
        for rank, row_idx in enumerate(order, start=1):
            base = metadata.iloc[int(row_idx)].to_dict()
            base.update(
                {
                    "topic_id": topic_idx + 1,
                    "rank": rank,
                    "topic_proportion": float(theta[row_idx, topic_idx]),
                    "document_row": int(row_idx),
                }
            )
            rows.append(base)
    return pd.DataFrame(rows)
