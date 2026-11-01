"""Run the locked S1-S4 Global NMF temporal-weighting experiment.

This is the canonical rerun entry point. It enforces one month-stratified 80:20
split, a train-only TF-IDF vocabulary, the square-root document weight,
row-normalized Temporal TF-IDF, and common-space validation/evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from paths import OUTPUT_DIR, PROCESSED_DATA_DIR, ensure_project_dirs
from preprocessing import get_stopwords, preprocess
from temporal_nmf_pipeline import (
    apply_temporal_tfidf,
    build_locked_nmf,
    build_scenario_matrices,
    compute_cv_coherence,
    compute_document_weights,
    compute_period_energy_distribution,
    compute_rre_valid,
    compute_soft_topic_prevalence,
    compute_temporal_idf,
    compute_tvd,
    encode_features_for_coherence,
    row_l2_norms,
    select_representative_documents,
    split_train_validation,
    validate_nonnegative,
)

DATASET = PROCESSED_DATA_DIR / "politik_full_year_2024_experiment_qc.csv"
EXPECTED_DATASET_SHA256 = "8d384717ce9f8576816d43312c8dd6815ad05df3110fea32fc0cad43a65b0187"
OUTPUT_ROOT = OUTPUT_DIR / "global_nmf_temporal_locked_rerun"
RANDOM_STATE = 42
VALIDATION_FRACTION = 0.20
N_TOPICS = 20
TOP_N_WORDS = 15
REPRESENTATIVE_DOCUMENTS = 5
MAX_FEATURES = 8000
MIN_DF = 10
MAX_DF = 0.60

SCENARIO_NAMES = {
    "S1_baseline": "TF-IDF + Global NMF",
    "S2_document_weighting": "Square-root Document Weighting + Global NMF",
    "S3_temporal_tfidf": "Row-normalized Temporal TF-IDF + Global NMF",
    "S4_dual_level": "Square-root Document Weighting + Row-normalized Temporal TF-IDF + Global NMF",
}


@dataclass(frozen=True)
class RunConfig:
    smoke: bool
    smoke_rows: int
    max_features: int
    min_df: int
    max_df: float
    n_topics: int = N_TOPICS
    top_n_words: int = TOP_N_WORDS
    random_state: int = RANDOM_STATE
    validation_fraction: float = VALIDATION_FRACTION
    max_iter: int = 400


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_sparse_matrix(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    """Return one deterministic CSR representation for storage and hashing."""
    x = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    x.sum_duplicates()
    x.eliminate_zeros()
    x.sort_indices()
    return x


def sha256_sparse_matrix(matrix: sparse.spmatrix) -> str:
    """Fingerprint canonical numeric CSR content, independent of internal ordering."""
    x = canonicalize_sparse_matrix(matrix)
    digest = hashlib.sha256()
    digest.update(np.asarray(x.shape, dtype=np.int64).tobytes())
    digest.update(x.indptr.astype(np.int64, copy=False).tobytes())
    digest.update(x.indices.astype(np.int64, copy=False).tobytes())
    digest.update(x.data.astype(np.float64, copy=False).tobytes())
    return digest.hexdigest()


def save_verified_sparse_npz(
    path: Path,
    matrix: sparse.spmatrix,
) -> tuple[str, sparse.csr_matrix]:
    """Save canonical CSR, reload it, and return the verified artifact digest."""
    canonical = canonicalize_sparse_matrix(matrix)
    sparse.save_npz(path, canonical)
    reloaded = canonicalize_sparse_matrix(sparse.load_npz(path))
    expected = sha256_sparse_matrix(canonical)
    observed = sha256_sparse_matrix(reloaded)
    if observed != expected:
        raise IOError(f"Sparse artifact changed during NPZ round-trip: {path}")
    return observed, reloaded


def sha256_text_sequence(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def ensure_empty_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use a new directory to prevent artifact mixing."
        )
    path.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path = DATASET) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Content_clean", "parsed_date", "source", "Judul", "Link"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df["parsed_date"] = pd.to_datetime(df["parsed_date"], errors="coerce")
    if df["parsed_date"].isna().any():
        raise ValueError(f"Dataset contains {int(df['parsed_date'].isna().sum())} invalid dates")
    df["year_month"] = df["parsed_date"].dt.to_period("M").astype(str)
    if df["year_month"].nunique() != 12:
        raise ValueError(f"Expected 12 monthly periods, found {df['year_month'].nunique()}")
    return df.reset_index(drop=True)


def stratified_smoke_sample(df: pd.DataFrame, n_rows: int, random_state: int) -> pd.DataFrame:
    """Take a reproducible proportional sample without erasing temporal imbalance."""
    if n_rows >= len(df):
        return df.copy()
    if n_rows < 2 * df["year_month"].nunique():
        raise ValueError("Smoke sample needs at least two rows per monthly stratum")
    sampled, _ = train_test_split(
        df,
        train_size=n_rows,
        random_state=random_state,
        stratify=df["year_month"],
        shuffle=True,
    )
    return sampled.reset_index(drop=True)


def build_preprocessed_frame(df: pd.DataFrame) -> pd.DataFrame:
    stopwords = get_stopwords()
    raw = (df["Judul"].fillna("") + " " + df["Content_clean"].fillna("")).astype(str)
    out = df.copy()
    out["preprocessed_text"] = [preprocess(text, stopwords) for text in raw]
    mask = out["preprocessed_text"].str.split().str.len().ge(5)
    if not mask.all():
        print(f"Dropping {int((~mask).sum())} rows with fewer than five processed tokens")
    return out.loc[mask].reset_index(drop=True)


def build_vectorizer(config: RunConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=config.max_features,
        min_df=config.min_df,
        max_df=config.max_df,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        sublinear_tf=True,
        smooth_idf=True,
        norm="l2",
        dtype=np.float64,
    )


def top_words(model, feature_names: np.ndarray, n: int) -> list[list[str]]:
    return [
        [str(feature_names[index]) for index in component.argsort()[::-1][:n]]
        for component in model.components_
    ]


def topic_diversity(topics: list[list[str]]) -> float:
    features = [feature for topic in topics for feature in topic]
    return len(set(features)) / len(features) if features else 0.0


def build_count_presence(vectorizer: TfidfVectorizer, train_texts: list[str]) -> sparse.csr_matrix:
    count_vectorizer = CountVectorizer(
        vocabulary=vectorizer.vocabulary_,
        binary=True,
        ngram_range=vectorizer.ngram_range,
        token_pattern=vectorizer.token_pattern,
        lowercase=vectorizer.lowercase,
    )
    return count_vectorizer.fit_transform(train_texts).tocsr()


def build_coherence_corpus(vectorizer: TfidfVectorizer, train_texts: list[str]) -> list[list[str]]:
    """Use the full train corpus but retain only features in the fitted vocabulary."""
    analyzer = vectorizer.build_analyzer()
    vocabulary = vectorizer.vocabulary_
    return encode_features_for_coherence(
        [feature for feature in analyzer(text) if feature in vocabulary]
        for text in train_texts
    )


def save_split_ledger(
    df_all: pd.DataFrame,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    """Persist corpus row IDs plus exact row positions in each transformed matrix."""
    ledger = df_all[["parsed_date", "year_month", "source", "Judul", "Link"]].copy()
    ledger["split"] = ""
    ledger["split_row"] = pd.Series(pd.NA, index=ledger.index, dtype="Int64")
    ledger.loc[train.index, "split"] = "train"
    ledger.loc[valid.index, "split"] = "validation"
    train_positions = pd.Series(np.arange(len(train), dtype=np.int64), index=train.index)
    valid_positions = pd.Series(np.arange(len(valid), dtype=np.int64), index=valid.index)
    ledger.loc[train.index, "split_row"] = train_positions
    ledger.loc[valid.index, "split_row"] = valid_positions
    if ledger["split"].eq("").any() or ledger["split_row"].isna().any():
        raise AssertionError("Split ledger contains unassigned rows")
    for split_name, expected_size in (("train", len(train)), ("validation", len(valid))):
        observed = ledger.loc[ledger["split"].eq(split_name), "split_row"].sort_values().to_numpy()
        if not np.array_equal(observed, np.arange(expected_size)):
            raise AssertionError(f"{split_name} split_row is not a complete zero-based sequence")
    ledger.to_csv(output / "split_ledger.csv", index=True, index_label="model_row")
    counts = ledger.groupby(["year_month", "split"], sort=True).size().reset_index(name="n_documents")
    counts.to_csv(output / "split_counts_by_month.csv", index=False)
    return ledger


def run(config: RunConfig, dataset_path: Path = DATASET, output_dir: Path = OUTPUT_ROOT) -> dict:
    start = time.time()
    ensure_project_dirs()
    ensure_empty_output_dir(output_dir)

    dataset_sha256 = sha256_file(dataset_path)
    if not config.smoke and dataset_sha256 != EXPECTED_DATASET_SHA256:
        raise ValueError(
            f"Full rerun requires dataset SHA-256 {EXPECTED_DATASET_SHA256}, found {dataset_sha256}"
        )
    df = load_dataset(dataset_path)
    if config.smoke:
        df = stratified_smoke_sample(df, config.smoke_rows, config.random_state)
    df = build_preprocessed_frame(df)

    train, valid = split_train_validation(
        df,
        validation_fraction=config.validation_fraction,
        random_state=config.random_state,
    )
    # Preserve original model-row indices for an auditable split ledger.
    split_ledger = save_split_ledger(df, train, valid, output_dir)

    vectorizer = build_vectorizer(config)
    train_texts = train["preprocessed_text"].tolist()
    valid_texts = valid["preprocessed_text"].tolist()
    all_texts = df["preprocessed_text"].tolist()
    x_base_train = vectorizer.fit_transform(train_texts).tocsr()
    fitted_vocabulary = dict(vectorizer.vocabulary_)
    fitted_idf = vectorizer.idf_.copy()
    x_base_valid = vectorizer.transform(valid_texts).tocsr()
    x_base_all = vectorizer.transform(all_texts).tocsr()
    vectorizer_state_unchanged_after_reference_transforms = bool(
        vectorizer.vocabulary_ == fitted_vocabulary
        and np.array_equal(vectorizer.idf_, fitted_idf)
    )
    base_sparse_hashes: dict[str, str] = {}
    base_sparse_hashes["X_base_train.npz"], x_base_train = save_verified_sparse_npz(
        output_dir / "X_base_train.npz", x_base_train
    )
    base_sparse_hashes["X_base_valid.npz"], x_base_valid = save_verified_sparse_npz(
        output_dir / "X_base_valid.npz", x_base_valid
    )
    base_sparse_hashes["X_base_all.npz"], x_base_all = save_verified_sparse_npz(
        output_dir / "X_base_all.npz", x_base_all
    )
    feature_names = vectorizer.get_feature_names_out()

    for name, matrix in {
        "X_base_train": x_base_train,
        "X_base_valid": x_base_valid,
        "X_base_all": x_base_all,
    }.items():
        validate_nonnegative(name, matrix)
    train_norms = row_l2_norms(x_base_train)
    nonzero_train_norms = train_norms[train_norms > 0]
    if not np.allclose(nonzero_train_norms, 1.0, atol=1e-10):
        raise AssertionError("Nonzero X_base_train rows are not L2-normalized")

    x_presence_train = build_count_presence(vectorizer, train_texts)
    temporal_idf = compute_temporal_idf(x_presence_train, train["year_month"])
    x_temporal_train = apply_temporal_tfidf(x_base_train, temporal_idf)
    temporal_norms = row_l2_norms(x_temporal_train)
    if not np.allclose(temporal_norms[temporal_norms > 0], 1.0, atol=1e-10):
        raise AssertionError("Nonzero X_temporal_train rows are not L2-normalized")

    document_weights, weight_audit = compute_document_weights(train["year_month"])
    if not np.allclose(weight_audit["weighted_energy"], weight_audit["n_bar"], atol=1e-10):
        raise AssertionError("Square-root weights did not equalize aggregate month energy")
    scenario_matrices = build_scenario_matrices(x_base_train, x_temporal_train, document_weights)
    for name, matrix in scenario_matrices.items():
        validate_nonnegative(name, matrix)

    pd.DataFrame({"feature": feature_names, "temporal_idf": temporal_idf}).to_csv(
        output_dir / "temporal_idf_train.csv", index=False
    )
    weight_audit.to_csv(output_dir / "document_weights_train_by_month.csv", index=False)
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer_train_only.joblib")

    coherence_corpus = build_coherence_corpus(vectorizer, train_texts)
    summaries: list[dict] = []
    all_energy_rows: list[pd.DataFrame] = []
    energy_by_scenario: dict[str, pd.DataFrame] = {}
    scenario_matrix_hashes: dict[str, str] = {}
    prevalence_rows_sum_to_one: dict[str, bool] = {}
    projection_reference_rows: dict[str, int] = {}
    representative_source_rows: dict[str, int] = {}

    metadata_columns = ["parsed_date", "year_month", "source", "Judul", "Link", "Content_clean"]
    for scenario_key, matrix in scenario_matrices.items():
        print(f"Running {scenario_key}: {SCENARIO_NAMES[scenario_key]}", flush=True)
        scenario_dir = output_dir / scenario_key
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_matrix_hashes[scenario_key], matrix = save_verified_sparse_npz(
            scenario_dir / "X_train.npz", matrix
        )
        scenario_matrices[scenario_key] = matrix

        energy = compute_period_energy_distribution(matrix, train["year_month"])
        energy.insert(0, "scenario", scenario_key)
        tvd = compute_tvd(energy["energy_share"].to_numpy())
        energy.to_csv(scenario_dir / "train_month_energy_tvd.csv", index=False)
        all_energy_rows.append(energy)
        energy_by_scenario[scenario_key] = energy.copy()

        model = build_locked_nmf(
            n_topics=config.n_topics,
            max_iter=config.max_iter,
            random_state=config.random_state,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit_transform(matrix)
        converged = model.n_iter_ < config.max_iter
        convergence_warnings = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]

        topics = top_words(model, feature_names, config.top_n_words)
        coherence = compute_cv_coherence(topics, coherence_corpus, processes=1)
        diversity = topic_diversity(topics)
        rre_valid, _ = compute_rre_valid(model, x_base_valid)

        # Every scenario is projected onto the same unweighted all-corpus matrix.
        w_eval = np.asarray(model.transform(x_base_all), dtype=float)
        projection_reference_rows[scenario_key] = int(w_eval.shape[0])
        prevalence, theta = compute_soft_topic_prevalence(w_eval, df["year_month"])
        topic_columns = [f"topic_{i}" for i in range(1, config.n_topics + 1)]
        prevalence_rows_sum_to_one[scenario_key] = bool(
            np.allclose(prevalence[topic_columns].sum(axis=1), 1.0, atol=1e-10)
        )
        prevalence.to_csv(scenario_dir / "soft_topic_prevalence_by_month.csv", index=False)
        np.save(scenario_dir / "projected_topic_proportions.npy", theta)

        assignments = df[["parsed_date", "year_month", "source", "Judul", "Link"]].copy()
        assignments["dominant_topic"] = theta.argmax(axis=1) + 1
        assignments["dominant_topic_proportion"] = theta.max(axis=1)
        assignments.to_csv(scenario_dir / "projected_document_topic_assignments.csv", index=False)

        representatives = select_representative_documents(
            df[metadata_columns], theta, top_n=REPRESENTATIVE_DOCUMENTS
        )
        representative_source_rows[scenario_key] = int(theta.shape[0])
        representatives.to_csv(scenario_dir / "representative_documents_top5.csv", index=False)
        pd.DataFrame(
            {"topic_id": range(1, config.n_topics + 1), "top_features": [", ".join(t) for t in topics]}
        ).to_csv(scenario_dir / "top_features_top15.csv", index=False)
        joblib.dump(model, scenario_dir / "nmf_model.joblib")

        summary = {
            "scenario": scenario_key,
            "scenario_name": SCENARIO_NAMES[scenario_key],
            "n_topics": config.n_topics,
            "n_iter": int(model.n_iter_),
            "converged_before_max_iter": bool(converged),
            "convergence_warnings": convergence_warnings,
            "tvd_train_energy": tvd,
            "coherence_cv": coherence.coherence_cv,
            "coherence_feature_coverage": coherence.coverage,
            "coherence_covered_positions": coherence.covered_features,
            "coherence_total_positions": coherence.total_features,
            "topic_diversity_top15": diversity,
            "rre_valid_on_x_base": rre_valid,
            "train_matrix_rows": int(matrix.shape[0]),
            "validation_reference_rows": int(x_base_valid.shape[0]),
            "vocabulary_size": int(matrix.shape[1]),
        }
        summaries.append(summary)
        (scenario_dir / "scenario_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    summary_frame = pd.DataFrame(summaries)
    summary_frame["convergence_warnings"] = summary_frame["convergence_warnings"].map(
        lambda warnings_list: json.dumps(warnings_list, ensure_ascii=False)
    )
    summary_frame.to_csv(output_dir / "S1_S4_metrics_summary.csv", index=False)
    pd.concat(all_energy_rows, ignore_index=True).to_csv(output_dir / "S1_S4_train_month_energy.csv", index=False)

    try:
        import gensim
        gensim_version = gensim.__version__
    except Exception as exc:  # compute_cv_coherence would already have failed; defensive only
        gensim_version = f"unavailable: {exc}"

    scenario_shapes_equal = len({matrix.shape for matrix in scenario_matrices.values()}) == 1
    split_disjoint = set(train.index).isdisjoint(valid.index)
    ledger_train_order = (
        split_ledger.loc[split_ledger["split"].eq("train")]
        .sort_values("split_row")
        .index.to_numpy()
    )
    ledger_valid_order = (
        split_ledger.loc[split_ledger["split"].eq("validation")]
        .sort_values("split_row")
        .index.to_numpy()
    )
    split_ledger_matches_matrix_order = bool(
        np.array_equal(ledger_train_order, train.index.to_numpy())
        and np.array_equal(ledger_valid_order, valid.index.to_numpy())
    )
    all_months_in_both_splits = train["year_month"].nunique() == 12 and valid["year_month"].nunique() == 12
    base_rows_l2_normalized = bool(np.allclose(nonzero_train_norms, 1.0, atol=1e-10))
    temporal_rows_l2_normalized = bool(
        np.allclose(temporal_norms[temporal_norms > 0], 1.0, atol=1e-10)
    )
    square_root_identity_holds = bool(
        np.allclose(weight_audit["weighted_energy"], weight_audit["n_bar"], atol=1e-10)
    )
    coherence_coverage_complete = bool(
        summary_frame["coherence_feature_coverage"].eq(1.0).all()
    )
    common_validation_reference = bool(
        summary_frame["validation_reference_rows"].eq(x_base_valid.shape[0]).all()
    )
    s2_energy = energy_by_scenario["S2_document_weighting"]["frobenius_energy"].to_numpy()
    s4_energy = energy_by_scenario["S4_dual_level"]["frobenius_energy"].to_numpy()
    weighted_scenario_energy_uniform = bool(
        np.allclose(s2_energy, s2_energy.mean(), atol=1e-10)
        and np.allclose(s4_energy, s4_energy.mean(), atol=1e-10)
    )
    s1_tvd = float(
        summary_frame.loc[summary_frame["scenario"].eq("S1_baseline"), "tvd_train_energy"].iloc[0]
    )
    s3_tvd = float(
        summary_frame.loc[summary_frame["scenario"].eq("S3_temporal_tfidf"), "tvd_train_energy"].iloc[0]
    )
    s1_s3_tvd_equal_after_row_l2 = bool(np.isclose(s1_tvd, s3_tvd, atol=1e-12))
    all_prevalence_rows_sum_to_one = bool(all(prevalence_rows_sum_to_one.values()))
    sparse_content_hashes = {
        **base_sparse_hashes,
        **{
            f"{scenario}/X_train.npz": digest
            for scenario, digest in scenario_matrix_hashes.items()
        },
    }
    saved_sparse_artifacts_match_reloaded_content_hashes = bool(
        all(
            sha256_sparse_matrix(sparse.load_npz(output_dir / relative)) == digest
            for relative, digest in sparse_content_hashes.items()
        )
    )
    all_scenario_matrices_nonnegative_finite = bool(
        all(
            matrix.data.size == 0
            or (np.isfinite(matrix.data).all() and (matrix.data >= 0).all())
            for matrix in scenario_matrices.values()
        )
    )
    common_all_corpus_projection = bool(
        set(projection_reference_rows) == set(SCENARIO_NAMES)
        and all(rows == x_base_all.shape[0] for rows in projection_reference_rows.values())
    )
    common_all_corpus_representatives = bool(
        set(representative_source_rows) == set(SCENARIO_NAMES)
        and all(rows == x_base_all.shape[0] for rows in representative_source_rows.values())
    )
    hard_qc = {
        "split_disjoint": split_disjoint,
        "split_ledger_matches_train_and_validation_matrix_order": split_ledger_matches_matrix_order,
        "all_12_months_in_train_and_validation": all_months_in_both_splits,
        "vectorizer_fit_scope_train_only": vectorizer_state_unchanged_after_reference_transforms,
        "scenario_shapes_equal": scenario_shapes_equal,
        "all_scenario_matrices_nonnegative_finite": all_scenario_matrices_nonnegative_finite,
        "x_base_nonzero_rows_l2_normalized": base_rows_l2_normalized,
        "x_temporal_nonzero_rows_l2_renormalized": temporal_rows_l2_normalized,
        "square_root_weight_identity_n_p_a_p_squared_equals_n_bar": square_root_identity_holds,
        "actual_S2_and_S4_month_energy_uniform": weighted_scenario_energy_uniform,
        "actual_S1_and_S3_tvd_equal_after_row_l2": s1_s3_tvd_equal_after_row_l2,
        "coherence_top15_dictionary_coverage_100_percent": coherence_coverage_complete,
        "rre_uses_common_x_base_valid": common_validation_reference,
        "soft_prevalence_uses_common_x_base_all": common_all_corpus_projection,
        "soft_prevalence_month_rows_sum_to_one": all_prevalence_rows_sum_to_one,
        "representative_documents_use_common_x_base_all": common_all_corpus_representatives,
        "saved_sparse_artifacts_match_reloaded_content_hashes": (
            saved_sparse_artifacts_match_reloaded_content_hashes
        ),
    }
    if not all(hard_qc.values()):
        failed = [name for name, passed in hard_qc.items() if not passed]
        raise AssertionError(f"Locked-method QC failed: {failed}")

    source_artifact_file_sha256 = {
        path.relative_to(output_dir).as_posix(): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }
    manifest = {
        "status": "smoke" if config.smoke else "full_rerun",
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "reference_artifacts": {
            "sparse_content_sha256": sparse_content_hashes,
            "source_artifact_file_sha256": source_artifact_file_sha256,
            "custom_stopwords_file": str(Path(__file__).with_name("custom_stopwords_id.txt")),
            "custom_stopwords_sha256": sha256_file(Path(__file__).with_name("custom_stopwords_id.txt")),
            "train_preprocessed_text_sequence_sha256": sha256_text_sequence(train_texts),
            "validation_preprocessed_text_sequence_sha256": sha256_text_sequence(valid_texts),
            "all_preprocessed_text_sequence_sha256": sha256_text_sequence(all_texts),
            "vectorizer_file": "tfidf_vectorizer_train_only.joblib",
            "split_ledger_file": "split_ledger.csv",
            "split_ledger_sha256": sha256_file(output_dir / "split_ledger.csv"),
            "x_base_all_file": "X_base_all.npz",
        },
        "configuration": asdict(config),
        "nmf_configuration": {
            "n_components": config.n_topics,
            "init": "nndsvda",
            "solver": "cd",
            "beta_loss": "frobenius",
            "max_iter": config.max_iter,
            "random_state": config.random_state,
            "alpha_W": 0.0,
            "alpha_H": 0.0,
            "l1_ratio": 0.0,
        },
        "formula_lock": {
            "S1": "X_base_train",
            "S2": "diag(sqrt(n_bar/n_p)) @ X_base_train",
            "S3": "RowNormalize_L2(X_base_train * temporal_idf_train)",
            "S4": "diag(sqrt(n_bar/n_p)) @ S3",
            "validation": "all scenario H projected onto identical X_base_valid",
            "temporal_profile": "all scenario H projected onto identical X_base_all",
        },
        "hard_qc": hard_qc,
        "hard_qc_pass": all(hard_qc.values()),
        "rows_after_preprocessing": int(len(df)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "vocabulary_size": int(x_base_train.shape[1]),
        "month_counts_train": train["year_month"].value_counts().sort_index().to_dict(),
        "month_counts_validation": valid["year_month"].value_counts().sort_index().to_dict(),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "gensim": gensim_version,
            "joblib": joblib.__version__,
        },
        "runtime_seconds": time.time() - start,
        "metrics_file": "S1_S4_metrics_summary.csv",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"manifest": manifest, "metrics": summaries}, indent=2, ensure_ascii=False))
    return {"manifest": manifest, "metrics": summaries}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--n-topics", type=int, choices=(15, 20, 25), default=N_TOPICS,
                        help="Number of NMF topics for an explicit sensitivity run")
    parser.add_argument("--smoke", action="store_true", help="Run a stratified small-corpus integration check")
    parser.add_argument("--smoke-rows", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        smoke=args.smoke,
        smoke_rows=args.smoke_rows,
        max_features=1200 if args.smoke else MAX_FEATURES,
        min_df=2 if args.smoke else MIN_DF,
        max_df=0.95 if args.smoke else MAX_DF,
        n_topics=args.n_topics,
        random_state=args.random_state,
    )
    run(config, dataset_path=args.dataset, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
