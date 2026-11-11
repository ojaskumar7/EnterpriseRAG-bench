#!/usr/bin/env python3
"""Benchmark lexical retrieval variants on the development split."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import asyncpg  # noqa: E402
import Stemmer  # noqa: E402

from app.db.session import build_engine, build_session_factory  # noqa: E402
from app.embeddings.bge import BGEEmbeddingProvider  # noqa: E402
from app.evaluation.dev import read_dev_questions  # noqa: E402
from app.evaluation.lexical_benchmark import (  # noqa: E402
    BM25SpikeRetriever,
    build_manifest,
    corpus_fingerprint,
    evaluate_variant,
    load_or_build_bm25_index,
    measure_top_k_latency,
    select_best_bm25_variant,
)
from app.evaluation.lexical_postgres import (  # noqa: E402
    PostgresBenchmarkConfig,
    PostgresLexicalRetriever,
    load_chunks,
    load_corpus_counts,
    load_document_frequencies,
)
from app.retrieval.dense import DenseRetriever  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402
from app.settings import (  # noqa: E402
    DEFAULT_KNOWLEDGE_BASE_ID,
    DEFAULT_TENANT_ID,
    get_settings,
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _ram_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def _drift(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    left_by_id = {
        str(item["question_id"]): item["retrieved_doc_ids"]
        for item in left["per_question"]  # type: ignore[index]
    }
    right_by_id = {
        str(item["question_id"]): item["retrieved_doc_ids"]
        for item in right["per_question"]  # type: ignore[index]
    }
    return {
        "questions_with_different_docs": sum(
            1
            for question_id, docs in left_by_id.items()
            if docs != right_by_id.get(question_id)
        ),
        "recall_at_10_delta": round(
            float(left["metrics"]["recall_at_10"])  # type: ignore[index]
            - float(right["metrics"]["recall_at_10"]),  # type: ignore[index]
            6,
        ),
        "mrr_delta": round(
            float(left["metrics"]["mrr"]) - float(right["metrics"]["mrr"]),  # type: ignore[index]
            6,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data/sample/dev_questions.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/bm25_index",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/lexical_benchmark.json",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Embedding model name; defaults to settings",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Embedding device; defaults to settings",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default=None,
        help="Embedding dtype; defaults to settings",
    )
    parser.add_argument("--pg-timeout", type=float, default=10.0)
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild persisted BM25 indexes",
    )
    parser.add_argument("--tenant-id", type=uuid.UUID, default=DEFAULT_TENANT_ID)
    parser.add_argument(
        "--knowledge-base-id",
        type=uuid.UUID,
        default=DEFAULT_KNOWLEDGE_BASE_ID,
    )
    return parser


async def run(args: argparse.Namespace) -> dict[str, object]:
    if "holdout" in args.questions.name.lower():
        raise ValueError("holdout questions are not allowed in this benchmark")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.top_k <= 0 or args.candidate_k <= 0 or args.rrf_k <= 0:
        raise ValueError("top-k, candidate-k, and rrf-k must be positive")
    if args.pg_timeout <= 0:
        raise ValueError("--pg-timeout must be positive")

    settings = get_settings()
    questions = read_dev_questions(args.questions, args.limit)
    if not questions:
        raise ValueError("no development questions were loaded")

    config = PostgresBenchmarkConfig(
        tenant_id=args.tenant_id,
        knowledge_base_id=args.knowledge_base_id,
        pg_timeout_s=args.pg_timeout,
    )
    dsn = settings.database_url.replace("+asyncpg", "")
    connection = await asyncpg.connect(dsn)
    engine = None
    try:
        documents, chunk_count = await load_corpus_counts(connection, config)
        if documents == 0 or chunk_count == 0:
            raise ValueError("benchmark corpus is empty")
        chunks = await load_chunks(connection, config)
        if len(chunks) != chunk_count:
            raise ValueError("loaded chunk count does not match the database")
        fingerprint = corpus_fingerprint(
            (chunk.chunk_id, chunk.content_hash) for chunk in chunks
        )
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        document_frequencies = await load_document_frequencies(connection, config)
        if not document_frequencies:
            raise ValueError("document frequencies are empty")
        hnsw_ef_search = int(await connection.fetchval("SHOW hnsw.ef_search"))

        provider = BGEEmbeddingProvider(
            args.model or settings.embedding_model,
            settings.embedding_dimension,
            device=args.device or settings.embedding_device,
            batch_size=settings.embedding_batch_size,
            dtype=args.dtype or settings.embedding_dtype,
            query_instruction=settings.embedding_query_instruction,
        )
        engine = build_engine(settings)
        session_factory = build_session_factory(engine)
        dense = DenseRetriever(
            provider,
            session_factory,
            tenant_id=args.tenant_id,
            knowledge_base_id=args.knowledge_base_id,
        )

        stemmer = Stemmer.Stemmer("english")
        bm25s_version = importlib.metadata.version("bm25s")
        pystemmer_version = importlib.metadata.version("PyStemmer")
        bm25_indexes: dict[str, dict[str, object]] = {}
        bm25_retrievers: dict[str, BM25SpikeRetriever] = {}
        for weighting in ("content", "title_x3"):
            manifest = build_manifest(
                index_name=weighting,
                corpus_fingerprint=fingerprint,
                documents=documents,
                chunks=chunk_count,
                title_weighting=weighting,
                bm25s_version=bm25s_version,
                pystemmer_version=pystemmer_version,
            )
            index, build_seconds, load_seconds, size_bytes = load_or_build_bm25_index(
                index_dir=args.index_dir / weighting,
                chunks=chunks,
                title_weighting=weighting,
                stemmer=stemmer,
                expected_manifest=manifest,
                rebuild=args.rebuild_index,
            )
            bm25_indexes[weighting] = {
                "build_seconds": round(build_seconds, 3),
                "load_seconds": round(load_seconds, 3),
                "size_bytes": size_bytes,
                "manifest": manifest,
            }
            bm25_retrievers[
                "bm25_content" if weighting == "content" else "bm25_title_weighted"
            ] = BM25SpikeRetriever(index, chunks, stemmer=stemmer)

        pg_retrievers = {
            "pg_and": PostgresLexicalRetriever(
                connection,
                chunks_by_id,
                config=config,
                document_frequencies=document_frequencies,
                mode="and",
            ),
            "pg_and_or": PostgresLexicalRetriever(
                connection,
                chunks_by_id,
                config=config,
                document_frequencies=document_frequencies,
                mode="and_or_8",
            ),
            "pg_relaxed3": PostgresLexicalRetriever(
                connection,
                chunks_by_id,
                config=config,
                document_frequencies=document_frequencies,
                mode="relaxed_and_3",
                n_threshold=3,
            ),
            "pg_relaxed5": PostgresLexicalRetriever(
                connection,
                chunks_by_id,
                config=config,
                document_frequencies=document_frequencies,
                mode="relaxed_and_5",
                n_threshold=5,
            ),
        }

        results: dict[str, dict[str, object]] = {}
        results["dense_a"] = await evaluate_variant(
            "dense_a",
            dense,
            questions,
            top_k=args.top_k,
        )
        results["dense_b"] = await evaluate_variant(
            "dense_b",
            dense,
            questions,
            top_k=args.top_k,
        )
        for name, retriever in pg_retrievers.items():
            results[name] = await evaluate_variant(
                name,
                retriever,
                questions,
                top_k=args.top_k,
            )
        for name, retriever in bm25_retrievers.items():
            results[name] = await evaluate_variant(
                name,
                retriever,
                questions,
                top_k=args.top_k,
            )
            results[name]["latency_top50"] = await measure_top_k_latency(
                retriever,
                questions,
                top_k=50,
            )

        best_bm25 = select_best_bm25_variant(results)
        hybrid = HybridRetriever(
            dense,
            bm25_retrievers[best_bm25],
            rrf_k=args.rrf_k,
            candidate_k=args.candidate_k,
        )
        results["hybrid_bm25"] = await evaluate_variant(
            "hybrid_bm25",
            hybrid,
            questions,
            top_k=args.top_k,
        )

        output: dict[str, object] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "commit": _git_commit(),
            "questions": {
                "path": str(args.questions),
                "count": len(questions),
            },
            "host": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cpu_count": os.cpu_count(),
                "ram_bytes": _ram_bytes(),
                "device": provider.device,
                "dtype": provider.dtype,
            },
            "corpus": {
                "documents": documents,
                "chunks": chunk_count,
                "fingerprint": fingerprint,
                "document_frequencies": len(document_frequencies),
            },
            "settings": {
                "model": provider.model_name,
                "top_k": args.top_k,
                "candidate_k": args.candidate_k,
                "rrf_k": args.rrf_k,
                "pg_timeout_s": args.pg_timeout,
                "hnsw_ef_search": hnsw_ef_search,
            },
            "bm25": {
                "k1": 1.5,
                "b": 0.75,
                "method": "lucene",
                "stopwords": "en",
                "stemmer": "PyStemmer english",
                "indexes": bm25_indexes,
                "best_variant": best_bm25,
            },
            "dense_drift": _drift(results["dense_a"], results["dense_b"]),
            "variants": results,
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        print(
            f"{'variant':<22}{'r@10':>8}{'r@20':>8}{'mrr':>8}"
            f"{'p95ms':>10}{'zero':>7}{'timeouts':>9}",
            flush=True,
        )
        for name, result in results.items():
            metrics = result["metrics"]
            latency = result["latency"]
            print(
                f"{name:<22}{float(metrics['recall_at_10']):>8.4f}"
                f"{float(metrics['recall_at_20']):>8.4f}"
                f"{float(metrics['mrr']):>8.4f}"
                f"{float(latency['p95_ms']):>10.2f}"
                f"{int(result['zero_result_questions']):>7}"
                f"{int(result['timeouts']):>9}",
                flush=True,
            )
        return output
    finally:
        if engine is not None:
            await engine.dispose()
        await connection.close()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except (OSError, ValueError, RuntimeError, asyncpg.PostgresError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
