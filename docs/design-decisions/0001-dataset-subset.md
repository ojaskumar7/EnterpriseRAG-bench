# 0001: EnterpriseRAG-Bench Block 1 Subset

- Status: accepted
- Date: 2026-11-06
- Scope: Block 1 ingestion and benchmark subset

## Context

Block 1 needs a realistic, reproducible enterprise corpus for ingestion, chunking,
embedding, retrieval, and evaluation. The full benchmark corpus is large enough that
downloading and indexing it would add cost and iteration time before the pipeline is
validated.

## Decision

### Dataset source

Use the official `onyx-dot-app/EnterpriseRAG-Bench` repository and its `v1.0.0`
release assets:

- Repository: https://github.com/onyx-dot-app/EnterpriseRAG-Bench
- Release base: https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/v1.0.0

Do not use the `SJChen02/EnterpriseRAG-Bench` mirror. The official repository is the
provenance source for this project.

The full corpus contains 511,962 documents and 500 benchmark questions. Block 1 does
not download the full parquet corpus.

### Block 1 sources

Download every release slice for the following four sources:

| Source | Slices | Document members |
|---|---:|---:|
| Confluence | 2 | 5,189 |
| Jira | 2 | 6,120 |
| GitHub | 2 | 8,052 |
| HubSpot | 4 | 15,017 |
| **Total** | **10** | **34,378** |

These sources provide a mix of long-form documentation, support tickets, technical
discussions, and CRM records. Slack and Gmail are excluded from Block 1 because they
dominate corpus volume and would require many additional slices. They can be added
when scaling to the full corpus.

### Label leakage rule

`source_types`, `expected_doc_ids`, `question_type`, and `gold_answer` are evaluation
metadata only. Retrieval and generation code must receive only the question text and
product-level filters chosen by the user. The question manifest must never be passed
to a retriever.

### Evaluation split

Use a stratified 80/20 development/holdout split with `question_type` as the
stratification key and a fixed seed of 42. Keep the holdout untouched until Block 6.

Actual Block 1 counts:

- Questions in release: 500
- Eligible retrieval questions: 211
- Development: 169
- Holdout: 42
- Abstention set: 30
- Excluded: 259
- Missing-gold-document failures: 0

All 259 excluded questions use at least one source type outside the Block 1 subset.
The 30 abstention questions have no expected documents and cover `high_level` and
`info_not_found` question types.

### Chunking and embedding defaults

- Embedding model: BGE-large-en-v1.5, 1024 dimensions
- Chunk size: 400 tokens
- Chunk overlap: 64 tokens
- Minimum chunk size: 50 tokens
- Token counting must use the embedding model tokenizer, not whitespace splitting

## Data-quality findings

The release slices contain 34,378 document members but only 34,375 unique `doc_id`
values:

- 2 `doc_id` values appear under conflicting source types:
  - one Confluence document also appears as a Jira ticket
  - one Confluence document also appears as a HubSpot company record
- 1 `doc_id` appears twice under Jira with different ticket titles

The inventory therefore maps each `doc_id` to the set of source types where it
appears and retains every source variant. This preserves the official release data
instead of silently dropping one copy.

Consequences for later tasks:

- Document storage must use a composite identity such as
  `(source_type, doc_id)`, not `doc_id` alone.
- Chunk IDs must include `source_type` or another document-level discriminator to
  remain unique.
- Retrieval evaluation should deduplicate retrieved results by `doc_id` when
  comparing against `expected_doc_ids`.
- Loaders must preserve the original member path and source type.

## Artifacts

- `data/raw/slices/*.zip`: 10 verified release slices
- `data/raw/questions.jsonl`: 500 benchmark questions
- `data/raw/download_manifest.json`: URLs, byte sizes, SHA256 hashes, and member counts
- `data/sample/dev_questions.jsonl`: 169 development questions
- `data/sample/holdout_questions.jsonl`: 42 holdout questions
- `data/sample/abstention_questions.jsonl`: 30 zero-gold questions
- `data/sample/excluded_questions.jsonl`: 259 excluded questions with reasons
- `data/sample/subset_manifest.json`: split IDs, counts, inventory summary, and exclusions

Download integrity: 11 assets, 84.5 MiB, all zip archives passed `testzip()`.
