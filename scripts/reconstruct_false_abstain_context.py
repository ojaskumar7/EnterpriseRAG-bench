#!/usr/bin/env python3
"""Rebuild local BM25 contexts for zero-API false-abstain classification."""
from __future__ import annotations
import argparse, asyncio, json, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from app.context import ContextBuilder
from app.db.session import build_engine, build_session_factory
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.bm25_index import BM25IndexManager
from app.settings import get_settings

async def run(args):
    records=[]
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        if r.get("answer",{}).get("insufficient_evidence") and set(r.get("expected_doc_ids",[])) & set(r.get("context",{}).get("selected_doc_ids",[])):
            records.append(r)
    engine=build_engine(get_settings())
    out=[]
    try:
        s=get_settings(); retriever=BM25Retriever(BM25IndexManager(s.bm25_index_root), build_session_factory(engine), tenant_id=s.default_tenant_id, knowledge_base_id=s.default_knowledge_base_id)
        builder=ContextBuilder(token_budget=4000,max_chunks_per_doc=3)
        for r in records:
            candidates=await retriever.retrieve(r["question"],top_k=50)
            context=builder.build(r["question"],candidates)
            gold=set(r["expected_doc_ids"])
            out.append({"question_id":r["question_id"],"question":r["question"],"gold_doc_ids":sorted(gold),"blocks":[{"doc_id":b.doc_id,"title":b.title,"content":b.content} for b in context.blocks],"gold_blocks":[{"doc_id":b.doc_id,"title":b.title,"content":b.content} for b in context.blocks if b.doc_id in gold]})
    finally: await engine.dispose()
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"wrote {len(out)} contexts to {args.output}")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,default=Path("data/processed/generation_dev.jsonl")); p.add_argument("--output",type=Path,default=Path("data/processed/context_present_false_abstains.json")); args=p.parse_args(); asyncio.run(run(args))
if __name__ == "__main__": main()
