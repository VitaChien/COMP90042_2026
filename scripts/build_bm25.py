"""Build BM25 index from data/evidence.json into cache/bm25_index/. Idempotent."""

from src.config import Config
from src.data_loader import load_evidence
from src.retriever_bm25 import build_bm25_index
from src.utils import get_logger, set_seed, timer


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)
    log = get_logger("bm25-build")
    out = cfg.cache_dir / "bm25_index"
    # bm25s saves multiple files into a directory; treat a populated dir as "built".
    if out.is_dir() and any(out.iterdir()):
        log.info("BM25 index already exists at %s; skipping. Delete to rebuild.", out)
        return
    log.info("Loading evidence ...")
    with timer("load_evidence", log):
        ev = load_evidence(cfg.evidence_path)
    log.info("Loaded %d evidence passages", len(ev))
    with timer("build_index", log):
        build_bm25_index(ev, cache_path=out)


if __name__ == "__main__":
    main()
