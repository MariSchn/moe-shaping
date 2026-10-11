"""Build a Megatron training corpus from the five domains of the specialization eval set.

Each domain is streamed from the training split of the source behind its eval domain (for
Wikipedia, which has no other split, the documents the eval set took are skipped), tokenized
with GPT-2 BPE and written as its own Megatron IndexedDataset, one document per sequence.
Train on them as an equal-weight blend:

    DATA_PATH="0.2 $OUT/wikipedia 0.2 $OUT/github 0.2 $OUT/arxiv 0.2 $OUT/books 0.2 $OUT/web"

    uv run --no-project --python 3.11 --with datasets --with tiktoken --with numpy \
        python data/build_domain_mix.py --out /iopsstor/scratch/cscs/$USER/moe-shaping/domain_mix \
        --eval-manifest /iopsstor/scratch/cscs/$USER/moe-shaping/domain_eval/manifest.json
"""

import argparse
import json
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset

DOMAINS = {
    "wikipedia": ("wikimedia/wikipedia", "20231101.en", "train", "text"),
    "github": ("codeparrot/codeparrot-clean-train", None, "train", "content"),
    "arxiv": ("ccdv/arxiv-summarization", None, "train", "article"),
    "books": ("emozilla/pg19", None, "train", "text"),
    "web": ("allenai/c4", "en", "train", "text"),
}
SHARES_EVAL_STREAM = {"wikipedia"}

_INDEX_HEADER = b"MMIDIDX\x00\x00"
_INDEX_VERSION = 1
_DTYPE_CODE_INT32 = 4
_DTYPE_SIZE = 4


def write_index(idx_path, lengths):
    lengths = np.asarray(lengths, dtype=np.int32)
    pointers = np.zeros(len(lengths), dtype=np.int64)
    pointers[1:] = np.cumsum(lengths[:-1].astype(np.int64)) * _DTYPE_SIZE
    document_indices = np.arange(len(lengths) + 1, dtype=np.int64)
    with open(idx_path, "wb") as f:
        f.write(_INDEX_HEADER)
        f.write(struct.pack("<Q", _INDEX_VERSION))
        f.write(struct.pack("<B", _DTYPE_CODE_INT32))
        f.write(struct.pack("<Q", len(lengths)))
        f.write(struct.pack("<Q", len(document_indices)))
        f.write(lengths.tobytes())
        f.write(pointers.tobytes())
        f.write(document_indices.tobytes())


def build_domain(name, out, target_tokens, skip_docs, batch_docs, threads):
    path, config, split, field = DOMAINS[name]
    enc = tiktoken.get_encoding("gpt2")
    stream = load_dataset(path, config, split=split, streaming=True)
    documents = (example[field] for example in islice(stream, skip_docs, None))
    lengths = []
    total = 0
    start = time.time()
    with open(out / f"{name}.bin", "wb") as bin_file:
        while total < target_tokens:
            batch = [text for text in islice(documents, batch_docs) if text]
            if not batch:
                break
            for tokens in enc.encode_ordinary_batch(batch, num_threads=threads):
                tokens.append(enc.eot_token)
                bin_file.write(np.asarray(tokens, dtype=np.int32).tobytes())
                lengths.append(len(tokens))
                total += len(tokens)
            print(f"{name}: {len(lengths)} docs, {total / 1e6:.0f}M tokens, {time.time() - start:.0f}s", flush=True)
    write_index(out / f"{name}.idx", lengths)
    return name, {"source": path, "split": split, "skipped_docs": skip_docs, "docs": len(lengths), "tokens": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--tokens-per-domain", type=float, default=360e6)
    parser.add_argument("--batch-docs", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eval_docs = {name: spec["docs"] for name, spec in json.loads(Path(args.eval_manifest).read_text())["domains"].items()}
    with ProcessPoolExecutor(len(args.domains)) as pool:
        futures = [
            pool.submit(
                build_domain, name, out, int(args.tokens_per_domain),
                eval_docs[name] if name in SHARES_EVAL_STREAM else 0, args.batch_docs, args.threads,
            )
            for name in args.domains
        ]
        manifest = dict(future.result() for future in futures)
    manifest_path = out / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest_path.write_text(json.dumps({**previous, **manifest}, indent=2))


if __name__ == "__main__":
    main()
