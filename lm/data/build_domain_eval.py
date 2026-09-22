"""Build the domain-labelled held-out set used for the expert-specialization analysis.

Each domain is streamed from Hugging Face, tokenized with the GPT-2 BPE the models train on,
joined with <|endoftext|> separators and cut into (num_seqs, seq_len + 1) windows, so the same
file serves both routing statistics and a next-token loss.

    uv run --no-project --python 3.11 --with datasets --with tiktoken \
        python data/build_domain_eval.py --out /iopsstor/scratch/cscs/$USER/moe-shaping/domain_eval
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset

DOMAINS = {
    "wikipedia": ("wikimedia/wikipedia", "20231101.en", "train", "text"),
    "github": ("codeparrot/codeparrot-clean-valid", None, "train", "content"),
    "arxiv": ("ccdv/arxiv-summarization", None, "test", "article"),
    "books": ("emozilla/pg19-test", None, "test", "text"),
    "web": ("allenai/c4", "en", "validation", "text"),
}


def build_domain(name, spec, enc, num_seqs, seq_len, max_doc_tokens):
    path, config, split, field = spec
    needed = num_seqs * (seq_len + 1)
    stream = []
    num_docs = 0
    for example in load_dataset(path, config, split=split, streaming=True):
        tokens = enc.encode_ordinary(example[field])[:max_doc_tokens]
        if not tokens:
            continue
        stream.extend(tokens)
        stream.append(enc.eot_token)
        num_docs += 1
        if len(stream) >= needed:
            break
    windows = np.asarray(stream[:needed], dtype=np.int32).reshape(num_seqs, seq_len + 1)
    print(f"{name}: {num_docs} docs -> {windows.shape}")
    return windows, num_docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-seqs", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--max-doc-tokens", type=int, default=16384)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    manifest = {"seq_len": args.seq_len, "domains": {}}
    for name, spec in DOMAINS.items():
        windows, num_docs = build_domain(
            name, spec, enc, args.num_seqs, args.seq_len, args.max_doc_tokens
        )
        np.save(out / f"{name}.npy", windows)
        manifest["domains"][name] = {"source": spec[0], "docs": num_docs, "shape": windows.shape}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
