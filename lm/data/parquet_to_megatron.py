"""Convert pre-tokenized Parquet shards to Megatron-LM .bin/.idx format.

Writes the IndexedDataset format directly (no Megatron import needed):
  {prefix}.bin  — flat int32 token data
  {prefix}.idx  — header + sequence lengths, byte pointers, document indices
"""

import argparse
import glob
import os
import struct
import time

import numpy as np
import pyarrow.parquet as pq

# Megatron IndexedDataset constants
_INDEX_HEADER = b"MMIDIDX\x00\x00"
_INDEX_VERSION = 1
# DType code 4 = numpy.int32 (used when vocab_size > 65536, i.e. GPT-2)
_DTYPE_CODE_INT32 = 4
_DTYPE = np.int32
_DTYPE_SIZE = 4  # bytes


def convert(args):
    parquet_files = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    assert parquet_files, f"No parquet files found in {args.input}"
    print(f"Found {len(parquet_files)} parquet shards")

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)

    bin_path = args.output_prefix + ".bin"
    idx_path = args.output_prefix + ".idx"

    sequence_lengths = []
    document_indices = [0]

    total_docs = 0
    total_tokens = 0
    t0 = time.time()

    with open(bin_path, "wb") as bin_file:
        for fi, fpath in enumerate(parquet_files):
            table = pq.read_table(fpath, columns=["tokens"])
            tokens_col = table.column("tokens")
            n = len(tokens_col)

            for row_idx in range(n):
                token_arr = tokens_col[row_idx]
                length = len(token_arr)
                # Write tokens as int32 directly from pyarrow → numpy
                np_tokens = token_arr.values.to_numpy().astype(_DTYPE)
                bin_file.write(np_tokens.tobytes())
                sequence_lengths.append(length)
                # Each row is one document with one sequence
                document_indices.append(len(sequence_lengths))
                total_tokens += length

            total_docs += n
            elapsed = time.time() - t0
            print(
                f"  Shard {fi+1}/{len(parquet_files)}: {os.path.basename(fpath)} "
                f"({n} docs) | cumulative: {total_docs} docs, "
                f"{total_tokens/1e9:.2f}B tokens, {elapsed:.0f}s"
            )

    # Write the .idx file
    sequence_lengths = np.array(sequence_lengths, dtype=np.int32)
    document_indices = np.array(document_indices, dtype=np.int64)

    # Compute byte pointers for each sequence
    sequence_pointers = np.zeros(len(sequence_lengths), dtype=np.int64)
    if len(sequence_lengths) > 0:
        sequence_pointers[1:] = np.cumsum(sequence_lengths[:-1].astype(np.int64)) * _DTYPE_SIZE

    with open(idx_path, "wb") as idx_file:
        idx_file.write(_INDEX_HEADER)
        idx_file.write(struct.pack("<Q", _INDEX_VERSION))
        idx_file.write(struct.pack("<B", _DTYPE_CODE_INT32))
        idx_file.write(struct.pack("<Q", len(sequence_lengths)))  # sequence count
        idx_file.write(struct.pack("<Q", len(document_indices)))  # document count
        idx_file.write(sequence_lengths.tobytes())
        idx_file.write(sequence_pointers.tobytes())
        idx_file.write(document_indices.tobytes())

    elapsed = time.time() - t0
    bin_size = os.path.getsize(bin_path) / (1024**3)
    idx_size = os.path.getsize(idx_path) / (1024**2)
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  {total_docs} documents, {total_tokens/1e9:.2f}B tokens")
    print(f"  {bin_path} ({bin_size:.2f} GB)")
    print(f"  {idx_path} ({idx_size:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing .parquet shards",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output path prefix (will create {prefix}.bin and {prefix}.idx)",
    )
    convert(parser.parse_args())
