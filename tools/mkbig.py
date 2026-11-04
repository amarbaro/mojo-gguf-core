"""A synthetic stand-in for the 20.9 GB model of the README bench: 142 KVs, a 248k-entry token list and 733 one-row F32 tensors,
a few MB on disk (the cost of opening is in the metadata walk, not the weights). Oracle only: product code is Mojo.
    mkbig.py OUT.gguf        Needs gguf-py (uv run --no-project --with gguf==0.19.0)."""
import sys
import numpy as np
from gguf import GGUFWriter

w = GGUFWriter(sys.argv[1], "llama")
w.add_name("mojo-gguf-big")
for i in range(138):
    w.add_uint32(f"big.key{i:03d}", i)
w.add_token_list([f"tok{i:06d}" for i in range(248000)])
for i in range(733):
    w.add_tensor(f"blk.{i // 9}.t{i}.weight", np.full((32,), i, dtype=np.float32))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
