"""Same work as tests/bench.mojo with gguf-py's GGUFReader."""
import sys, time
t = time.perf_counter()
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
f = r.fields.get("tokenizer.ggml.tokens")
vocab = len(f.data) if f else 0
b = sum(x.n_bytes for x in r.tensors)
print(len(r.fields) - 3, "kvs", len(r.tensors), "tensors", vocab, "tokens", b, "B", int((time.perf_counter() - t) * 1000), "ms")
