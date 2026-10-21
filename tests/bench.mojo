# Open argv[1] (a .gguf), read every KV's payload, the vocabulary if any, and every tensor's
# info. Prints counts and ms. Same work as tools/bench_oracle.py.
from std.sys import argv
from std.time import perf_counter_ns
from gguf import Gguf


def main() raises:
    var t = perf_counter_ns()
    var g = Gguf(String(argv()[1]))
    var v = g.find("tokenizer.ggml.tokens")
    var vocab = len(g.strs(v)) if v >= 0 else 0
    var b = 0
    for i in range(len(g.tensors)):
        b += g.tensors[i].nbytes
    print(len(g.kvs), "kvs", len(g.tensors), "tensors", vocab, "tokens", b, "B", (perf_counter_ns() - t) // 1_000_000, "ms")
