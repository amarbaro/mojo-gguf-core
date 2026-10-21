# mojo-gguf-core

[![CI](https://github.com/amarbaro/mojo-gguf-core/actions/workflows/ci.yml/badge.svg)](https://github.com/amarbaro/mojo-gguf-core/actions/workflows/ci.yml)

Read and write [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) files (the
llama.cpp model format) from Mojo. The reader mmaps the file and parses the header,
key-values and tensor table in place. Tensor data is a `Span` into the mapping, so a 20 GB
model opens in 36 ms at 29 MB RSS. The writer produces the bytes gguf-py's `GGUFWriter`
produces. The gate checks everything against gguf-py 0.19.0. 360 lines, no dependencies.

```mojo
from gguf import Gguf, Writer, T_UINT32

def main() raises:
    var g = Gguf("model.gguf")                          # mmapped, nothing copied
    print(g.str(g.find("general.architecture")), len(g.tensors))
    var vocab = g.strs(g.find("tokenizer.ggml.tokens"))
    var w = g.data(g.tensor("token_embd.weight"))      # Span[UInt8] into the file

    var out = Writer()
    out.add_str("general.name", "demo")
    out.add_int("demo.context_length", T_UINT32, 4096)
    out.save("demo.gguf")
```

## Install

Requires Mojo 1.1 on Linux. Builds on Mojo 1.2 nightly too.

| Way | Command |
|---|---|
| Source path | `mojo run -I path/to/mojo-gguf-core/src app.mojo` |
| Precompiled | `mojo precompile src/gguf -o gguf.mojoc` |
| Vendor | copy `src/gguf/` into your project |

## What you get

- `Gguf(path)`: GGUF v2 and v3, little-endian. `kvs` (`KV`: `key`, `vtype`, `atype`, `count`) and
  `tensors` (`TensorInfo`: `name`, `dims` ne0 first, `ttype`, `offset`, `nbytes`, `numel()`),
  both in file order. Getters `int`, `float`, `str`, `strs`, `ints`; `value(i)` gives the raw
  payload, `data(i)` the tensor bytes, `bytes()` the whole file.
- It rejects what gguf-py rejects: bad magic, other versions, unknown value or ggml types,
  duplicate keys or tensors, a non power-of-two or non-uint32 `general.alignment`, truncation
  anywhere, tensor data past the end.
- All 34 ggml types gguf-py 0.19 knows (`type_size`), K-quants, IQ, TQ, MXFP4, NVFP4 and Q1_0 included.
- `Writer`: `add_int`, `add_float`, `add_str`, `add_strs`, `add_ints`, `add_floats`, `add_raw`,
  `add_tensor`, `save`. Same bytes as `GGUFWriter`: KVs in first-added order (re-adding
  replaces in place), tensors in added order, offsets and data padded to the alignment, which a
  `general.alignment` KV sets.

API: [docs/api.md](docs/api.md). Design: [docs/design.md](docs/design.md).

## Performance

`scripts/bench.sh` (`tests/bench.mojo` vs `tools/bench_oracle.py`: open, all KVs, vocabulary, tensor infos):

| | mojo-gguf-core | gguf-py 0.19 |
|---|---|---|
| 20.9 GB Q4_K_S model, 142 KVs, 733 tensors, 248k tokens | 36 ms, 29 MB | 5,535 ms, 863 MB |
| gate fixture, 21 KVs, 9 tensors | under 1 ms, 11 MB | 60 ms, 36 MB |

## Checks

| Script | What |
|---|---|
| `scripts/check.sh` | type table equals gguf-py's; 20 files (6 written by GGUFWriter from `tests/specs/`, 14 corrupted) read or rejected as GGUFReader does; copies and typed builds byte-identical to GGUFWriter; real local files when present; API and example |
| `scripts/selfcheck.sh` | 7 mutants (duplicate keys, padding, sign extension, string arrays, Q4_K size, re-added key, bounds) all rejected |
| `scripts/bench.sh` | the speed numbers above, 3 runs each, peak RSS |
| `scripts/ci.sh` | install and gate, same on GitHub and a local VM |

## License

Apache-2.0.
