---
name: mojo-gguf-core
description: "Read (mmapped, zero copy) and write GGUF model files from Mojo, byte-compatible with gguf-py. Use when Mojo code inspects GGUF metadata or tensors, or writes a GGUF."
---

# mojo-gguf-core

`mojo run -I <repo>/src app.mojo`. Package name `gguf`.

    from gguf import Gguf, Writer, T_UINT32
    var g = Gguf("m.gguf")                         # opens a 20 GB file in 0.04 s
    var arch = g.str(g.find("general.architecture"))
    var w = g.data(g.tensor("token_embd.weight"))  # Span into the mmap; keep g alive

Writer: `add_int/add_float/add_str/add_strs/add_ints/add_floats/add_raw/add_tensor`, `save`.
Check: `scripts/check.sh` ends `PASS all`.
