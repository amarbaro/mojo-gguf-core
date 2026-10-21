# Design

Rules: the file is mapped, never read into buffers; the header is parsed in place; the writer
emits what gguf-py's GGUFWriter emits, byte for byte.

## mmap instead of a sliding window

mojo-gguf's first reader walked the header through a 256 KiB `pread` window. The mmap reader
replaces it and the iTools `gguf_mmap.mojo` cursor with one design. `KV` records only the
payload span (`start`, `end`), and getters decode from the mapping on demand. A 248k-token
vocabulary is one `strs()` call, nothing before it. `scripts/bench.sh` on a 20.9 GB model
(142 KVs, 733 tensors, vocabulary read) runs in 36 ms at 29 MB, against gguf-py's 5,535 ms at
863 MB.

## Validation is gguf-py's

Versions 2 and 3; non-power-of-two or non-uint32 `general.alignment`, unknown value or ggml
types, duplicate keys and tensors, truncation and data past the end all raise. Big-endian files
raise too (gguf-py byte-swaps them; nobody ships them). Each bound is checked before a length is
used, so a corrupt count cannot make the walk allocate or run off the mapping.

## The ggml type table

Two flat compile-time lists indexed by type id give the block size and bytes per block. The
gate compares all 34 entries with gguf-py's `GGML_QUANT_SIZES`, and a mutant changing one entry
must fail.

## Writer

KVs are stored encoded (type tag + payload), in first-added order; adding a key again replaces
it in place, as a Python dict assignment does. Tensor data is copied into one blob; `save`
writes header, KVs, tensor infos with padded offsets, header padding (even with no tensors),
then each tensor followed by its padding. `add_raw` takes `Gguf.value(i)` unchanged, so a copy
of any file, nested arrays included, needs no decoding.
