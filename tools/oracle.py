"""Oracle: tests/cases.mojo's operations with gguf-py (the llama.cpp project's Python package).
Run with: uv run --no-project --with gguf==0.19.0 python -I tools/oracle.py OP ARGS"""
import struct, sys
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType as V, GGML_QUANT_SIZES, GGMLQuantizationType as Q

def fnv(b):
    h = 0xCBF29CE484222325
    for x in bytes(b):
        h = ((h ^ x) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h

def esc(s):
    return s.replace("\\", "\\\\").replace("\n", "\\n")

def mojo_float(x):
    """Mojo's String(Float64): repr, with a trailing .0 kept and exponent written as Python does."""
    return repr(float(x))

NAMES = ["u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool", "str", "arr", "u64", "i64", "f64"]
FIX = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

def payload(name, n):
    return bytes((k * 7 + len(name.encode())) % 251 for k in range(n))

op, out = sys.argv[1], []
try:
    if op == "read":
        raw = open(sys.argv[2], "rb").read()
        r = GGUFReader(sys.argv[2])
        fields = [f for f in r.fields.values() if not f.name.startswith("GGUF.")]
        out.append("gguf %d %d %d align %d data %d" % (r.fields["GGUF.version"].parts[0][0], len(r.tensors), len(fields), r.alignment, r.data_offset))
        for f in fields:
            start = f.offset + 8 + len(f.name.encode()) + 4
            end = f.offset + sum(int(p.nbytes) for p in f.parts)
            t = f.types[0]
            at = int(f.types[1]) if t == V.ARRAY else -1
            if t == V.STRING:
                v = esc(bytes(f.parts[-1]).decode("utf-8", "replace"))
            elif t == V.ARRAY:
                n = int(f.parts[4][0])
                v = "n=%d" % n
                if at == V.STRING:
                    v += " last=" + (esc(bytes(f.parts[-1]).decode("utf-8", "replace")) if n else "")
                elif at not in (V.ARRAY, V.FLOAT32, V.FLOAT64):
                    v += " last=" + (str(int(f.parts[-1][0])) if n else "")
            elif t in (V.FLOAT32, V.FLOAT64):
                v = mojo_float(f.parts[-1][0])
            else:
                v = str(int(f.parts[-1][0]))
            out.append("kv %s %d %d %d %s" % (f.name, int(t), at, fnv(raw[start:end]), v))
        for t in r.tensors:
            out.append("t %s %d %s %d %d %d" % (t.name, int(t.tensor_type), ",".join(str(int(d)) for d in t.shape),
                       t.data_offset - r.data_offset, t.n_bytes, fnv(raw[t.data_offset:t.data_offset + t.n_bytes])))
    elif op in ("copy", "build"):
        w = GGUFWriter(sys.argv[3], arch=None)
        w.kv_data[0].clear()  # no general.architecture: only what the source or spec holds
        tensors = []
        if op == "copy":
            r = GGUFReader(sys.argv[2])
            for f in r.fields.values():
                if f.name.startswith("GGUF."):
                    continue
                t = f.types[0]
                if t == V.ARRAY:
                    et = f.types[1]
                    if et == V.ARRAY:
                        raise ValueError("nested arrays are not rebuilt by the oracle")
                    vals = [bytes(f.parts[i]).decode() if et == V.STRING else f.parts[i][0].item() for i in f.data]
                    w.add_key_value(f.name, vals, V.ARRAY, sub_type=et)
                else:
                    w.add_key_value(f.name, bytes(f.parts[-1]).decode() if t == V.STRING else f.parts[-1][0].item(), t)
                if f.name == "general.alignment":
                    w.data_alignment = int(f.parts[-1][0])
            for t in r.tensors:
                tensors.append((t.name, int(t.tensor_type), [int(d) for d in t.shape], bytes(r.data[t.data_offset:t.data_offset + t.n_bytes])))
        else:
            for line in open(sys.argv[2]).read().splitlines():
                f = line.split(" ", 3)
                if len(f) < 3:
                    continue
                kind, key, typ = f[0], f[1], f[2]
                rest = f[3] if len(f) > 3 else ""
                if kind == "kv":
                    t = V(NAMES.index(typ))
                    val = rest.replace("\\n", "\n") if t == V.STRING else (float(rest) if t in (V.FLOAT32, V.FLOAT64) else (bool(int(rest)) if t == V.BOOL else int(rest)))
                    w.add_key_value(key, val, t)
                    if key == "general.alignment":
                        w.data_alignment = val
                elif kind == "arr":
                    t = V(NAMES.index(typ))
                    vals = rest.split("|") if t == V.STRING else [float(x) if t in (V.FLOAT32, V.FLOAT64) else int(x) for x in rest.split(",")]
                    w.add_key_value(key, vals, V.ARRAY, sub_type=t)
                else:
                    tt = int(typ); d = [int(x) for x in rest.split(",")]
                    bs, ts = GGML_QUANT_SIZES[Q(tt)]
                    n = int(np.prod(d))
                    tensors.append((key, tt, d, payload(key, n // bs * ts)))
        for name, tt, d, data in tensors:
            bs, ts = GGML_QUANT_SIZES[Q(tt)]
            shape = list(reversed(d))
            arr = np.frombuffer(data, dtype=np.uint8).reshape(shape[:-1] + [shape[-1] // bs * ts] if d else [len(data)])
            w.add_tensor(name, arr, raw_dtype=Q(tt))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        out = ["ok"]
    else:
        out = ["%d %d %d" % (int(k), v[0], v[1]) for k, v in sorted(GGML_QUANT_SIZES.items())]
except Exception as e:
    out = ["error"]
    if "-v" in sys.argv: raise
print("\n".join(out))
