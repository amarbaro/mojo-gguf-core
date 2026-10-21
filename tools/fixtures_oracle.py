"""Builds the gate's GGUF files into DIR: each tests/specs/*.txt through gguf-py's GGUFWriter
(tools/oracle.py build), then corrupted copies of all.gguf that gguf-py must reject."""
import os, struct, subprocess, sys
D = sys.argv[1]
os.makedirs(D, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
for s in sorted(os.listdir(os.path.join(here, "..", "tests", "specs"))):
    out = os.path.join(D, s[:-4] + ".gguf")
    r = subprocess.run([sys.executable, "-I", os.path.join(here, "oracle.py"), "build", os.path.join(here, "..", "tests", "specs", s), out], capture_output=True, text=True)
    assert r.stdout.strip() == "ok", (s, r.stdout, r.stderr)
b = open(os.path.join(D, "all.gguf"), "rb").read()
def put(name, data):
    open(os.path.join(D, "bad_" + name + ".gguf"), "wb").write(data)
def find(s):
    i = b.index(s.encode()); return i
put("magic", b"GGUX" + b[4:])
put("v1", b[:4] + struct.pack("<I", 1) + b[8:])
put("v4", b[:4] + struct.pack("<I", 4) + b[8:])
put("bigendian", b[:4] + struct.pack(">I", 3) + b[8:])
put("trunc_kv", b[:find("a.i64") + 3])
put("trunc_data", b[:-40])
put("empty_file", b"")
k = find("a.u8") + 4  # value type of a.u8
put("vtype", b[:k] + struct.pack("<I", 13) + b[k + 4:])
t = find("blk.0.i8") + 8 + 4 + 8  # ggml type of blk.0.i8 (1 dim)
put("ttype", b[:t] + struct.pack("<I", 4) + b[t + 4:])
i = find("blk.0.q4.weight"); put("dup_tensor", b[:i] + b"tok_embd.weight" + b[i + 15:])
i = find("a.i16"); put("dup_key", b[:i] + b"a.u16" + b[i + 5:])
a = open(os.path.join(D, "align64.gguf"), "rb").read()
j = a.index(b"general.alignment") + 17 + 4
put("align3", a[:j] + struct.pack("<I", 3) + a[j + 4:])
put("align_type", a[:j - 4] + struct.pack("<I", 5) + a[j:])
put("count_lie", b[:16] + struct.pack("<Q", 40) + b[24:])
