# Builds only if the library's libc declarations do not clash with the standard library's own
# (FileHandle.seek declares lseek): a program that seeks a file and opens a GGUF must link.
from gguf import Gguf


def main() raises:
    with open(".work/fx/all.gguf", "r") as f:
        _ = f.seek(4)
        _ = f.read_bytes(4)
    var g = Gguf(".work/fx/all.gguf")
    print("coexist: ok", len(g.tensors))
