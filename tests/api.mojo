# Getters and errors as documented; data() is a view into the mapping.
from gguf import Gguf, Writer, T_UINT32, T_FLOAT32


def main() raises:
    var g = Gguf(".work/fx/all.gguf")
    if g.str(g.find("general.name")) != "tiny test model" or g.int(g.find("a.i8")) != -100:
        raise Error("api: scalar getters")
    if g.strs(g.find("tokenizer.ggml.tokens"))[3] != "wörld" or g.ints(g.find("a.arr_i32"))[0] != -1:
        raise Error("api: array getters")
    var t = g.tensor("blk.0.q4.weight")
    if Int(g.data(t).unsafe_ptr()) != g.base + g.data_offset + g.tensors[t].offset or len(g.data(t)) != 72:
        raise Error("api: data is not a view into the mapping")
    var raised = False
    try:
        _ = g.int(g.find("general.name"))
    except:
        raised = True
    var w = Writer()
    var two: List[UInt8] = [0, 0]
    for bad in [(2, 1), (0, 8)]:  # Q4_0 with 1 element; F32 with 2 bytes
        try:
            w.add_tensor("x", [bad[0] + 1], bad[1], two)
            raised = False
        except:
            pass
    if not raised:
        raise Error("api: an invalid call did not raise")
    var lax = Gguf(".work/fx/bad_ttype.gguf", known_types=False)
    var u = lax.tensor("blk.0.i8")
    if lax.tensors[u].ttype != 4 or lax.tensors[u].nbytes != -1 or len(lax.data(u)) != 0:
        raise Error("api: known_types=False")
    print("api: ok")
