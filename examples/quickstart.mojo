from gguf import Gguf, Writer, T_UINT32, T_FLOAT32


def main() raises:
    var w = Writer()
    w.add_str("general.name", "demo")
    w.add_int("demo.context_length", T_UINT32, 4096)
    var data = List[UInt8](length=4 * 8, fill=0)
    w.add_tensor("weight", [4, 2], 0, data)  # F32, ne0 = 4, ne1 = 2
    w.save(".work/quick.gguf")

    var g = Gguf(".work/quick.gguf")  # mmapped; data() is a view, no copy
    print(g.str(g.find("general.name")), g.int(g.find("demo.context_length")))
    for i in range(len(g.tensors)):
        print(g.tensors[i].name, g.tensors[i].ttype, g.tensors[i].numel(), len(g.data(i)))
