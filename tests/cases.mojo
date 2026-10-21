# read FILE       -> header line, "kv ..." per KV, "t ..." per tensor (file order); or "error"
# copy IN OUT     -> IN re-emitted through Writer.add_raw/add_tensor
# build SPEC OUT  -> a file from tests/specs/*.txt through the typed Writer.add_* calls
# types           -> "ID BLOCK BYTES" for every known ggml type
# tools/oracle.py does the same with gguf-py.
from std.sys import argv
from gguf import Gguf, Writer, type_size, T_STRING, T_ARRAY, T_FLOAT32, T_FLOAT64, T_UINT32


def fnv(data: Span[UInt8, _]) -> UInt64:
    var h = UInt64(0xCBF29CE484222325)
    for b in data:
        h = (h ^ UInt64(b)) * 0x100000001B3
    return h


def esc(s: String) -> String:
    return s.replace("\\", "\\\\").replace("\n", "\\n")


def dims(d: List[Int]) -> String:
    var s = String()
    for i in range(len(d)):
        s += ("," if i else "") + String(d[i])
    return s


def payload(name: String, n: Int) -> List[UInt8]:
    var out = List[UInt8](capacity=n)
    for k in range(n):
        out.append(UInt8((k * 7 + name.byte_length()) % 251))
    return out^


def vtype(name: String) raises -> Int:
    var names: List[String] = ["u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool", "str", "arr", "u64", "i64", "f64"]
    for i in range(len(names)):
        if names[i] == name:
            return i
    raise Error("unknown type " + name)


def main() raises:
    var a = argv()
    var op = String(a[1])
    var out = List[String]()
    try:
        if op == "read":
            var g = Gguf(String(a[2]))
            out.append("gguf " + String(g.version) + " " + String(len(g.tensors)) + " " + String(len(g.kvs)) + " align " + String(g.alignment) + " data " + String(g.data_offset))
            for i in range(len(g.kvs)):
                ref kv = g.kvs[i]
                var v: String
                if kv.vtype == T_STRING:
                    v = esc(g.str(i))
                elif kv.vtype == T_ARRAY:
                    v = "n=" + String(kv.count)
                    if kv.atype == T_STRING:
                        v += " last=" + (esc(g.strs(i)[kv.count - 1]) if kv.count else "")
                    elif kv.atype != T_ARRAY and kv.atype != T_FLOAT32 and kv.atype != T_FLOAT64:
                        v += " last=" + (String(g.ints(i)[kv.count - 1]) if kv.count else "")
                elif kv.vtype == T_FLOAT32 or kv.vtype == T_FLOAT64:
                    v = String(g.float(i))
                else:
                    v = String(g.int(i))
                out.append("kv " + kv.key + " " + String(kv.vtype) + " " + String(kv.atype) + " " + String(fnv(g.value(i))) + " " + v)
            for i in range(len(g.tensors)):
                ref t = g.tensors[i]
                out.append("t " + t.name + " " + String(t.ttype) + " " + dims(t.dims) + " " + String(t.offset) + " " + String(t.nbytes) + " " + String(fnv(g.data(i))))
        elif op == "copy":
            var g = Gguf(String(a[2]))
            var w = Writer()
            for i in range(len(g.kvs)):
                w.add_raw(g.kvs[i].key, g.kvs[i].vtype, g.value(i))
            for i in range(len(g.tensors)):
                w.add_tensor(g.tensors[i].name, g.tensors[i].dims, g.tensors[i].ttype, g.data(i))
            w.save(String(a[3]))
            out.append("ok")
        elif op == "build":
            var w = Writer()
            for line in open(String(a[2]), "r").read().splitlines():
                var f = String(line).split(" ", 3)
                if len(f) < 3:
                    continue
                var kind = String(f[0])
                var key = String(f[1])
                var rest = String(f[3]) if len(f) > 3 else ""
                if kind == "kv":
                    var t = vtype(String(f[2]))
                    if t == T_STRING:
                        w.add_str(key, rest.replace("\\n", "\n"))
                    elif t == T_FLOAT32 or t == T_FLOAT64:
                        w.add_float(key, t, Float64(rest))
                    else:
                        w.add_int(key, t, Int(rest))
                elif kind == "arr":
                    var t = vtype(String(f[2]))
                    if t == T_STRING:
                        var xs = List[String]()
                        for x in rest.split("|"):
                            xs.append(String(x))
                        w.add_strs(key, xs)
                    elif t == T_FLOAT32 or t == T_FLOAT64:
                        var xs = List[Float64]()
                        for x in rest.split(","):
                            xs.append(Float64(String(x)))
                        w.add_floats(key, t, xs)
                    else:
                        var xs = List[Int]()
                        for x in rest.split(","):
                            xs.append(Int(String(x)))
                        w.add_ints(key, t, xs)
                else:  # tensor NAME TTYPE ne0,ne1
                    var d = List[Int]()
                    for x in String(f[3]).split(","):
                        d.append(Int(String(x)))
                    var n = 1
                    for x in d:
                        n *= x
                    var bs = type_size(Int(String(f[2])))
                    w.add_tensor(key, d, Int(String(f[2])), payload(key, n // bs[0] * bs[1]))
            w.save(String(a[3]))
            out.append("ok")
        else:
            for t in range(64):
                var bs = type_size(t)
                if bs[0]:
                    out.append(String(t) + " " + String(bs[0]) + " " + String(bs[1]))
    except e:
        out = ["error"]
    for l in out:
        print(l)
