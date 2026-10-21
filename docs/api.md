# API

`from gguf import Gguf, KV, TensorInfo, Writer, type_size, pad_to, T_UINT8 ... T_FLOAT64`

Value type constants: `T_UINT8 0, T_INT8 1, T_UINT16 2, T_INT16 3, T_UINT32 4, T_INT32 5,
T_FLOAT32 6, T_BOOL 7, T_STRING 8, T_ARRAY 9, T_UINT64 10, T_INT64 11, T_FLOAT64 12`.

## Gguf(path, known_types=True) raises

`known_types=False` accepts ggml type ids this library does not know (forks such as PQ2_0 = 142):
those tensors get `nbytes = -1`, no bounds check, and an empty `data()`.

mmaps the whole file read-only (unmapped when the value is destroyed) and parses it. Errors are
`gguf: <path>: <what>`: `cannot open`, `bad magic`, `big-endian files are not supported`,
`version N not supported`, `truncated at byte N`, `truncated string ...`, `truncated array ...`,
`unknown value type N`, `duplicate key K`, `general.alignment is not uint32`,
`alignment N is not a power of two`, `tensor X has unknown type N`,
`tensor X runs past the end of the file`, `duplicate tensor X`.

| Member | What |
|---|---|
| `version`, `alignment`, `data_offset`, `size`, `path` | header facts; `data_offset` is the padded end of the tensor table |
| `kvs: List[KV]` | file order. `KV`: `key`, `vtype`, `atype` (array element type, -1 otherwise), `count` (array length, 1 otherwise), `start`/`end` (payload span in the file) |
| `find(key) -> Int` | index into `kvs`, -1 when absent |
| `int(i)`, `float(i)`, `str(i)` raises | scalar values; `int` takes integer and bool types (signed ones sign-extended), `float` takes any number |
| `strs(i)`, `ints(i)` raises | elements of a string array, of an integer or bool array |
| `value(i) -> Span[UInt8]` | the raw payload of `kvs[i]` after its type tag |
| `tensors: List[TensorInfo]` | file order. `name`, `dims` (ne0 first), `ttype`, `offset` (in the data section), `nbytes`, `numel()` |
| `tensor(name) -> Int` | index into `tensors`, -1 when absent |
| `data(i) -> Span[UInt8]` | the tensor bytes inside the mapping; valid while the `Gguf` lives |
| `bytes() -> Span[UInt8]` | the whole file |

## Writer

| Member | What |
|---|---|
| `add_int(key, vtype, v)` | any integer type or `T_BOOL` |
| `add_float(key, vtype, v)` | `T_FLOAT32` or `T_FLOAT64` |
| `add_str(key, v)`, `add_strs(key, list)` | string, string array |
| `add_ints(key, etype, list)`, `add_floats(key, etype, list)` | numeric arrays |
| `add_raw(key, vtype, payload)` | an encoded value, for example `Gguf.value(i)`; the only way to write nested arrays |
| `add_tensor(name, dims, ttype, data)` | copies `data`; raises unless it is exactly the size `dims` need for `ttype` and ne0 is a multiple of the block size; duplicate names raise |
| `save(path)` | GGUF v3 |

Adding a key again replaces its value and keeps its position. `general.alignment` (uint32,
power of two) also sets the alignment the file is written with (default 32).

## type_size(ttype) -> (block, bytes)

Elements per block and bytes per block for a ggml type id; `(0, 0)` when unknown.
