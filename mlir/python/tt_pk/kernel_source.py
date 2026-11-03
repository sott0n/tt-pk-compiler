"""Load a compute-kernel C++ body for inline embedding in TTNN text.

The compiler never links against Resident; it transports a C++ source
string through the IR. A caller feeds the body as a path or as raw text,
and this module provides the string that `#ttnn.source_compute_kernel`
wants plus the MLIR-string escaping to splice it into the emitted text.
"""

from __future__ import annotations

import os
from typing import Union

PathOrText = Union[str, "os.PathLike[str]"]


class KernelSourceError(RuntimeError):
    pass


def load(source: PathOrText) -> str:
    """Return the C++ body.

    A path to an existing file is read; anything else is treated as the
    body itself. The compiler accepts both so callers can pin kernels to
    files on disk or carry snippets inline.
    """
    if isinstance(source, (bytes,)):
        raise KernelSourceError("expected str or PathLike, got bytes")
    text = os.fspath(source) if hasattr(source, "__fspath__") else source
    if os.sep in text or text.endswith(".cpp") or text.endswith(".h"):
        if not os.path.isfile(text):
            raise KernelSourceError(f"kernel source file not found: {text}")
        with open(text, "r", encoding="utf-8") as f:
            return f.read()
    return text


def escape_for_mlir(body: str) -> str:
    """Escape a C++ body for embedding in an MLIR `"..."` StringAttr.

    MLIR string literals accept `\\\\`, `\\"`, `\\n`, `\\t`, `\\r`, and
    `\\xx` hex escapes. Bytes that fall outside printable ASCII become
    hex escapes so the result stays round-trippable through ttmlir-opt.
    """
    out = []
    for c in body:
        o = ord(c)
        if c == "\\":
            out.append("\\\\")
        elif c == "\"":
            out.append("\\\"")
        elif c == "\n":
            out.append("\\n")
        elif c == "\t":
            out.append("\\t")
        elif c == "\r":
            out.append("\\r")
        elif 0x20 <= o <= 0x7E:
            out.append(c)
        else:
            out.append(f"\\{o:02x}")
    return "".join(out)
