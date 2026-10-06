"""Lecture des fichiers .somno exportés par l'app web ou l'app iOS.

Format : "SOMNO001" | uint32 LE longueur d'en-tête | en-tête JSON | sections float32 LE.
"""
import json
import struct

import numpy as np


def read_somno(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:8] != b"SOMNO001":
        raise ValueError("fichier .somno invalide")
    (hlen,) = struct.unpack("<I", raw[8:12])
    meta = json.loads(raw[12:12 + hlen].decode("utf-8"))
    base = 12 + hlen
    out = {"meta": meta}
    for s in meta["sections"]:
        a = np.frombuffer(raw, dtype="<f4", count=s["n"] * s["cols"], offset=base + s["offset"])
        out[s["kind"]] = dict(data=a.reshape(s["n"], s["cols"]).astype(np.float64), rate=s["rate"],
                              columns=s.get("columns"), epochs=s.get("epochs"))
    return out


def write_somno(path, meta, sections):
    """sections : liste de (kind, array (n, cols), rate, extra dict) — utile pour les tests."""
    blobs, descr, off = [], [], 0
    for kind, arr, rate, extra in sections:
        a = np.ascontiguousarray(arr, dtype="<f4")
        d = dict(kind=kind, n=a.shape[0], cols=a.shape[1], dtype="float32", offset=off, bytes=a.nbytes, rate=rate)
        d.update(extra or {})
        descr.append(d)
        blobs.append(a.tobytes())
        off += a.nbytes
    head = json.dumps(dict(meta, sections=descr, format="somno/1")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(b"SOMNO001" + struct.pack("<I", len(head)) + head + b"".join(blobs))
