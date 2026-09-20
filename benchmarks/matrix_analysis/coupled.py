#!/usr/bin/env python3
"""Memory-mapped actions of a complete captured WAE matrix and its ILU factors."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CHUNK = 65536
PARTS = ("cross_rank", "frequency_shift", "directional", "boundary_trace", "local_factor")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def kernels():
    source = Path(__file__).with_name("coupled_kernels.cpp")
    build = ROOT / "build/coupled-kernels" / sha256(source)
    build.mkdir(parents=True, exist_ok=True)
    library = build / "coupled_kernels.so"
    if not library.exists():
        command = ["c++", "-std=c++17", "-O2", "-shared", "-fPIC", "-Wall", "-Wextra",
                   str(source), "-o", str(library)]
        (build / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (build / "build.log").open("w") as stream:
            subprocess.run(command, cwd=ROOT, env=dict(os.environ, TMPDIR=str(build)),
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        (build / "binary.sha256").write_text(sha256(library) + "\n")
    if sha256(library) != (build / "binary.sha256").read_text().strip():
        raise ValueError("analysis kernel binary changed")
    lib = ctypes.CDLL(str(library))
    index, pointer, integer = ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int
    lib.matrix_action.argtypes = [index, index, pointer, pointer, pointer, pointer,
                                  integer, pointer, pointer, integer]
    lib.factor_action.argtypes = [index, index, pointer, pointer, pointer, pointer,
                                  pointer, integer, pointer, pointer, integer]
    lib.matrix_parts.argtypes = [index, index, index, index, pointer, pointer, pointer,
                                 pointer, pointer, integer, pointer, pointer, pointer]
    for name in ("matrix_action", "factor_action", "matrix_parts"):
        getattr(lib, name).restype = None
    return lib, library


def address(array):
    return array.ctypes.data


def finite(array):
    return all(np.isfinite(array[k:k + CHUNK]).all() for k in range(0, len(array), CHUNK))


class CoupledSnapshot:
    """Natural global-node vector layout; original rank-local factor ordering."""

    def __init__(self, prefix):
        prefix = Path(prefix).resolve()
        manifests = sorted(prefix.parent.glob(prefix.name + "-rank-*.json"))
        if not manifests:
            raise ValueError("no matrix manifests: " + str(prefix))
        records = sorted(((json.loads(p.read_text()), p) for p in manifests),
                         key=lambda record: record[0]["rank"])
        first = records[0][0]
        if first["format"] != 1 or first["byte_order"] != sys.byteorder or first["scalar_bytes"] not in (4, 8):
            raise ValueError("expected native-endian version-1 float32/float64 snapshot")
        if [m["rank"] for m, _ in records] != list(range(first["ranks"])):
            raise ValueError("missing or duplicate ranks")
        self.ns, self.nd, self.nodes = first["ns"], first["nd"], first["global_nodes"]
        if min(self.ns, self.nd, self.nodes) <= 0:
            raise ValueError("nonpositive snapshot dimensions")
        self.bins = self.ns * self.nd
        self.size = self.nodes * self.bins
        if self.size > np.iinfo(np.int64).max:
            raise ValueError("snapshot exceeds supported signed-index range")
        self.dtype = np.dtype("f" + str(first["scalar_bytes"]))
        self.files = set(manifests)
        self.owners = np.full(self.nodes, np.iinfo(np.uint64).max, dtype=np.uint64)
        self.parts = []
        for meta, path in records:
            for key in ("format", "byte_order", "scalar_bytes", "ranks", "global_nodes", "ns", "nd", "dt", "preconditioner"):
                if meta[key] != first[key]:
                    raise ValueError("inconsistent snapshot field: " + key)
            n = meta["owned_nodes"]
            if n < 0 or n > self.nodes:
                raise ValueError("invalid owned-node count")
            def read(suffix, dtype=None, count=None):
                return self.read(path, suffix, dtype, count)
            nodes = read("nodes.u64", "u8")
            if len(nodes) < n or np.any(nodes >= self.nodes) or len(np.unique(nodes)) != len(nodes):
                raise ValueError("invalid local-to-global node map")
            owned = nodes[:n]
            if np.any(self.owners[owned] != np.iinfo(np.uint64).max):
                raise ValueError("duplicate ownership")
            self.owners[owned] = meta["rank"]
            ptr = read("indptr.u64", "u8", n * self.bins + 1)
            col = read("indices.u64", "u8", meta["nnz"])
            val = read("values.bin", count=meta["nnz"])
            fp = read("ilu-indptr.u64", "u8", n + 1)
            fc = read("ilu-indices.u64", "u8")
            fv = read("ilu-values.bin", count=len(fc) * self.bins)
            inv = read("ilu-inverse-diagonal.bin", count=n * self.bins)
            for offsets, columns, bound in ((ptr, col, self.size), (fp, fc, n)):
                if offsets[0] != 0 or offsets[-1] != len(columns):
                    raise ValueError("invalid CSR endpoints")
                if any(np.any(offsets[k:min(k + CHUNK, len(offsets) - 1)] >
                              offsets[k + 1:min(k + CHUNK, len(offsets) - 1) + 1])
                       for k in range(0, len(offsets) - 1, CHUNK)):
                    raise ValueError("decreasing CSR offsets")
                if any(np.any(columns[k:k + CHUNK] >= bound) for k in range(0, len(columns), CHUNK)):
                    raise ValueError("CSR column out of range")
            for i in range(n):
                columns = fc[int(fp[i]):int(fp[i + 1])]
                if np.count_nonzero(columns == i) != 1 or np.any(columns[:-1] >= columns[1:]):
                    raise ValueError("factor pattern must be sorted with one diagonal per row")
            kind = read("kind.u8", "u1", n * self.bins)
            if np.any(kind > 3) or np.any(inv <= 0):
                raise ValueError("invalid row kind or nonpositive reciprocal pivot")
            if not all(finite(array) for array in (val, fv, inv)):
                raise ValueError("nonfinite matrix or factor data")
            self.parts.append(dict(meta=meta, path=path, n=n, nodes=owned, ptr=ptr, col=col,
                                   val=val, fp=fp, fc=fc, fv=fv, inv=inv, kind=kind))
        if np.any(self.owners == np.iinfo(np.uint64).max):
            raise ValueError("owned nodes do not cover global mesh")
        self.lib, self.library = kernels()

    def read(self, manifest, suffix, dtype=None, count=None):
        path = manifest.with_name(manifest.stem + "-" + suffix)
        dtype = self.dtype if dtype is None else np.dtype(dtype)
        size = path.stat().st_size
        if size % dtype.itemsize or (count is not None and size != count * dtype.itemsize):
            raise ValueError("invalid array extent: " + str(path))
        self.files.add(path)
        return np.memmap(path, dtype=dtype, mode="r") if size else np.empty(0, dtype=dtype)

    def vector(self, suffix):
        result = np.empty(self.size)
        for part in self.parts:
            value = self.read(part["path"], suffix, count=part["n"] * self.bins)
            if not finite(value):
                raise ValueError("nonfinite vector: " + suffix)
            result.reshape(self.nodes, self.bins)[part["nodes"]] = value.reshape(part["n"], self.bins)
        return result

    def _vector(self, x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        if x.shape != (self.size,):
            raise ValueError("wrong vector dimensions")
        return x

    def action(self, x, mode=0):
        if mode not in (0, 1, 2, 3):
            raise ValueError("unknown matrix action")
        x = self._vector(x)
        y = np.zeros(self.size) if mode >= 2 else np.empty(self.size)
        for p in self.parts:
            self.lib.matrix_action(p["n"] * self.bins, self.bins, address(p["nodes"]),
                address(p["ptr"]), address(p["col"]), address(p["val"]), self.dtype.itemsize,
                address(x), address(y), mode)
        return y

    def factor(self, x, mode=0):
        if mode not in (0, 1, 2):
            raise ValueError("unknown factor action")
        x = self._vector(x)
        y = np.empty(self.size)
        for p in self.parts:
            self.lib.factor_action(p["n"], self.bins, address(p["nodes"]), address(p["fp"]),
                address(p["fc"]), address(p["fv"]), address(p["inv"]), self.dtype.itemsize,
                address(x), address(y), mode)
        return y

    def actions_by_part(self, x):
        x = self._vector(x)
        y = np.zeros((self.size, 5))
        for p in self.parts:
            self.lib.matrix_parts(p["n"] * self.bins, self.bins, self.nd, p["meta"]["rank"],
                address(self.owners), address(p["nodes"]), address(p["ptr"]), address(p["col"]),
                address(p["val"]), self.dtype.itemsize, address(p["kind"]), address(x), address(y))
        return y

    def verify(self):
        checks = []
        for k in range(3):
            x = self.vector(f"probe-{k}-input.bin")
            ax = self.vector(f"probe-{k}-action.bin")
            scale = np.maximum(self.action(np.abs(x), 1), np.finfo(self.dtype).tiny)
            matrix_gap = float(np.max(np.abs(self.action(x) - ax) / scale))
            del scale
            v = (self.vector("rhs.bin").astype(self.dtype) - ax.astype(self.dtype)).astype(float) if k == 0 else x
            expected = self.vector(f"probe-{k}-inverse.bin")
            inverse_gap = float(np.linalg.norm(self.factor(v) - expected) / max(np.linalg.norm(expected), 1e-300))
            expected_action = self.vector(f"probe-{k}-preconditioned-action.bin")
            product_gap = float(np.linalg.norm(self.action(expected) - expected_action) /
                                max(np.linalg.norm(expected_action), 1e-300))
            item = dict(probe=k, matrix_scaled_inf_gap=matrix_gap,
                        inverse_relative_l2_gap=inverse_gap, preconditioned_relative_l2_gap=product_gap)
            if not all(np.isfinite(gap) and gap <= 512 * np.finfo(self.dtype).eps
                       for gap in (matrix_gap, inverse_gap, product_gap)):
                raise ValueError("exported probe replay failed: " + str(item))
            checks.append(item)
        return checks
