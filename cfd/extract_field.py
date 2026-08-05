#!/usr/bin/env python3
"""
Extract cell-centred fields (x,y,z, u,v,w, T, p, k, rho) from a Fluent CFF
.cas.h5 / .dat.h5 pair into a compact .npz for post-processing and surrogate
training. Cell centroids are reconstructed from the face/node topology using
the identity: the mean of a tetrahedron's four face centroids is its centroid.
"""
import h5py, numpy as np, sys, json, os

def load_case(base, U, V, Tdis=313.0, Tamb=298.0):
    cas = base + ".cas.h5"; dat = base + ".dat.h5"
    with h5py.File(cas, "r") as f:
        coords = f["meshes/1/nodes/coords/1"][:]                 # (Nn,3)
        fn = f["meshes/1/faces/nodes/1/nodes"][:].astype(np.int64)   # flat, 1-based
        nnodes = f["meshes/1/faces/nodes/1/nnodes"][:].astype(np.int64)
        c0 = f["meshes/1/faces/c0/1"][:].astype(np.int64)        # (Nf,) 1-based
        c1 = f["meshes/1/faces/c1/1"][:].astype(np.int64)        # (Nint,) 1-based
    assert np.all(nnodes == 3), "expected all triangular faces"
    Nf = c0.shape[0]; Nint = c1.shape[0]
    fnodes = fn.reshape(Nf, 3) - 1
    fc = coords[fnodes].mean(axis=1)                             # face centroids (Nf,3)

    Ncell = int(max(c0.max(), c1.max()))
    accum = np.zeros((Ncell + 1, 3)); cnt = np.zeros(Ncell + 1)
    np.add.at(accum, c0, fc);              np.add.at(cnt, c0, 1.0)
    np.add.at(accum, c1, fc[:Nint]);       np.add.at(cnt, c1, 1.0)
    cent = accum[1:] / cnt[1:, None]                            # (Ncell,3) 0-based cell i-1
    ncell_faces = cnt[1:]

    with h5py.File(dat, "r") as f:
        cg = f["results/1/phase-1/cells"]
        def g(name): return cg[name]["1"][:]
        T = g("SV_T"); u = g("SV_U"); v = g("SV_V"); w = g("SV_W")
        p = g("SV_P"); rho = g("SV_DENSITY"); k = g("SV_K")
    out = dict(xyz=cent, T=T, u=u, v=v, w=w, p=p, rho=rho, k=k,
               U_inf=np.float64(U), V_dis=np.float64(V),
               Tdis=np.float64(Tdis), Tamb=np.float64(Tamb),
               ncell_faces=ncell_faces)
    return out

if __name__ == "__main__":
    base = sys.argv[1]; U = float(sys.argv[2]); V = float(sys.argv[3])
    o = load_case(base, U, V)
    outnpz = sys.argv[4] if len(sys.argv) > 4 else base + "_field.npz"
    np.savez_compressed(outnpz, **o)
    c = o["xyz"]
    print(json.dumps({
        "file": outnpz, "ncells": int(c.shape[0]),
        "x_range": [float(c[:,0].min()), float(c[:,0].max())],
        "y_range": [float(c[:,1].min()), float(c[:,1].max())],
        "z_range": [float(c[:,2].min()), float(c[:,2].max())],
        "T_range": [float(o["T"].min()), float(o["T"].max())],
        "speed_max": float(np.sqrt(o["u"]**2+o["v"]**2+o["w"]**2).max()),
        "cells_with_4_faces": int((o["ncell_faces"]==4).sum()),
    }, indent=0))
