#!/usr/bin/env python3
"""
3D tetrahedral mesh generator for a generic cooling-tower array in an
atmospheric crosswind. Builds an atmospheric box minus 6 rectangular
cooling-tower cells (OCC boolean cut) in gmsh, then writes a native Fluent
.msh with named boundary zones (the meshio Fluent writer drops face zones,
so the exporter is hand-written and vectorized with numpy).

Zones written:
  inlet       velocity-inlet  (x = xmin, crosswind enters)
  outlet      pressure-outlet (x = xmax)
  sides       symmetry        (y = ymin, ymax)
  top         symmetry        (z = zmax)
  ground      wall            (z = 0, outside tower footprints)
  tower_top   velocity-inlet  (z = Ht faces of towers, warm plume discharge)
  tower_walls wall            (vertical faces of towers, intake surfaces)

Geometry is FIXED so the parametric sweep varies only boundary conditions,
which keeps one mesh for the whole matrix and a single spatial domain for the
downstream surrogate model.
"""
import gmsh, sys, json, math, os
import numpy as np

# ----------------------------------------------------------------------
# Geometry definition (metres)
# ----------------------------------------------------------------------
XMIN, XMAX = -60.0, 220.0
YMIN, YMAX = -70.0,  70.0
ZMIN, ZMAX =   0.0,  80.0
HW  = 6.0     # tower half-width (footprint 12 x 12 m)
HT  = 10.0    # tower height (m)
TOWER_XC = [0.0, 40.0, 80.0]   # 3 columns along wind
TOWER_YC = [-16.0, 16.0]       # 2 rows across wind
TOWERS = [(xc, yc) for xc in TOWER_XC for yc in TOWER_YC]  # 6 towers

EPS = 1e-3

def classify(cx, cy, cz, bb):
    """Return zone name for a boundary surface from its centre of mass + bbox."""
    x0,y0,z0,x1,y1,z1 = bb
    flat_x = abs(x1-x0) < EPS
    flat_y = abs(y1-y0) < EPS
    flat_z = abs(z1-z0) < EPS
    if flat_x and abs(cx-XMIN) < EPS: return "inlet"
    if flat_x and abs(cx-XMAX) < EPS: return "outlet"
    if flat_y and (abs(cy-YMIN) < EPS or abs(cy-YMAX) < EPS): return "sides"
    if flat_z and abs(cz-ZMAX) < EPS: return "top"
    if flat_z and abs(cz-ZMIN) < EPS: return "ground"
    if flat_z and abs(cz-HT)   < EPS: return "tower_top"
    # vertical face, 0 < z < HT, interior to domain -> tower side wall
    return "tower_walls"

def build(out, size_near=2.0, size_far=13.0, grow_dist=55.0):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("towerarray")
    occ = gmsh.model.occ

    dom = occ.addBox(XMIN, YMIN, ZMIN, XMAX-XMIN, YMAX-YMIN, ZMAX-ZMIN)
    tools = []
    for (xc, yc) in TOWERS:
        tools.append((3, occ.addBox(xc-HW, yc-HW, 0.0, 2*HW, 2*HW, HT)))
    occ.cut([(3, dom)], tools, removeObject=True, removeTool=True)
    occ.synchronize()

    # --- classify boundary surfaces -----------------------------------
    zones = {}
    for (d, tag) in gmsh.model.getEntities(2):
        com = occ.getCenterOfMass(2, tag)
        bb  = gmsh.model.getBoundingBox(2, tag)
        nm  = classify(com[0], com[1], com[2], bb)
        zones.setdefault(nm, []).append(tag)
    vols = [t for (d, t) in gmsh.model.getEntities(3)]

    def phys(dim, tags, name):
        g = gmsh.model.addPhysicalGroup(dim, tags)
        gmsh.model.setPhysicalName(dim, g, name)
    phys(3, vols, "fluid")
    for nm, tags in zones.items():
        phys(2, tags, nm)

    # --- size field: fine near towers, coarse in the far field --------
    tower_surf = zones.get("tower_top", []) + zones.get("tower_walls", [])
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", tower_surf)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 200)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", size_near)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", size_far)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 4.0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", grow_dist)
    # Box refinement over the plume / wake corridor (above and downwind of towers)
    size_box = max(size_near, 3.5)
    gmsh.model.mesh.field.add("Box", 3)
    gmsh.model.mesh.field.setNumber(3, "VIn", size_box)
    gmsh.model.mesh.field.setNumber(3, "VOut", size_far)
    gmsh.model.mesh.field.setNumber(3, "XMin", -15.0)
    gmsh.model.mesh.field.setNumber(3, "XMax", 160.0)
    gmsh.model.mesh.field.setNumber(3, "YMin", -42.0)
    gmsh.model.mesh.field.setNumber(3, "YMax", 42.0)
    gmsh.model.mesh.field.setNumber(3, "ZMin", 0.0)
    gmsh.model.mesh.field.setNumber(3, "ZMax", 45.0)
    gmsh.model.mesh.field.setNumber(3, "Thickness", 20.0)
    gmsh.model.mesh.field.add("Min", 4)
    gmsh.model.mesh.field.setNumbers(4, "FieldsList", [2, 3])
    gmsh.model.mesh.field.setAsBackgroundMesh(4)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)   # Delaunay
    do_netgen = os.environ.get("NETGEN", "0") == "1"
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1 if do_netgen else 0)
    gmsh.model.mesh.generate(3)
    if do_netgen:
        gmsh.model.mesh.optimize("Netgen")

    # --- pull nodes + tets --------------------------------------------
    ntags, ncoords, _ = gmsh.model.mesh.getNodes()
    ntags = np.array(ntags, dtype=np.int64)
    ncoords = np.array(ncoords).reshape(-1, 3)
    maxt = ntags.max()
    remap = np.zeros(maxt+1, dtype=np.int64)
    remap[ntags] = np.arange(1, len(ntags)+1)
    coords = np.empty((len(ntags), 3))
    coords[remap[ntags]-1] = ncoords
    N = len(ntags)

    tt, tn = gmsh.model.mesh.getElementsByType(4)      # tets
    T = remap[np.array(tn, dtype=np.int64).reshape(-1, 4)]   # (M,4) 1-based
    M = T.shape[0]

    # boundary triangle -> zone name (sorted node triple key)
    gname = {}
    for (d, tg) in gmsh.model.getPhysicalGroups(2):
        nm = gmsh.model.getPhysicalName(2, tg)
        for e in gmsh.model.getEntitiesForPhysicalGroup(2, tg):
            et, en = gmsh.model.mesh.getElementsByType(2, e)
            if len(et) == 0: continue
            tri = remap[np.array(en, dtype=np.int64).reshape(-1, 3)]
            for row in np.sort(tri, axis=1):
                gname[tuple(row)] = nm
    gmsh.finalize()

    return _write(out, coords, T, M, N, gname)

def _write(out, coords, T, M, N, gname):
    # 4 triangular faces per tet (local node combos)
    FL = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
    faces = T[:, FL].reshape(-1, 3)                 # (4M,3) global node ids
    owner = np.repeat(np.arange(1, M+1), 4)         # owner cell per face
    key = np.sort(faces, axis=1)
    order = np.lexsort((key[:,2], key[:,1], key[:,0]))
    ks = key[order]; fs = faces[order]; ow = owner[order]
    same = np.all(ks[1:] == ks[:-1], axis=1)
    paired = np.zeros(len(ks), dtype=bool)
    paired[:-1] |= same; paired[1:] |= same
    pf = np.where(same)[0]
    int_faces = fs[pf]; int_c0 = ow[pf]; int_c1 = ow[pf+1]
    bnd = ~paired
    bnd_faces = fs[bnd]; bnd_c0 = ow[bnd]; bnd_key = ks[bnd]

    # cell centroids for outward-normal orientation
    cent = coords[T-1].mean(axis=1)                 # (M,3)

    def orient(fac, c0):
        A = coords[fac[:,0]-1]; B = coords[fac[:,1]-1]; C = coords[fac[:,2]-1]
        n = np.cross(B-A, C-A)
        fc = (A+B+C)/3.0
        P = cent[c0-1]
        d = np.einsum('ij,ij->i', n, fc-P)
        flip = d < 0.0
        out = fac.copy()
        out[flip, 1], out[flip, 2] = fac[flip, 2], fac[flip, 1]
        return out
    int_faces = orient(int_faces, int_c0)
    bnd_faces = orient(bnd_faces, bnd_c0)

    names = ["inlet","outlet","sides","top","ground","tower_top","tower_walls"]
    btmap = {"interior":2,"inlet":10,"outlet":5,"sides":7,"top":7,
             "ground":3,"tower_top":10,"tower_walls":3}
    tnmap = {2:"interior",10:"velocity-inlet",5:"pressure-outlet",
             7:"symmetry",3:"wall"}
    idxby = {nm: [] for nm in names}
    miss = 0
    for i in range(bnd_key.shape[0]):
        nm = gname.get(tuple(bnd_key[i]))
        if nm is None:
            nm = "tower_walls"; miss += 1
        idxby[nm].append(i)

    Nint = int_faces.shape[0]
    Nf = Nint + bnd_faces.shape[0]
    Hx = lambda n: format(int(n), 'x')
    with open(out, 'w') as f:
        f.write('(0 "cooling-tower array  cells=%d nodes=%d")\n(2 3)\n' % (M, N))
        f.write('(10 (0 1 %s 0 3))\n' % Hx(N))
        f.write('(10 (1 1 %s 1 3)(\n' % Hx(N))
        np.savetxt(f, coords, fmt='%.9g %.9g %.9g')
        f.write('))\n')
        f.write('(12 (0 1 %s 0 0))\n(12 (2 1 %s 1 2))\n' % (Hx(M), Hx(M)))
        f.write('(13 (0 1 %s 0 0))\n' % Hx(Nf))
        start = 1; zid = 2; zinfo = []
        def block(name, fac, c0, c1):
            nonlocal start, zid
            if fac.shape[0] == 0: return
            zid += 1; end = start + fac.shape[0] - 1; bt = btmap[name]
            f.write('(13 (%s %s %s %s 3)(\n' % (Hx(zid), Hx(start), Hx(end), Hx(bt)))
            # Fluent wants the stored winding normal to point INTO owner c0,
            # i.e. reverse of geometric-outward (swap the last two nodes).
            arr = np.column_stack([fac[:,0], fac[:,2], fac[:,1], c0, c1])
            np.savetxt(f, arr, fmt='%x %x %x %x %x')
            f.write('))\n'); zinfo.append((zid, bt, name)); start = end + 1
        block("interior", int_faces, int_c0, int_c1)
        for nm in names:
            idx = np.array(idxby[nm], dtype=np.int64)
            if idx.size == 0: continue
            block(nm, bnd_faces[idx], bnd_c0[idx], np.zeros(idx.size, dtype=np.int64))
        for zid_, bt, nm in zinfo:
            f.write('(45 (%d %s %s)())\n' % (zid_, tnmap[bt], nm))
        f.write('(45 (2 fluid fluid)())\n')
    return dict(file=out, nodes=int(N), cells=int(M), faces=int(Nf),
                interior=int(Nint), unmatched_boundary=int(miss),
                zones={nm: len(idxby[nm]) for nm in names})

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "tower_array.msh"
    sn = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    sf = float(sys.argv[3]) if len(sys.argv) > 3 else 13.0
    print(json.dumps(build(out, size_near=sn, size_far=sf), indent=0))
