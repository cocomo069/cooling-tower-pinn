#!/usr/bin/env python3
"""
Physics-informed parametric surrogate for the cooling-tower array.

Maps  (x, y, z, U_inf, V_dis)  ->  (u, v, w, theta)  where theta is the
normalised temperature (T - T_amb)/(T_dis - T_amb).  Trained on a handful of
steady RANS cases from Fluent and constrained by a mass-conservation
(divergence-free) physics residual evaluated at random collocation points and
random operating conditions, so it generalises across the (wind, discharge)
design space rather than only interpolating the sampled cells.

Backend: autograd (reverse-mode AD) + a small manual Adam loop, so it runs
anywhere without a heavyweight DL install. The divergence residual uses one
level of nested AD (d(output)/d(input) inside a loss that is itself
differentiated w.r.t. the network weights).

Held-out generalisation test: the network is trained WITHOUT the centre case
(c1: U=3, V=6) and then asked to predict that entire 3D field, which it has
never seen, from only the surrounding corner/edge cases.
"""
import autograd.numpy as anp
from autograd import grad
import numpy as np
import glob, os, json, sys, time

# ----------------------------------------------------------------------
# Normalisation constants (fixed, physical -> ~[-1,1])
# ----------------------------------------------------------------------
X0 = np.array([80.0, 0.0, 40.0, 3.75, 6.0])     # centres
XS = np.array([140.0, 70.0, 40.0, 2.25, 3.0])   # half-ranges
U_SCALE = 10.0                                   # velocity scale (m/s)
TAMB, TDIS = 298.0, 313.0
# domain + tower geometry (must match mesh_gen.py)
DOM = dict(xmin=-60, xmax=220, ymin=-70, ymax=70, zmin=0, zmax=80)
TOWER_XC = [0.0, 40.0, 80.0]; TOWER_YC = [-16.0, 16.0]; HW = 6.0; HT = 10.0

def norm_inputs(x, y, z, U, V):
    X = np.stack([x, y, z, np.full_like(x, U), np.full_like(x, V)], axis=1)
    return (X - X0) / XS

def in_tower(x, y, z):
    m = np.zeros(len(x), dtype=bool)
    for xc in TOWER_XC:
        for yc in TOWER_YC:
            m |= (np.abs(x-xc) < HW) & (np.abs(y-yc) < HW) & (z < HT)
    return m

# ----------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------
def init_params(layers, seed=0):
    rng = np.random.RandomState(seed)
    P = []
    for a, b in zip(layers[:-1], layers[1:]):
        W = rng.randn(a, b) * np.sqrt(2.0/(a+b))
        bvec = np.zeros(b)
        P.append((W, bvec))
    return P

_FREQS = np.array([1.0, 2.0, 4.0, 8.0])
def features(X):
    """Fourier-encode the 3 spatial inputs (sharp plume gradients); keep U,V raw."""
    sp = X[:, :3]; uv = X[:, 3:]
    enc = [sp, uv]
    for fq in _FREQS:
        enc.append(anp.sin(fq*np.pi*sp)); enc.append(anp.cos(fq*np.pi*sp))
    return anp.concatenate(enc, axis=1)          # dim = 5 + 6*len(_FREQS)
FEAT_DIM = 5 + 6*len(_FREQS)

def net(params, X):
    H = features(X)
    for W, b in params[:-1]:
        H = anp.tanh(anp.dot(H, W) + b)
    W, b = params[-1]
    return anp.dot(H, W) + b          # (N,4): u', v', w', theta

# ----------------------------------------------------------------------
# Divergence residual (physical, up to constant U_SCALE)
#   du/dx + dv/dy + dw/dz, with chain-rule factors from normalised coords
# ----------------------------------------------------------------------
# physical central-difference steps (m) -> normalised steps
_DX = np.array([1.0, 1.0, 0.5])
def _shift(X, axis, hn):
    d = anp.zeros(X.shape[1]); d[axis] = hn
    return X + d
def divergence(params, Xc):
    # central finite differences of the network's velocity field: no nested AD,
    # so the outer gradient w.r.t weights stays single-level and fast.
    dudx = (net(params, _shift(Xc, 0,  _DX[0]/XS[0]))[:, 0]
            - net(params, _shift(Xc, 0, -_DX[0]/XS[0]))[:, 0]) / (2*_DX[0])
    dvdy = (net(params, _shift(Xc, 1,  _DX[1]/XS[1]))[:, 1]
            - net(params, _shift(Xc, 1, -_DX[1]/XS[1]))[:, 1]) / (2*_DX[1])
    dwdz = (net(params, _shift(Xc, 2,  _DX[2]/XS[2]))[:, 2]
            - net(params, _shift(Xc, 2, -_DX[2]/XS[2]))[:, 2]) / (2*_DX[2])
    return (dudx + dvdy + dwdz) * U_SCALE   # (N,) physical 1/s

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_dataset(datadir, test_tag="c1"):
    files = sorted(glob.glob(os.path.join(datadir, "c*_field.npz")))
    train, test = [], None
    for fp in files:
        tag = os.path.basename(fp).split("_")[0]
        d = np.load(fp)
        xyz = d["xyz"]; U = float(d["U_inf"]); V = float(d["V_dis"])
        rec = dict(tag=tag, xyz=xyz, u=d["u"], v=d["v"], w=d["w"],
                   T=d["T"], U=U, V=V)
        if tag == test_tag:
            test = rec
        else:
            train.append(rec)
    return train, test

def build_training_arrays(train, per_case=2500, seed=1, active_frac=0.6):
    """Stratified sampling: oversample the thermally-active plume/near-field
    (theta>0.02) so the surrogate learns the sparse hot region, not just the
    dominant ambient far-field."""
    rng = np.random.RandomState(seed)
    Xs, Ys = [], []
    for rec in train:
        n = rec["xyz"].shape[0]
        th_all = (rec["T"]-TAMB)/(TDIS-TAMB)
        active = np.where(th_all > 0.02)[0]
        ambient = np.where(th_all <= 0.02)[0]
        n_act = min(int(per_case*active_frac), len(active))
        n_amb = min(per_case-n_act, len(ambient))
        idx = np.concatenate([rng.choice(active, n_act, replace=False),
                              rng.choice(ambient, n_amb, replace=False)])
        x, y, z = rec["xyz"][idx].T
        Xn = norm_inputs(x, y, z, rec["U"], rec["V"])
        u = rec["u"][idx]/U_SCALE; v = rec["v"][idx]/U_SCALE; w = rec["w"][idx]/U_SCALE
        th = th_all[idx]
        Xs.append(Xn); Ys.append(np.stack([u, v, w, th], axis=1))
    return np.vstack(Xs), np.vstack(Ys)

def sample_collocation(n, seed=2):
    rng = np.random.RandomState(seed)
    pts = np.empty((0, 3))
    while len(pts) < n:
        x = rng.uniform(DOM["xmin"], DOM["xmax"], n)
        y = rng.uniform(DOM["ymin"], DOM["ymax"], n)
        z = rng.uniform(DOM["zmin"], DOM["zmax"], n)
        keep = ~in_tower(x, y, z)
        pts = np.vstack([pts, np.stack([x, y, z], axis=1)[keep]])
    pts = pts[:n]
    U = rng.uniform(1.5, 6.0, n); V = rng.uniform(3.0, 9.0, n)
    return (pts[:, 0]-X0[0])/XS[0], pts, U, V

# ----------------------------------------------------------------------
# Data cache (deterministic) + loss builder
# ----------------------------------------------------------------------
def make_batches(datadir, per_case=2500, n_coll=800, test_tag="c1"):
    train, test = load_dataset(datadir, test_tag)
    Xd, Yd = build_training_arrays(train, per_case=per_case)
    _, cpts, cU, cV = sample_collocation(n_coll)
    Xc = np.stack([(cpts[:,0]-X0[0])/XS[0], (cpts[:,1]-X0[1])/XS[1],
                   (cpts[:,2]-X0[2])/XS[2], (cU-X0[3])/XS[3],
                   (cV-X0[4])/XS[4]], axis=1)
    return anp.array(Xd), anp.array(Yd), anp.array(Xc), train, test

_COLW = anp.array([1.0, 1.0, 1.0, 4.0])   # weight temperature (col 3) more
def make_loss(Xd_a, Yd_a, Xc_a, w_phys):
    def loss(params):
        pred = net(params, Xd_a)
        data = anp.mean(((pred - Yd_a)**2) * _COLW)
        div = divergence(params, Xc_a)
        return data + w_phys*anp.mean(div**2)
    return loss

# ----------------------------------------------------------------------
# Checkpointed Adam trainer (resumable across separate invocations)
# ----------------------------------------------------------------------
def _flatten(params):
    return [np.asarray(x) for WB in params for x in WB]
def _to_params(flat):
    return [(flat[i], flat[i+1]) for i in range(0, len(flat), 2)]

def save_ckpt(path, params, mW, vW, it, hist, layers):
    fl = _flatten(params)
    mf = [x for WB in mW for x in WB]; vf = [x for WB in vW for x in WB]
    d = {f"p{i}": a for i, a in enumerate(fl)}
    d.update({f"m{i}": a for i, a in enumerate(mf)})
    d.update({f"v{i}": a for i, a in enumerate(vf)})
    d["it"] = np.array(it); d["hist"] = np.array(hist) if hist else np.zeros((0,2))
    d["layers"] = np.array(layers)
    np.savez(path, **d)

def load_ckpt(path):
    d = np.load(path)
    n = len([k for k in d.files if k.startswith("p")])
    fl = [d[f"p{i}"] for i in range(n)]
    mf = [d[f"m{i}"] for i in range(n)]
    vf = [d[f"v{i}"] for i in range(n)]
    params = _to_params(fl); mW = _to_params(mf); vW = _to_params(vf)
    return params, mW, vW, int(d["it"]), d["hist"].tolist(), list(d["layers"])

def fit_chunk(datadir, ckpt, add_epochs, layers=(FEAT_DIM,96,96,4), w_phys=1.0,
              lr=2e-3, per_case=2200, n_coll=600, test_tag="c1", seed=0):
    Xd, Yd, Xc, train, test = make_batches(datadir, per_case, n_coll, test_tag)
    loss = make_loss(Xd, Yd, Xc, w_phys)
    gloss = grad(loss)
    b1, b2, eps = 0.9, 0.999, 1e-8
    if os.path.exists(ckpt):
        params, mW, vW, it0, hist, layers = load_ckpt(ckpt)
    else:
        params = init_params(list(layers), seed)
        mW = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
        vW = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
        it0, hist = 0, []
    t0 = time.time()
    for k in range(1, add_epochs+1):
        it = it0 + k
        g = gloss(params); newp = []
        for i, ((W, b), (gW, gb)) in enumerate(zip(params, g)):
            mWW, mWb = mW[i]; vWW, vWb = vW[i]
            mWW = b1*mWW + (1-b1)*gW; mWb = b1*mWb + (1-b1)*gb
            vWW = b2*vWW + (1-b2)*gW**2; vWb = b2*vWb + (1-b2)*gb**2
            mW[i] = (mWW, mWb); vW[i] = (vWW, vWb)
            W = W - lr*(mWW/(1-b1**it))/(np.sqrt(vWW/(1-b2**it))+eps)
            b = b - lr*(mWb/(1-b1**it))/(np.sqrt(vWb/(1-b2**it))+eps)
            newp.append((W, b))
        params = newp
    l = float(loss(params))
    hist.append([it0+add_epochs, l])
    save_ckpt(ckpt, params, mW, vW, it0+add_epochs, hist, layers)
    ev = evaluate(params, test)
    dt = time.time()-t0
    print(f"epochs {it0}->{it0+add_epochs}  loss {l:.4e}  "
          f"T_r2 {ev['T_r2']:.3f}  spd_r2 {ev['speed_r2']:.3f}  "
          f"({dt:.0f}s, {dt/add_epochs*1000:.0f} ms/ep)", flush=True)
    return params, test, ev

def evaluate(params, test):
    x, y, z = test["xyz"].T
    Xn = norm_inputs(x, y, z, test["U"], test["V"])
    pred = net(params, anp.array(Xn))
    pred = np.array(pred)
    u_p, v_p, w_p = pred[:,0]*U_SCALE, pred[:,1]*U_SCALE, pred[:,2]*U_SCALE
    T_p = pred[:,3]*(TDIS-TAMB)+TAMB
    def metrics(a, b):
        rmse = float(np.sqrt(np.mean((a-b)**2)))
        ss = float(1 - np.sum((a-b)**2)/np.sum((b-np.mean(b))**2))
        return rmse, ss
    spd_p = np.sqrt(u_p**2+v_p**2+w_p**2)
    spd_c = np.sqrt(test["u"]**2+test["v"]**2+test["w"]**2)
    rT, r2T = metrics(T_p, test["T"])
    rS, r2S = metrics(spd_p, spd_c)
    return dict(T_rmse=rT, T_r2=r2T, speed_rmse=rS, speed_r2=r2S,
                T_pred=T_p, T_cfd=test["T"], spd_pred=spd_p, spd_cfd=spd_c)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "chunk"
    datadir = "../data"; ckpt = "pinn_ckpt_ff.npz"
    if mode == "chunk":
        add = int(sys.argv[2]) if len(sys.argv) > 2 else 150
        lr = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-3
        fit_chunk(datadir, ckpt, add, lr=lr)
    elif mode == "finalize":
        params, mW, vW, it, hist, layers = load_ckpt(ckpt)
        ev = evaluate(params, load_dataset(datadir, "c1")[1])
        print(json.dumps({k: float(v) for k, v in ev.items()
                          if not isinstance(v, np.ndarray)}, indent=2))
        flat = {f"W{i}": W for i, (W, b) in enumerate(params)}
        flat.update({f"b{i}": b for i, (W, b) in enumerate(params)})
        np.savez("pinn_params.npz", **flat)
        np.savez("pinn_eval.npz", T_pred=ev["T_pred"], T_cfd=ev["T_cfd"],
                 spd_pred=ev["spd_pred"], spd_cfd=ev["spd_cfd"],
                 hist=np.array(hist))
        print("saved pinn_params.npz, pinn_eval.npz  (total epochs %d)" % it)
