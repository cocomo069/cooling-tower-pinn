#!/usr/bin/env python3
"""
Post-analysis of the tower-array CFD sweep:
  - per-tower intake recirculation coefficient RC = (T_intake - Tamb)/(Tdis - Tamb)
  - array-level worst / mean recirculation vs operating point
  - quantitative matplotlib figures (recirculation trends, PINN vs CFD parity,
    training-loss history)
  - a results table (JSON + CSV) consumed by the report/deck.
Hero 3D contour figures are produced separately in ParaView.
"""
import numpy as np, glob, os, json, csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAMB, TDIS = 298.0, 313.0
TOWER_XC = [0.0, 40.0, 80.0]; TOWER_YC = [-16.0, 16.0]; HW = 6.0; HT = 10.0
TOWERS = [(xc, yc) for xc in TOWER_XC for yc in TOWER_YC]
CASE_PARAMS = {"c1": (3.0, 6.0), "c2": (1.5, 6.0), "c3": (6.0, 6.0),
               "c4": (3.0, 3.0), "c5": (3.0, 9.0), "c6": (1.5, 3.0),
               "c7": (6.0, 9.0)}

def intake_temp(xyz, T, xc, yc):
    """Mean air temperature in the intake band just outside a tower's walls."""
    x, y, z = xyz.T
    cheb = np.maximum(np.abs(x-xc), np.abs(y-yc))
    band = (cheb >= HW) & (cheb <= HW+5.0) & (z >= 1.0) & (z <= 8.0)
    if band.sum() < 5:
        band = (cheb >= HW) & (cheb <= HW+8.0) & (z >= 0.5) & (z <= 10.0)
    return float(T[band].mean()), int(band.sum())

def analyze(datadir, figdir):
    os.makedirs(figdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(datadir, "c*_field.npz")))
    rows = []
    for fp in files:
        tag = os.path.basename(fp).split("_")[0]
        d = np.load(fp); xyz = d["xyz"]; T = d["T"]
        U, V = float(d["U_inf"]), float(d["V_dis"])
        rcs = []
        for (xc, yc) in TOWERS:
            Ti, n = intake_temp(xyz, T, xc, yc)
            rcs.append((Ti-TAMB)/(TDIS-TAMB)*100.0)
        rcs = np.array(rcs)
        rows.append(dict(case=tag, U=U, V=V, ratio=V/U,
                         rc_mean=float(rcs.mean()), rc_max=float(rcs.max()),
                         rc_per_tower=rcs.tolist()))
    rows.sort(key=lambda r: r["case"])

    # table out
    with open(os.path.join(figdir, "recirculation_table.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(figdir, "recirculation_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case","U_inf(m/s)","V_dis(m/s)","V/U","RC_mean(%)","RC_max(%)"])
        for r in rows:
            w.writerow([r["case"], r["U"], r["V"], f'{r["ratio"]:.2f}',
                        f'{r["rc_mean"]:.2f}', f'{r["rc_max"]:.2f}'])

    # Fig 1: RC vs bend-over ratio V/U
    plt.figure(figsize=(6.2, 4.4))
    rr = sorted(rows, key=lambda r: r["ratio"])
    x = [r["ratio"] for r in rr]
    plt.plot(x, [r["rc_max"] for r in rr], "o-", label="worst tower", color="#c0392b")
    plt.plot(x, [r["rc_mean"] for r in rr], "s--", label="array mean", color="#2c3e50")
    for r in rr:
        plt.annotate(r["case"], (r["ratio"], r["rc_max"]),
                     textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.xlabel("plume bend-over ratio  V_dis / U_inf")
    plt.ylabel("hot-air recirculation  RC  (%)")
    plt.title("Recirculation vs plume/wind ratio")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(figdir, "fig_rc_vs_ratio.png"), dpi=150)
    plt.close()

    # Fig 2: per-tower RC heat-strip per case
    plt.figure(figsize=(7.2, 4.4))
    M = np.array([r["rc_per_tower"] for r in rows])
    im = plt.imshow(M.T, aspect="auto", cmap="inferno", origin="lower")
    plt.colorbar(im, label="RC (%)")
    plt.yticks(range(len(TOWERS)), [f"T{i+1}({int(x)},{int(y)})"
               for i, (x, y) in enumerate(TOWERS)], fontsize=7)
    plt.xticks(range(len(rows)), [r["case"] for r in rows])
    plt.xlabel("case"); plt.title("Per-tower recirculation coefficient")
    plt.tight_layout(); plt.savefig(os.path.join(figdir, "fig_rc_per_tower.png"), dpi=150)
    plt.close()

    return rows

def pinn_figures(evfile, figdir):
    if not os.path.exists(evfile):
        return
    d = np.load(evfile)
    # parity: temperature
    plt.figure(figsize=(5.2, 5.0))
    Tc, Tp = d["T_cfd"], d["T_pred"]
    idx = np.random.RandomState(0).choice(len(Tc), size=min(6000, len(Tc)), replace=False)
    plt.scatter(Tc[idx], Tp[idx], s=3, alpha=0.25, color="#2980b9")
    lo, hi = 296, 315
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.xlabel("CFD temperature (K)"); plt.ylabel("surrogate temperature (K)")
    plt.title("Surrogate vs CFD, unseen centre case c1")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "fig_pinn_parity_T.png"), dpi=150); plt.close()

    # training history
    if "hist" in d and d["hist"].size:
        h = d["hist"]
        plt.figure(figsize=(5.6, 4.0))
        plt.semilogy(h[:, 0], h[:, 1], "-", color="#8e44ad")
        plt.xlabel("epoch"); plt.ylabel("training loss")
        plt.title("PINN training history"); plt.grid(alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, "fig_pinn_loss.png"), dpi=150); plt.close()

if __name__ == "__main__":
    datadir = sys.argv[1] if len(sys.argv) > 1 else "../data"
    figdir = sys.argv[2] if len(sys.argv) > 2 else "../figures"
    rows = analyze(datadir, figdir)
    for r in rows:
        print(f'{r["case"]}: U={r["U"]} V={r["V"]} V/U={r["ratio"]:.2f} '
              f'RC_mean={r["rc_mean"]:.2f}% RC_max={r["rc_max"]:.2f}%')
    pinn_figures(os.path.join(os.path.dirname(figdir), "pinn", "pinn_eval.npz"), figdir)
