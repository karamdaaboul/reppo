"""Per-eval diagnostics for a training run, on one shared time axis.

Parses the driver's log lines rather than the exported meta, so it works on runs that
are still in flight and on runs finished before `alpha_curve` / `kl_curve` were added.

Also answers the lead/lag question directly: around the largest single-eval drop in
return, does eta move first, at the same time, or after? Reported as a cross-correlation
between the first differences of return and of each control variable at lags -3..+3
evals, plus a local table around the worst drop. A NEGATIVE lag means the control
variable moves BEFORE the return change (it leads).
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LINE = re.compile(
    r"step=(?P<step>\d+) ret=(?P<ret>[-\d.]+) len=(?P<len>[\d.]+) sps=(?P<sps>[\d.]+) \| "
    r"ent=(?P<ent>[-\d.naif]+) sigma=(?P<sigma>[-\d.naif]+) "
    r"\[(?P<smin>[-\d.naif]+),(?P<smax>[-\d.naif]+)\] "
    r"temp=(?P<temp>[-\d.naif]+) kl=(?P<kl>[-\d.naif]+) \| "
    r"ess=(?P<ess>[-\d.naif]+) w_max=(?P<wmax>[-\d.naif]+)"
    # qspread and eta were added later; older logs (arm A) lack them
    r"(?: qspread=(?P<qspread>[-\d.naif]+))?"
    r"(?: eta=(?P<eta>[-\d.naif]+))?"
)

SERIES = ["ret", "eta", "temp", "ent", "sigma", "kl", "ess"]
LABEL = {
    "ret": "eval return", "eta": "eta (E-step temp)", "temp": "alpha (entropy dual)",
    "ent": "entropy", "sigma": "pi sigma", "kl": "KL(pi_old||pi)", "ess": "ESS",
}


def parse(path):
    rows = []
    with open(path) as f:
        for line in f:
            m = LINE.search(line)
            if m:
                d = m.groupdict()
                rows.append({k: float(v) if v is not None else np.nan
                             for k, v in d.items()})
    if not rows:
        return None
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def leadlag(ret, x, max_lag=3):
    """Cross-correlation of first differences. Negative lag = x leads return."""
    dr, dx = np.diff(ret), np.diff(x)
    ok = np.isfinite(dr) & np.isfinite(dx)
    dr, dx = dr[ok], dx[ok]
    if dr.size < 5 or dr.std() == 0 or dx.std() == 0:
        return {}
    dr = (dr - dr.mean()) / dr.std()
    dx = (dx - dx.mean()) / dx.std()
    out = {}
    for L in range(-max_lag, max_lag + 1):
        if L < 0:
            a, b = dr[-L:], dx[: len(dx) + L]
        elif L > 0:
            a, b = dr[: len(dr) - L], dx[L:]
        else:
            a, b = dr, dx
        n = min(len(a), len(b))
        out[L] = float(np.dot(a[:n], b[:n]) / n) if n > 2 else np.nan
    return out


def plot_run(d, title, out_png):
    x = d["step"] / 1e6
    fig, axes = plt.subplots(len(SERIES), 1, figsize=(11, 13), sharex=True)
    for ax, k in zip(axes, SERIES):
        y = d.get(k)
        ax.plot(x, y, marker="o", ms=3, lw=1.4, color="#1f77b4")
        ax.set_ylabel(LABEL[k], fontsize=9)
        ax.grid(alpha=0.3)
        if k in ("eta", "temp") and np.nanmin(y) > 0:
            ax.set_yscale("log")
    # mark the worst single-eval drop in return
    dr = np.diff(d["ret"])
    if dr.size:
        j = int(np.argmin(dr)) + 1
        for ax in axes:
            ax.axvline(x[j], color="crimson", ls="--", lw=1.2, alpha=0.8)
        axes[0].annotate(f"worst drop {dr[j-1]:+.0f}", (x[j], d["ret"][j]),
                         color="crimson", fontsize=9,
                         xytext=(6, 10), textcoords="offset points")
    axes[-1].set_xlabel("environment steps (millions)")
    axes[0].set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for path in args.logs:
        d = parse(path)
        name = os.path.basename(path).replace(".log", "")
        if d is None:
            print(f"{name}: no parsable eval lines yet")
            continue
        png = plot_run(d, name, os.path.join(args.outdir, f"diag_{name}.png"))
        dr = np.diff(d["ret"])
        j = int(np.argmin(dr)) + 1
        print(f"\n=== {name}  ({len(d['ret'])} evals) ===")
        print(f"  worst single-eval return drop: {dr[j-1]:+.1f} "
              f"at {d['step'][j]/1e6:.1f}M (eval {j})")
        print(f"  {'eval':>5} {'Mstep':>7} {'return':>8} {'eta':>8} {'alpha':>8} "
              f"{'ent':>7} {'sigma':>7} {'kl':>7} {'ess':>6}")
        for i in range(max(0, j - 3), min(len(d["ret"]), j + 4)):
            mark = " <<<" if i == j else ""
            print(f"  {i:>5} {d['step'][i]/1e6:>7.1f} {d['ret'][i]:>8.1f} "
                  f"{d['eta'][i]:>8.4f} {d['temp'][i]:>8.5f} {d['ent'][i]:>7.2f} "
                  f"{d['sigma'][i]:>7.3f} {d['kl'][i]:>7.4f} {d['ess'][i]:>6.2f}{mark}")
        print("  lead/lag corr of d(x) vs d(return)   "
              "[negative lag = x LEADS the return change]")
        for k in ("eta", "temp", "sigma", "ent", "kl", "ess"):
            ll = leadlag(d["ret"], d[k])
            if ll:
                best = max(ll, key=lambda L: abs(ll[L]))
                cells = "  ".join(f"{L:+d}:{ll[L]:+.2f}" for L in sorted(ll))
                print(f"    {k:<6} {cells}   | strongest at lag {best:+d} "
                      f"({ll[best]:+.2f})")
        print(f"  -> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
