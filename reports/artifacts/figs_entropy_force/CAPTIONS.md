# Stage A figure — draft caption

`fig_entropy_force.{pdf,png}`, from `scripts/analysis/entropy_force_measurement.py`,
job `3795539`, commit `56d48fb`, measured 2026-09-07T11:43:48+02:00.
Source artifact `entropy_force_percoord.csv` (on-arm, per task x cell x seed x coordinate).

> **Actor-entropy force F_H against policy width, per action coordinate, on-arm states.**
> `F := -dL/dlog_sigma`, so `F > 0` widens and `F < 0` contracts;
> `F_H = alpha * [1 - 2*sigma^2*E(sech^2 y)]` with `y = mu + sigma*u`, `u ~ N(0,I)`,
> 256 draws per state-coordinate. Each task uses its own frozen alpha, shown per panel:
> Walker 1.45e-02, G1 2.08e-04, LEAP 7.82e-04. One point per action coordinate per seed,
> reduced as the median over that cell's own on-arm states, which are addressed by the
> bank's per-seed `source` label. Log x-axis. The dashed line marks `sigma = 1/sqrt(2)`,
> which is **sufficient but not necessary** for widening: below it `F_H > 0` is guaranteed
> because `E[sech^2] <= 1`, above it the sign depends on mu and the state distribution.
>
> **This figure shows the on-arm view only, where every cell is positive.** The neutral-bank
> view differs materially for two cells — Walker `WML_noent` and LEAP `WML_noent` both
> measure negative there — and that divergence is the reason the instrument fails its
> validation gate at 3/4. See `reports/entropy_force_measurement.md` for both populations
> side by side; reading this figure alone would give the misleading impression that the
> entropy term widens everywhere.
