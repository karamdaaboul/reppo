"""Driver: Probe 1 over the 10 exported k=16 finals x 2 reference laws."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.probe1_restricted_z as P

OUT = "scripts/probe1_out"
os.makedirs(OUT, exist_ok=True)
cks = [f"exports/WalkerRun_{arm}_pad16_s{s}_final"
       for arm in ("pathwise_fa", "weighted_mle") for s in range(5)]
t0 = time.time()
for ck in cks:
    for law in ("ckpt", "std"):
        out = f"{OUT}/{os.path.basename(ck)}__{law}.npz"
        if os.path.exists(out):
            print(f"skip {out}", flush=True); continue
        print(f"=== {ck} law={law}  [{time.time()-t0:.0f}s] ===", flush=True)
        P.run(ck, law, out)
print(f"ALL DONE {time.time()-t0:.0f}s", flush=True)
