import sys, os, traceback, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import test_planted as T
names = [n for n in dir(T) if n.startswith("test_")]
fails = 0
for n in sorted(names):
    t0 = time.time()
    try:
        getattr(T, n)()
        print("PASS  %-56s %5.1fs" % (n, time.time() - t0), flush=True)
    except Exception:
        fails += 1
        print("FAIL  %-56s" % n, flush=True)
        traceback.print_exc()
print("\n%d/%d passed" % (len(names) - fails, len(names)))
sys.exit(1 if fails else 0)
