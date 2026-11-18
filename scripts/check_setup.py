"""Pre-flight check for the demo. Run from the repo root:  python scripts/check_setup.py [eval_dir]

Checks Python packages, model artifacts, the evaluation-trip folder and the port,
and prints this laptop's IP address for the UI laptop.
"""
import importlib
import os
import socket
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
ok = True


def check(cond, msg, fix=""):
    global ok
    print(("  [OK]   " if cond else "  [FAIL] ") + msg + ("" if cond or not fix else f"\n         fix: {fix}"))
    ok &= bool(cond)


print("1. Python and packages")
check(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]} (need 3.10+)", "install Python 3.10 or newer")
for mod in ["numpy", "pandas", "scipy", "sklearn", "lightgbm", "joblib", "fastapi", "uvicorn", "pydantic", "folium"]:
    try:
        importlib.import_module(mod); check(True, mod)
    except Exception as e:
        check(False, f"{mod}: {e}", "pip install -r requirements.txt")

print("2. Model artifacts")
for f in ["models/final.joblib", "models/final_model.json", "models/hmm_params.json"]:
    check(os.path.exists(f), f, "git pull (these files are in the repo)")
try:
    import joblib
    m = joblib.load("models/final.joblib"); check(True, f"final model loads ({type(m).__name__})")
except Exception as e:
    check(False, f"final model failed to load: {e}", "pip install -r requirements.txt (same lightgbm/sklearn)")

print("3. Evaluation trips")
ev = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PATHSNAP_EVAL_DIR", "data/student_test")
check(os.path.isdir(ev), f"folder {ev}", "put the trip folders there, or pass the right path")
if os.path.isdir(ev):
    trips = sorted(d for d in os.listdir(ev) if os.path.isdir(os.path.join(ev, d)))
    check(len(trips) > 0, f"{len(trips)} trip folders found")
    for t in trips:
        have = [e for e in ("nodes", "arcs", "track", "route") if os.path.exists(os.path.join(ev, t, f"{t}.{e}"))]
        miss = {"nodes", "arcs", "track"} - set(have)
        check(not miss, f"{t}: {', '.join('.' + h for h in have)}" + ("" if "route" in have else "  (no .route: metrics will be blank)"),
              f"missing {miss}")

print("4. Network")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
free = s.connect_ex(("127.0.0.1", 8000)) != 0; s.close()
check(free, "port 8000 is free" if free else "port 8000 already in use (an API may already be running)",
      "close the other API window, or stop the process using port 8000")
try:
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); u.connect(("10.255.255.255", 1)); ip = u.getsockname()[0]; u.close()
except Exception:
    ip = "127.0.0.1"
print(f"\n  This laptop's IP: {ip}")
print(f"  UI laptop should open:  http://localhost:8080/?api=http://{ip}:8000")
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED: fix the items marked [FAIL] above")
sys.exit(0 if ok else 1)
