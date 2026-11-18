# PathSnap: API + UI deployment guide (two laptops)

```
 Laptop 2 (UI)                         Laptop 1 (API)
 ┌───────────────────────┐   Wi-Fi   ┌───────────────────────────────────┐
 │ browser               │ ───────►  │ FastAPI  :8000                     │
 │ http://localhost:8080 │  trip ID  │  candidates → LightGBM → HMM/Viterbi│
 │ (ui/index.html)       │ ◄───────  │  scores vs hidden .route (server)  │
 └───────────────────────┘ route +   │  logs every request in the terminal│
                           metrics   └───────────────────────────────────┘
```
The UI never sees reference routes. It receives only the GPS points, the predicted routes and aggregate metrics.

---

## Part 1: Laptop 1 (API) — one-time setup

1. **Get the code.** Clone the repo the first time:
   ```bash
   git clone https://github.com/Kabir7code/pathsnap.git
   ```
   ```bash
   cd pathsnap
   ```
   If you already have it, run `git pull` inside the folder instead.

2. **Create a Python environment.** You need Python 3.10 or newer.

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   ```
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

   macOS/Linux:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   ```

   Then, on any system:
   ```bash
   pip install -r requirements.txt
   ```
   On a Mac, if `lightgbm` fails to import, run `brew install libomp`.

3. **Add the dataset.** Unzip `Student_folder.zip` and move its `train/` and `student_test/` folders into `pathsnap/data/`:
   ```
   pathsnap/data/train/00000001/00000001.nodes ...
   pathsnap/data/student_test/00000000/00000000.nodes ...
   ```

4. **Allow port 8000 through the firewall.**
   - Windows (PowerShell **as Administrator**):
     ```powershell
     New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
     ```
     Also click **Allow** if Windows asks about Python the first time the API starts. Tick **Private** networks, and **Public** too if the venue Wi-Fi is marked public.
   - macOS: click **Allow** when asked "Do you want python to accept incoming connections?".

5. **Run the pre-flight check:**
   ```bash
   python scripts/check_setup.py
   ```
   Every line should say `[OK]`. It also prints **this laptop's IP** and the exact URL for laptop 2.

## Part 2: Laptop 1 — start the API

- Windows: double-click **`start_api.bat`**, or run it in a terminal.
- macOS/Linux:
  ```bash
  ./start_api.sh
  ```

Both run the check first, then start the server on `0.0.0.0:8000` using `data/student_test`.

**For the real evaluation**, put the evaluators' 10 trips in a folder, one sub-folder per trip in the raw format with `.route` included, for example `pathsnap/eval_data/`. Never commit this folder (it's git-ignored). Then start the API pointing at it:
- Windows: `start_api.bat eval_data`
- macOS/Linux: `./start_api.sh eval_data`

**Keep this terminal visible during the demo.** Every click on the UI appears here, for example:
```
INFO | POST /predict trip=00000052 : running pipeline
INFO | trip=00000052 points=262 candidates=1641 predicted_route_arcs=95 time=0.16s
INFO | trip=00000052 METRICS top1 cls=0.9656 hmm=0.9962 | route F1 cls=0.9073 hmm=0.9894 | continuity hmm=0.9617
INFO:     192.168.x.x:52344 - "POST /predict HTTP/1.1" 200 OK
```
The brief requires you to show these logs to the evaluator. Take a screenshot of them for the report.

Quick test from laptop 1 itself: open `http://localhost:8000/docs` in a browser. That's FastAPI's built-in page, where you can try each endpoint.

## Part 3: Laptop 2 (UI)

1. Get the code (`git clone` or `git pull`). The UI needs only Python, with no packages: it's a static page.
2. Start it:
   - Windows: `start_ui.bat 192.168.1.23`, using laptop 1's IP. It opens Chrome automatically.
   - macOS/Linux: `./start_ui.sh 192.168.1.23`, then open the printed URL.
   - Or by hand: `cd ui`, then `python -m http.server 8080`, then open `http://localhost:8080/?api=http://<LAPTOP1_IP>:8000`.
3. In the page: the top bar should show a green **API online** pill. Then either:
   - click a trip to draw it and see its metrics, or
   - click **Run all trips**, then open the **All trips** tab for the summary table.
   - Tabs: **Trip result**, **All trips**, **Model** (settings), **API log** (client-side request log).
   - Map layers (top right): toggle the **classifier-only route** to show before vs after the HMM. Choose **No basemap (offline)** if there's no internet.

## Part 4: Demo-day checklist
- [ ] Both laptops are on the **same network**. College Wi-Fi often blocks laptop-to-laptop traffic, so if it fails, put both on a **phone hotspot**.
- [ ] Laptop 1: `python scripts/check_setup.py eval_data` shows all `[OK]`.
- [ ] Laptop 1: start the API with the evaluators' folder, and keep its terminal visible.
- [ ] Laptop 2: open the UI with laptop 1's IP, check the green pill, check all 10 trips are listed.
- [ ] Run one trip as a smoke test, and confirm a `METRICS` line appears on laptop 1.
- [ ] Have `results/models_dashboard.html` open in another tab for model questions.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| UI says **Unreachable** | Wrong IP; API not started with `--host 0.0.0.0` (the start scripts do this); firewall blocking port 8000; laptops on different networks or client isolation (use a hotspot). Test from laptop 2's browser: `http://<IP>:8000/health` should show `{"status":"ok",...}`. |
| `Address already in use` / port 8000 busy | An API is already running. Close the other window, or on Windows run `netstat -ano | findstr :8000` then `taskkill /PID <pid> /F`. |
| `ModuleNotFoundError` | The virtual environment isn't activated, or `pip install -r requirements.txt` hasn't been run. |
| `lightgbm` import error on Mac | `brew install libomp` |
| API starts but shows 0 trips | Wrong folder. It must contain one sub-folder per trip: `eval_data/<id>/<id>.track` and so on. |
| Metrics are blank for a trip | That trip folder has no `.route` file, so the server can't score it. Predictions still show. |
| Map background is blank | No internet for map tiles. Choose **No basemap (offline)**; routes and points still draw. |
| A trip takes a few seconds | Normal for long trips (up to about 6 s). The spinner shows while the server computes. |

## API reference

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `{"status": "ok", "model": "lightgbm", "eval_dir": ...}` |
| GET | `/info` | Final model settings, HMM parameters, candidate settings |
| GET | `/trips` | List of `{trip_id, gps_points}` in the evaluation folder |
| POST | `/predict` with body `{"trip_id": "00000052"}` | GPS points, predicted route (classifier + HMM), classifier-only route, predicted arc IDs, server-side metrics, timing, decoder stats. An unknown ID returns 404. |
| GET | `/docs` | Interactive API documentation (auto-generated by FastAPI) |
