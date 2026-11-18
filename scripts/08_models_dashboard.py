"""One self-contained HTML page showing every model at once -> results/00_overview/models_dashboard.html

Built only from recorded results (results/*.csv|json|png) and the saved model
files; open it in any browser (works offline, images are embedded).
"""
import base64
import html
import json
import os

import pandas as pd

R = "results"
NAMES = {"logreg": "Logistic Regression", "gnb": "Gaussian Naive Bayes", "knn": "K-Nearest Neighbours",
         "dtree": "Decision Tree", "rf": "Random Forest", "extratrees": "Extra Trees",
         "histgb": "HistGradientBoosting", "lightgbm": "LightGBM", "mlp": "MLP (neural network)"}


def img(path, caption):
    if not os.path.exists(path):
        return ""
    b = base64.b64encode(open(path, "rb").read()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


cmp_ = pd.read_csv(f"{R}/03_concurrent_training/model_comparison.csv").sort_values("test_top1", ascending=False)
tvu = pd.read_csv(f"{R}/04_bayesian_optimisation/tuned_vs_untuned.csv")
hmm = pd.read_csv(f"{R}/05_hmm_viterbi/hmm_summary.csv").set_index("decoder")
conc = json.load(open(f"{R}/03_concurrent_training/run_summary.json"))
final = json.load(open("models/final_model.json"))
S6, S7 = f"{R}/08_final_model_selection", f"{R}/09_final_tuning_lightgbm_with_hmm"
sel = pd.read_csv(f"{S6}/summary_three_models.csv").set_index(["split", "model"])
pst = pd.concat([pd.read_csv(f"{S6}/paired_stats_lgbm_vs_rf.csv").query("decoder == 'own'").assign(comparison="Random Forest − LightGBM").rename(columns={"mean_RF_minus_LGBM": "mean_diff"}),
                 pd.read_csv(f"{S6}/paired_stats_histgb_vs_others.csv").query("comparison == 'HistGB - LightGBM (tuned)'").assign(comparison="HistGB − LightGBM")])
cost = json.load(open(f"{S6}/decision.json"))["probability_quality_and_cost"]
tv7 = pd.read_csv(f"{S7}/tuned_vs_untuned_lightgbm.csv").set_index("variant")
d7 = json.load(open(f"{S7}/decision_lightgbm.json"))
cards = open(f"{R}/00_overview/model_cards.md").read().split("\n## ")[1:]
card_by = {c.split("\n", 1)[0].strip(): c.split("\n", 1)[1] for c in cards}

best = cmp_.iloc[0]
rows = ""
for i, r in enumerate(cmp_.itertuples()):
    cls = ' class="best"' if i == 0 else ""
    rows += (f"<tr{cls}><td>{i + 1}</td><td><b>{NAMES[r.model]}</b><br><small>{html.escape(r.family)}</small></td>"
             f"<td>{r.cv_macro_f1:.4f} <small>± {r.cv_macro_f1_std:.4f}</small></td><td>{r.test_top1:.4f}</td>"
             f"<td>{r.test_macro_f1:.4f}</td><td>{r.test_precision:.4f}</td><td>{r.test_recall:.4f}</td>"
             f"<td>{r.test_log_loss:.4f}</td><td>{r.test_continuity:.4f}</td>"
             f"<td>{r.infer_ms_per_point:.4f}</td><td>{r.duration_s:.0f}</td></tr>")

card_html = ""
for key in list(cmp_.model) + ["final"]:
    body = card_by.get(key, "")
    items = "".join(f"<li>{html.escape(l[2:])}</li>" for l in body.splitlines() if l.startswith("- "))
    title = NAMES.get(key, "Final model: tuned LightGBM (deployed)")
    card_html += f'<div class="card{" final" if key == "final" else ""}"><h3>{title}</h3><ul>{items}</ul></div>'

t_un, t_tu = tvu.set_index("variant").loc["untuned"], tvu.set_index("variant").loc["tuned"]
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>PathSnap Models</title>
<style>
:root {{ --ink:#1b2130; --muted:#667085; --line:#e2e5eb; --bg:#f6f7f9; --accent:#d62728; --good:#e7f6ec; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink); background:var(--bg); }}
header {{ background:#fff; border-bottom:1px solid var(--line); padding:18px 24px; position:sticky; top:0; z-index:5; }}
header h1 {{ margin:0; font-size:20px; }} header h1 span {{ color:var(--accent); }}
nav a {{ margin-right:14px; color:var(--muted); text-decoration:none; font-size:13px; }} nav a:hover {{ color:var(--ink); }}
main {{ max-width:1200px; margin:0 auto; padding:16px; }}
section {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:18px 20px; margin:16px 0; }}
h2 {{ font-size:17px; margin:0 0 12px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; }}
.kpi {{ border:1px solid var(--line); border-radius:9px; padding:10px 12px; }}
.kpi .l {{ font-size:12px; color:var(--muted); }} .kpi .v {{ font-size:24px; font-weight:700; }}
.tablewrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; font-size:13px; }}
th, td {{ padding:7px 8px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
th {{ font-size:12px; color:var(--muted); cursor:pointer; user-select:none; }}
th:nth-child(2), td:nth-child(2) {{ text-align:left; }}
tr.best {{ background:var(--good); }}
small {{ color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); gap:14px; }}
figure {{ margin:0; border:1px solid var(--line); border-radius:9px; padding:8px; background:#fff; }}
figure img {{ width:100%; height:auto; display:block; }}
figcaption {{ font-size:12px; color:var(--muted); padding-top:4px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:12px; }}
.card {{ border:1px solid var(--line); border-radius:9px; padding:12px 14px; }}
.card.final {{ border-color:var(--accent); }}
.card h3 {{ margin:0 0 6px; font-size:15px; }} .card ul {{ margin:0; padding-left:18px; font-size:12.5px; }}
.note {{ color:var(--muted); font-size:12.5px; }}
@media (max-width:600px) {{ .grid {{ grid-template-columns:1fr; }} header {{ position:static; }} }}
</style></head><body>
<header><h1>Path<span>Snap</span>: all models</h1>
<nav><a href="#summary">Summary</a><a href="#table">Comparison</a><a href="#plots">Plots</a><a href="#tuning">Bayesian tuning</a><a href="#selection">Final selection</a><a href="#finaltuning">Final tuning</a>
<a href="#hmm">HMM</a><a href="#concurrent">Concurrent training</a><a href="#cards">Model cards</a></nav></header>
<main>
<section id="summary"><h2>Summary</h2><div class="kpis">
<div class="kpi"><div class="l">Models compared</div><div class="v">{len(cmp_)}</div><small>7 algorithm families</small></div>
<div class="kpi"><div class="l">Best model</div><div class="v">{NAMES[best.model]}</div><small>by every metric</small></div>
<div class="kpi"><div class="l">Best test Top-1 (classifier)</div><div class="v">{best.test_top1:.4f}</div><small>nearest-road baseline: 0.5418</small></div>
<div class="kpi"><div class="l">With HMM (mean per trip)</div><div class="v">{hmm.loc['tuned','hmm_top1']:.4f}</div><small>classifier alone: {hmm.loc['tuned','cls_top1']:.4f}</small></div>
<div class="kpi"><div class="l">Route F1 with HMM</div><div class="v">{hmm.loc['tuned','hmm_route_f1']:.4f}</div><small>classifier alone: {hmm.loc['tuned','cls_route_f1']:.4f}</small></div>
</div></section>

<section id="table"><h2>Model comparison</h2>
<p class="note">CV = 5-fold GroupKFold over training trips. Test = 10 student-test trips. Click a column header to sort.</p>
<div class="tablewrap"><table id="cmp"><thead><tr><th>#</th><th>Model</th><th>CV macro F1</th><th>Test Top-1</th><th>Test macro F1</th>
<th>Precision</th><th>Recall</th><th>Log loss</th><th>Continuity</th><th>Infer ms/pt</th><th>Train time s</th></tr></thead>
<tbody>{rows}</tbody></table></div></section>

<section id="plots"><h2>Comparison plots</h2><div class="grid">
{img(f"{R}/02_model_training/plots/cv_macro_f1.png", "Cross-validation macro F1 per model (dots = folds)")}
{img(f"{R}/02_model_training/plots/student_test_top1_f1.png", "Student-test Top-1 accuracy and macro F1")}
{img(f"{R}/02_model_training/plots/log_loss.png", "Log loss, CV and student test (lower is better)")}
{img(f"{R}/02_model_training/plots/speed_vs_accuracy.png", "Inference speed vs accuracy")}
{img(f"{R}/02_model_training/plots/confusion_matrices.png", "Confusion matrices on student test (threshold 0.5)")}
{img(f"{R}/02_model_training/plots/mlp_training_curves.png", "MLP training/validation curves (only model with epoch history)")}
</div></section>

<section id="tuning"><h2>Bayesian hyperparameter optimisation (LightGBM)</h2>
<p class="note">Optuna TPE, {final['trials']} trials, objective = 3-fold GroupKFold CV macro F1, search time {final['search_seconds']:.0f}s.</p>
<div class="tablewrap"><table><thead><tr><th></th><th>Variant</th><th>CV macro F1</th><th>Test Top-1</th><th>Test macro F1</th><th>Test log loss</th><th>Continuity</th><th>Infer ms/pt</th></tr></thead><tbody>
<tr><td></td><td>Untuned</td><td>{t_un.cv_macro_f1:.4f}</td><td>{t_un.test_top1:.4f}</td><td>{t_un.test_macro_f1:.4f}</td><td>{t_un.test_log_loss:.4f}</td><td>{t_un.test_continuity:.4f}</td><td>{t_un.infer_ms_per_point:.4f}</td></tr>
<tr><td></td><td>Tuned</td><td>{t_tu.cv_macro_f1:.4f}</td><td>{t_tu.test_top1:.4f}</td><td>{t_tu.test_macro_f1:.4f}</td><td>{t_tu.test_log_loss:.4f}</td><td>{t_tu.test_continuity:.4f}</td><td>{t_tu.infer_ms_per_point:.4f}</td></tr>
</tbody></table></div>
<p class="note">Best parameters: {html.escape(", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in final['params'].items()))}</p>
<div class="grid">{img(f"{R}/04_bayesian_optimisation/optuna_history_lightgbm.png", "Optimisation history")}{img(f"{R}/04_bayesian_optimisation/tuned_vs_untuned.png", "Tuned vs untuned")}</div></section>

<section id="hmm"><h2>Classifier vs classifier + HMM/Viterbi</h2>
<div class="tablewrap"><table><thead><tr><th></th><th>Metric (mean of 10 test trips)</th><th>Classifier only</th><th>+ HMM (default)</th><th>+ HMM (tuned)</th></tr></thead><tbody>
{"".join(f"<tr><td></td><td>{lab}</td><td>{hmm.loc['tuned','cls_'+k]:.4f}</td><td>{hmm.loc['default','hmm_'+k]:.4f}</td><td><b>{hmm.loc['tuned','hmm_'+k]:.4f}</b></td></tr>" for k, lab in [("top1","Point Top-1"),("route_precision","Route precision"),("route_recall","Route recall"),("route_f1","Route F1"),("continuity","Continuity")])}
</tbody></table></div>
<div class="grid">{img(f"{R}/05_hmm_viterbi/hmm_comparison.png", "Per trip: classifier only vs classifier + HMM")}</div></section>

<section id="selection"><h2>Final model selection: LightGBM vs Random Forest vs HistGB (88 development trips)</h2>
<p class="note">Complete systems (classifier + HMM) on 78 out-of-fold training trips + 10 student-test trips. Each model has its own tuned HMM decoder. Pre-declared rule: Top-1 after the HMM, then route F1 after the HMM (95% paired bootstrap CI must exclude 0), then log loss, speed and size. The evaluator-test trips were not used.</p>
<div class="tablewrap"><table><thead><tr><th></th><th>Model</th><th>Top-1 classifier</th><th>Top-1 + HMM</th><th>Route F1 classifier</th><th>Route F1 + HMM</th><th>Log loss (OOF)</th><th>ms/point</th><th>Model MB</th></tr></thead><tbody>
{"".join(f"<tr{' class=best' if m.startswith('LightGBM') else ''}><td></td><td><b>{m}</b></td><td>{sel.loc[('all_88', m), 'cls_top1']:.4f}</td><td>{sel.loc[('all_88', m), 'hmm_top1']:.4f}</td><td>{sel.loc[('all_88', m), 'cls_route_f1']:.4f}</td><td>{sel.loc[('all_88', m), 'hmm_route_f1']:.4f}</td><td>{cost[m]['oof_log_loss']:.4f}</td><td>{cost[m]['test_infer_ms_per_point']:.4f}</td><td>{cost[m]['model_file_mb']:.2f}</td></tr>" for m in ["LightGBM (tuned)", "HistGB", "Random Forest"])}
</tbody></table></div>
<div class="tablewrap" style="margin-top:10px"><table><thead><tr><th></th><th>Comparison</th><th>Metric</th><th>Mean difference</th><th>95% CI</th><th>Significant?</th></tr></thead><tbody>
{"".join(f"<tr><td></td><td>{r.comparison}</td><td>{r.metric}</td><td>{r.mean_diff:+.4f}</td><td>[{r.ci95_low:+.4f}, {r.ci95_high:+.4f}]</td><td>{'yes' if not (r.ci95_low <= 0 <= r.ci95_high) else 'no (tie)'}</td></tr>" for r in pst.itertuples())}
</tbody></table></div>
<p class="note"><b>Decision: LightGBM.</b> After the HMM all three are tied on Top-1. LightGBM is significantly better than Random Forest on route F1 after the HMM, significantly better than both before the HMM, and has the best-calibrated probabilities.</p>
<div class="grid">{img(f"{S6}/three_models_before_after_hmm.png", "Top-1 before vs after HMM, three candidates")}{img(f"{S6}/lgbm_vs_rf_difference_ci.png", "Random Forest − LightGBM: paired differences with 95% CI")}{img(f"{S6}/lgbm_vs_rf_per_trip_scatter.png", "Per-trip comparison (above the line = RF better)")}</div></section>

<section id="finaltuning"><h2>Final tuning: LightGBM with the HMM in the loop</h2>
<p class="note">Optuna TPE, {d7['search']['trials']} trials. Objective: {html.escape(d7['search']['objective'])}. The new settings replace the deployed ones only if Top-1 after the HMM improves significantly over the 88 development trips.</p>
<div class="tablewrap"><table><thead><tr><th></th><th>Variant</th><th>CV macro F1</th><th>Test Top-1 (pooled)</th><th>Test log loss</th><th>Test Top-1 + HMM</th><th>88 trips Top-1 + HMM</th><th>88 trips route F1 + HMM</th></tr></thead><tbody>
{"".join(f"<tr><td></td><td>{v}</td><td>{r.cv_macro_f1:.4f}</td><td>{r.test_top1:.4f}</td><td>{r.test_log_loss:.4f}</td><td>{r.test_hmm_top1_trip_mean:.4f}</td><td>{r.dev88_hmm_top1:.4f}</td><td>{r.dev88_hmm_route_f1:.4f}</td></tr>" for v, r in tv7.iterrows())}
</tbody></table></div>
<p class="note">Re-tuned − current, Top-1 + HMM over 88 trips: {d7['tuned_minus_untuned_dev88_hmm_top1']:+.4f}, 95% CI [{d7['ci95'][0]:+.4f}, {d7['ci95'][1]:+.4f}]. <b>Deployed: {html.escape(d7['deployed_variant'])}</b>.</p>
<div class="grid">{img(f"{S7}/optuna_history_lightgbm.png", "Optimisation history (HMM objective)")}{img(f"{S7}/param_importance_lightgbm.png", "Hyperparameter importance")}{img(f"{S7}/tuned_vs_untuned_lightgbm.png", "Current vs re-tuned")}</div></section>

<section id="concurrent"><h2>Concurrent training</h2>
<p class="note">{conc['models']} models trained in parallel on {conc['workers']} worker processes: wall clock {conc['wall_clock_s']:.0f}s vs {conc['sum_of_model_times_s']:.0f}s if run one after another ({conc['speedup']:.1f}× speed-up). Results identical to the sequential run.</p>
<div class="grid">{img(f"{R}/03_concurrent_training/timeline.png", "Which model ran on which worker, and when")}</div></section>

<section id="cards"><h2>Model cards: what each saved model contains</h2><div class="cards">{card_html}</div></section>
</main>
<script>
document.querySelectorAll("#cmp th").forEach((th, i) => th.onclick = () => {{
  const tb = document.querySelector("#cmp tbody"), rows = [...tb.rows];
  const asc = th.dataset.asc = th.dataset.asc === "1" ? "0" : "1";
  const val = (r) => {{ const t = r.cells[i].innerText; const n = parseFloat(t); return isNaN(n) ? t : n; }};
  rows.sort((a, b) => (val(a) > val(b) ? 1 : -1) * (asc === "1" ? 1 : -1)).forEach((r) => tb.appendChild(r));
}});
</script>
</body></html>"""
open(f"{R}/00_overview/models_dashboard.html", "w").write(page)
print("wrote", f"{R}/00_overview/models_dashboard.html", round(len(page) / 1e6, 2), "MB")
