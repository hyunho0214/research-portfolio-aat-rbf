"""Create interactive HTML graphs from a Panama simulation output folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output") / "panama_my_sigmas")
    parser.add_argument("--days", type=float, default=15.0, help="Initial segment length for the Figure 4i-style view.")
    parser.add_argument("--html-name", default="panama_interactive.html")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    predictions_dir = output_dir / "predictions"
    metrics_path = output_dir / "panama_rbf_metrics.csv"
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions folder not found: {predictions_dir}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_path}")

    prediction_files = sorted(predictions_dir.glob("panama_predictions_N*.csv"))
    if not prediction_files:
        raise FileNotFoundError(f"No prediction CSV files found in {predictions_dir}")

    predictions = {}
    true_series = None
    for path in prediction_files:
        n_value = int(path.stem.split("_N")[-1])
        df = pd.read_csv(path)
        required = {"day", "true_demand", "predicted_demand"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if true_series is None:
            true_series = {
                "x": df["day"].astype(float).round(8).tolist(),
                "y": df["true_demand"].astype(float).round(8).tolist(),
            }
        predictions[str(n_value)] = {
            "x": df["day"].astype(float).round(8).tolist(),
            "y": df["predicted_demand"].astype(float).round(8).tolist(),
        }

    metrics = pd.read_csv(metrics_path)
    metric_rows = metrics.to_dict(orient="records")
    metadata_path = output_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    html = build_html(
        true_series=true_series,
        predictions=predictions,
        metrics=metric_rows,
        metadata=metadata,
        initial_days=args.days,
        source_dir=str(output_dir),
    )
    interactive_dir = output_dir / "interactive"
    interactive_dir.mkdir(exist_ok=True)
    html_path = interactive_dir / args.html_name
    html_path.write_text(html, encoding="utf-8")
    print(f"Wrote {html_path}")
    print(f"Open with: explorer {html_path}")


def build_html(
    *,
    true_series: dict,
    predictions: dict[str, dict],
    metrics: list[dict],
    metadata: dict,
    initial_days: float,
    source_dir: str,
) -> str:
    payload = {
        "trueSeries": true_series,
        "predictions": predictions,
        "metrics": metrics,
        "metadata": metadata,
        "initialDays": initial_days,
        "sourceDir": source_dir,
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Interactive Panama RBF Results</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: #1f2933;
      background: #f6f7f9;
    }}
    header {{
      padding: 18px 24px;
      background: #ffffff;
      border-bottom: 1px solid #d8dde6;
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 20px;
      font-weight: 700;
    }}
    .meta {{
      font-size: 13px;
      color: #52606d;
      line-height: 1.5;
    }}
    main {{
      padding: 18px 24px 40px;
      max-width: 1500px;
      margin: 0 auto;
    }}
    section {{
      background: #ffffff;
      border: 1px solid #d8dde6;
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 18px;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      font-size: 13px;
      color: #3e4c59;
      margin-bottom: 10px;
    }}
    button {{
      border: 1px solid #b8c2cc;
      background: #ffffff;
      border-radius: 6px;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 13px;
    }}
    button:hover {{
      background: #f0f3f7;
    }}
    label {{
      display: inline-flex;
      gap: 4px;
      align-items: center;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(320px, 1fr));
      gap: 14px;
    }}
    .chart-wrap {{
      border: 1px solid #edf0f4;
      border-radius: 6px;
      padding: 10px;
      background: #ffffff;
    }}
    canvas {{
      display: block;
      width: 100%;
      height: 420px;
      cursor: grab;
    }}
    canvas.dragging {{
      cursor: grabbing;
    }}
    .hint {{
      font-size: 12px;
      color: #7b8794;
      margin-top: 6px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid #d8dde6;
      padding: 6px 8px;
      text-align: right;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      background: #f0f3f7;
    }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
      canvas {{ height: 340px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Interactive Panama RBF Results</h1>
    <div class="meta" id="meta"></div>
  </header>
  <main>
    <section>
      <h2>Full Test Overlay</h2>
      <div class="toolbar" id="fullControls"></div>
      <canvas id="fullChart"></canvas>
      <div class="hint">Mouse wheel: zoom, drag: pan, double-click: reset. Checkboxes toggle series.</div>
    </section>

    <section>
      <h2>15-Day Segments</h2>
      <div class="grid" id="segmentGrid"></div>
      <div class="hint">Each panel starts with the first {initial_days:g} days. Use mouse wheel/drag/double-click inside each panel.</div>
    </section>

    <section>
      <h2>Metrics</h2>
      <div class="grid">
        <div class="chart-wrap">
          <canvas id="mseChart"></canvas>
        </div>
        <div class="chart-wrap">
          <canvas id="r2Chart"></canvas>
        </div>
      </div>
    </section>

    <section>
      <h2>Metric Table</h2>
      <div id="metricTable"></div>
    </section>
  </main>

<script id="payload" type="application/json">{payload_json}</script>
<script>
const DATA = JSON.parse(document.getElementById("payload").textContent);
const COLORS = ["#f2c94c", "#f2994a", "#eb5757", "#d90429", "#2f80ed", "#27ae60", "#9b51e0"];

function niceNumber(value) {{
  if (!Number.isFinite(value)) return "";
  const abs = Math.abs(value);
  if (abs >= 1000 || abs < 0.001 && abs !== 0) return value.toExponential(3);
  return value.toFixed(abs >= 10 ? 2 : 4).replace(/0+$/, "").replace(/\\.$/, "");
}}

class LineChart {{
  constructor(canvas, series, options) {{
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.series = series;
    this.options = options || {{}};
    this.visible = new Set(series.map(s => s.name));
    this.dragging = false;
    this.lastX = 0;
    this.setInitialView();
    this.bind();
    this.resize();
  }}
  setInitialView() {{
    const xs = [];
    const ys = [];
    this.series.forEach(s => {{
      s.x.forEach((v, i) => {{
        if (Number.isFinite(v) && Number.isFinite(s.y[i])) {{
          xs.push(v); ys.push(s.y[i]);
        }}
      }});
    }});
    this.fullX = [Math.min(...xs), Math.max(...xs)];
    this.fullY = [Math.min(...ys), Math.max(...ys)];
    if (this.options.initialXMax !== undefined) {{
      this.xRange = [this.fullX[0], Math.min(this.options.initialXMax, this.fullX[1])];
    }} else {{
      this.xRange = [...this.fullX];
    }}
    this.yRange = [...this.fullY];
    this.padY();
  }}
  padY() {{
    const span = this.yRange[1] - this.yRange[0] || 1;
    this.yRange = [this.yRange[0] - span * 0.08, this.yRange[1] + span * 0.08];
  }}
  bind() {{
    window.addEventListener("resize", () => this.resize());
    this.canvas.addEventListener("wheel", e => {{
      e.preventDefault();
      const rect = this.canvas.getBoundingClientRect();
      const xNorm = (e.clientX - rect.left - this.margin.left) / this.plotW();
      if (xNorm < 0 || xNorm > 1) return;
      const xAtMouse = this.xRange[0] + xNorm * (this.xRange[1] - this.xRange[0]);
      const factor = e.deltaY > 0 ? 1.18 : 0.85;
      const newSpan = (this.xRange[1] - this.xRange[0]) * factor;
      this.xRange = [xAtMouse - xNorm * newSpan, xAtMouse + (1 - xNorm) * newSpan];
      this.clampX();
      this.draw();
    }});
    this.canvas.addEventListener("mousedown", e => {{
      this.dragging = true;
      this.lastX = e.clientX;
      this.canvas.classList.add("dragging");
    }});
    window.addEventListener("mouseup", () => {{
      this.dragging = false;
      this.canvas.classList.remove("dragging");
    }});
    window.addEventListener("mousemove", e => {{
      if (!this.dragging) return;
      const dx = e.clientX - this.lastX;
      this.lastX = e.clientX;
      const span = this.xRange[1] - this.xRange[0];
      const shift = -dx / this.plotW() * span;
      this.xRange = [this.xRange[0] + shift, this.xRange[1] + shift];
      this.clampX();
      this.draw();
    }});
    this.canvas.addEventListener("dblclick", () => this.reset());
  }}
  resize() {{
    const ratio = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.max(320, Math.floor(rect.width * ratio));
    this.canvas.height = Math.max(260, Math.floor(rect.height * ratio));
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.width = rect.width;
    this.height = rect.height;
    this.margin = {{left: 62, right: 18, top: 28, bottom: 48}};
    this.draw();
  }}
  plotW() {{ return Math.max(1, this.width - this.margin.left - this.margin.right); }}
  plotH() {{ return Math.max(1, this.height - this.margin.top - this.margin.bottom); }}
  clampX() {{
    const fullSpan = this.fullX[1] - this.fullX[0];
    const span = Math.min(this.xRange[1] - this.xRange[0], fullSpan);
    if (this.xRange[0] < this.fullX[0]) this.xRange = [this.fullX[0], this.fullX[0] + span];
    if (this.xRange[1] > this.fullX[1]) this.xRange = [this.fullX[1] - span, this.fullX[1]];
  }}
  reset() {{
    this.setInitialView();
    this.draw();
  }}
  dataToX(x) {{
    return this.margin.left + (x - this.xRange[0]) / (this.xRange[1] - this.xRange[0]) * this.plotW();
  }}
  dataToY(y) {{
    return this.margin.top + (1 - (y - this.yRange[0]) / (this.yRange[1] - this.yRange[0])) * this.plotH();
  }}
  drawAxes() {{
    const c = this.ctx;
    c.clearRect(0, 0, this.width, this.height);
    c.fillStyle = "#ffffff";
    c.fillRect(0, 0, this.width, this.height);
    c.strokeStyle = "#d8dde6";
    c.lineWidth = 1;
    c.strokeRect(this.margin.left, this.margin.top, this.plotW(), this.plotH());
    c.font = "12px Arial";
    c.fillStyle = "#52606d";
    c.textAlign = "center";
    c.textBaseline = "top";
    for (let i = 0; i <= 5; i++) {{
      const x = this.xRange[0] + i / 5 * (this.xRange[1] - this.xRange[0]);
      const px = this.dataToX(x);
      c.strokeStyle = "#edf0f4";
      c.beginPath(); c.moveTo(px, this.margin.top); c.lineTo(px, this.margin.top + this.plotH()); c.stroke();
      c.fillText(niceNumber(x), px, this.margin.top + this.plotH() + 8);
    }}
    c.textAlign = "right";
    c.textBaseline = "middle";
    for (let i = 0; i <= 5; i++) {{
      const y = this.yRange[0] + i / 5 * (this.yRange[1] - this.yRange[0]);
      const py = this.dataToY(y);
      c.strokeStyle = "#edf0f4";
      c.beginPath(); c.moveTo(this.margin.left, py); c.lineTo(this.margin.left + this.plotW(), py); c.stroke();
      c.fillStyle = "#52606d";
      c.fillText(niceNumber(y), this.margin.left - 8, py);
    }}
    c.save();
    c.fillStyle = "#1f2933";
    c.font = "13px Arial";
    c.textAlign = "center";
    c.fillText(this.options.xLabel || "Days", this.margin.left + this.plotW() / 2, this.height - 24);
    c.translate(18, this.margin.top + this.plotH() / 2);
    c.rotate(-Math.PI / 2);
    c.fillText(this.options.yLabel || "Electric Demand (MW)", 0, 0);
    c.restore();
    if (this.options.title) {{
      c.fillStyle = "#1f2933";
      c.font = "bold 14px Arial";
      c.textAlign = "center";
      c.textBaseline = "top";
      c.fillText(this.options.title, this.margin.left + this.plotW() / 2, 6);
    }}
  }}
  draw() {{
    this.drawAxes();
    const c = this.ctx;
    this.series.forEach(s => {{
      if (!this.visible.has(s.name)) return;
      c.beginPath();
      let started = false;
      for (let i = 0; i < s.x.length; i++) {{
        const x = s.x[i], y = s.y[i];
        if (x < this.xRange[0] || x > this.xRange[1] || !Number.isFinite(y)) continue;
        const px = this.dataToX(x), py = this.dataToY(y);
        if (!started) {{ c.moveTo(px, py); started = true; }}
        else c.lineTo(px, py);
      }}
      c.strokeStyle = s.color;
      c.lineWidth = s.width || 1.4;
      c.setLineDash(s.dash ? [6, 4] : []);
      c.stroke();
      c.setLineDash([]);
    }});
  }}
}}

function makeControls(container, chart) {{
  chart.series.forEach(s => {{
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = true;
    input.addEventListener("change", () => {{
      if (input.checked) chart.visible.add(s.name);
      else chart.visible.delete(s.name);
      chart.draw();
    }});
    const swatch = document.createElement("span");
    swatch.textContent = s.name;
    swatch.style.color = s.color;
    swatch.style.fontWeight = "700";
    label.appendChild(input);
    label.appendChild(swatch);
    container.appendChild(label);
  }});
  const reset = document.createElement("button");
  reset.textContent = "Reset";
  reset.addEventListener("click", () => chart.reset());
  container.appendChild(reset);
}}

function init() {{
  const meta = DATA.metadata || {{}};
  document.getElementById("meta").innerHTML =
    `Source: ${{DATA.sourceDir}}<br>` +
    `Sigma: ${{meta.sigma_min ?? ""}} to ${{meta.sigma_max ?? ""}}, scale=${{meta.sigma_scale ?? ""}}, selection=${{meta.sigma_selection ?? ""}}`;

  const nValues = Object.keys(DATA.predictions).sort((a,b) => Number(a) - Number(b));
  const fullSeries = [
    {{name: "True", x: DATA.trueSeries.x, y: DATA.trueSeries.y, color: "#333333", width: 1.0}},
    ...nValues.map((n, i) => ({{name: `N=${{n}}`, x: DATA.predictions[n].x, y: DATA.predictions[n].y, color: COLORS[i % COLORS.length], dash: true}}))
  ];
  const full = new LineChart(document.getElementById("fullChart"), fullSeries, {{title: "Full Test Overlay"}});
  makeControls(document.getElementById("fullControls"), full);

  const grid = document.getElementById("segmentGrid");
  nValues.forEach((n, i) => {{
    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    const controls = document.createElement("div");
    controls.className = "toolbar";
    const canvas = document.createElement("canvas");
    wrap.appendChild(controls);
    wrap.appendChild(canvas);
    grid.appendChild(wrap);
    const chart = new LineChart(canvas, [
      {{name: "True", x: DATA.trueSeries.x, y: DATA.trueSeries.y, color: "#333333", width: 1.0}},
      {{name: `N=${{n}}`, x: DATA.predictions[n].x, y: DATA.predictions[n].y, color: COLORS[i % COLORS.length], dash: true}}
    ], {{title: `15-Day Segment, N=${{n}}`, initialXMax: DATA.initialDays}});
    makeControls(controls, chart);
  }});

  const metricX = DATA.metrics.map(r => Number(r.N));
  const mseSeries = [{{name: "MSE", x: metricX, y: DATA.metrics.map(r => Number(r.MSE_mean)), color: "#d9821f", width: 1.5}}];
  new LineChart(document.getElementById("mseChart"), mseSeries, {{title: "MSE vs N", xLabel: "N", yLabel: "MSE"}});
  const r2Series = [{{name: "R2", x: metricX, y: DATA.metrics.map(r => Number(r.R2_mean)), color: "#b46b38", width: 1.5}}];
  new LineChart(document.getElementById("r2Chart"), r2Series, {{title: "R2 vs N", xLabel: "N", yLabel: "R2"}});

  const table = document.createElement("table");
  const keys = Object.keys(DATA.metrics[0] || {{}});
  table.innerHTML = "<thead><tr>" + keys.map(k => `<th>${{k}}</th>`).join("") + "</tr></thead>" +
    "<tbody>" + DATA.metrics.map(row => "<tr>" + keys.map(k => `<td>${{typeof row[k] === "number" ? niceNumber(row[k]) : row[k]}}</td>`).join("") + "</tr>").join("") + "</tbody>";
  document.getElementById("metricTable").appendChild(table);
}}
init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
