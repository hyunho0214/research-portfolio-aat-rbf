from pathlib import Path
import csv
import sys

sys.path.insert(0, "d:/FEDL/Data/FEDL_Data/plotting")
from matlab_generator import SeriesRef, detect_demo_groups, build_matlab_script, default_output_formats

base = Path("d:/FEDL/Data/FEDL_Data/plotting")
raw_path = base / "raw_data.csv"
out_dir = base / "outputs"
script_path = out_dir / "generated_plotting_demo.m"

pairs: list[SeriesRef] = []
seen: set[tuple[str, str]] = set()
with raw_path.open("r", encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        source_file = (row.get("source_file") or "").strip()
        setup_title = (row.get("setup_title") or "").strip()
        key = (source_file, setup_title)
        if not source_file or not setup_title or key in seen:
            continue
        seen.add(key)
        pairs.append(SeriesRef(source_file=source_file, setup_title=setup_title))

groups, plots = detect_demo_groups(pairs)
script = build_matlab_script(
    raw_data_path=str(raw_path),
    output_dir=str(out_dir),
    groups=groups,
    plots=plots,
    x_column="V1",
    y_column="Abs_Id",
    y_scale="log",
    x_label="V1",
    y_label="Abs_Id",
    x_min="-22",
    x_max="22",
    y_min="1e-13",
    y_max="1e-6",
    x_major_tick="10",
    x_minor_tick="1",
    y_major_tick="2",
    y_minor_tick="1",
    output_formats=default_output_formats(),
)
script_path.write_text(script, encoding="utf-8")
print(f"Regenerated: {script_path}")
