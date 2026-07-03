"""Plotting 탭 – PlottingApp 리팩터링"""

from __future__ import annotations

import csv
import math
import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matlab_generator import (
    GroupDefinition,
    PlotDefinition,
    SeriesRef,
    build_matlab_script,
    default_output_formats,
    detect_demo_groups,
)


class PlottingTab:
    MANUAL_TEXT = """[FEDL Plotting Demo 사용 매뉴얼]

1. raw_data.csv 로드
- 기본값은 plotting/raw_data.csv 입니다.
- 다른 CSV를 쓰려면 [raw_data 선택]으로 바꿉니다.

2. Series / Group / Plot 빌드
- 현재 1) Series Browser, 2) Group Builder, 3) Plot Builder 구조는 유지됩니다.
- Group 내부의 Series는 선택 순서대로 이어 붙여 한 곡선으로 처리됩니다.

3. 축 설정
- X 컬럼과 Y 컬럼은 raw_data 안의 숫자형 컬럼 중에서 선택합니다.
- Y 스케일은 Linear 또는 Log 중에서 고릅니다.
- 축 이름은 직접 입력할 수 있습니다.

4. Demo 자동 구성
- [Demo 구성] 버튼을 누르면 source_file 기준으로 Sweep1~N, Plot1이 자동 생성됩니다.

5. MATLAB 스크립트 생성 / 실행
- [Graph 출력] 버튼은 R2025a를 호출해 .png를 저장합니다.

[참고]
- MATLAB 내장 plot 함수는 별도 설치 대상이 아닙니다.
- 현재 폴더의 plot.m 파일이 있으면 내장 plot을 가릴 수 있어서, 생성기에서는 builtin 함수를 직접 호출합니다.
"""

    def __init__(self, parent: ttk.Frame):
        self.parent = parent

        base_dir = Path(__file__).resolve().parent
        self.base_dir = base_dir
        self.raw_data_path_var = tk.StringVar(value=str(base_dir / "raw_data.csv"))
        self.output_dir_var = tk.StringVar(value=str(base_dir / "outputs"))
        self.matlab_path_var = tk.StringVar(value=self._default_matlab_path())
        self.generated_script_var = tk.StringVar(value="")

        self.x_column_var = tk.StringVar()
        self.y_column_var = tk.StringVar()
        self.x_scale_var = tk.StringVar(value="Linear")
        self.y_scale_var = tk.StringVar(value="Log10")
        self.x_label_var = tk.StringVar()
        self.y_label_var = tk.StringVar()
        self.x_min_var = tk.StringVar(value="")
        self.x_max_var = tk.StringVar(value="")
        self.y_min_var = tk.StringVar(value="")
        self.y_max_var = tk.StringVar(value="")
        self.x_major_tick_var = tk.StringVar(value="")
        self.x_minor_tick_var = tk.StringVar(value="")
        self.y_major_tick_var = tk.StringVar(value="")
        self.y_minor_tick_var = tk.StringVar(value="")
        self.x_reverse_var = tk.BooleanVar(value=False)
        self.y_reverse_var = tk.BooleanVar(value=False)
        self.output_option_summary_var = tk.StringVar(value="")

        self.SCALE_TYPES = ["Linear", "Log10"]
        self.SCALE_TO_MATLAB = {"Linear": "linear", "Log10": "log"}

        self.output_format_vars = {"png": tk.BooleanVar(value=True)}

        self.series_catalog: list[SeriesRef] = []
        self.series_iid_map: dict[str, SeriesRef] = {}
        self.available_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.raw_rows: list[dict[str, str]] = []

        self.staged_series: list[SeriesRef] = []
        self.groups: dict[str, GroupDefinition] = {}
        self.group_order: list[str] = []
        self.plots: dict[str, PlotDefinition] = {}
        self.plot_order: list[str] = []

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.running_process = False

        self._build_ui()
        self._refresh_output_option_summary()
        self.parent.after(200, self._poll_log_queue)

        # 시작 시 raw_data.csv가 있으면 로드
        if Path(self.raw_data_path_var.get()).exists():
            self.load_raw_data(auto_demo=True)
            self._log_plot_shadow_notice()

    # ── Public API (전처리 탭에서 호출) ───────────────────────────────

    def set_raw_data_and_load(self, csv_path: str):
        """외부에서 CSV 경로를 지정하고 로드"""
        self.raw_data_path_var.set(csv_path)
        self.load_raw_data(auto_demo=True)

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        top = ttk.Frame(self.parent, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="raw_data.csv").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.raw_data_path_var).grid(row=1, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(top, text="raw_data 선택", command=self.pick_raw_data).grid(row=1, column=1, padx=(0, 10))
        ttk.Button(top, text="다시 로드", command=lambda: self.load_raw_data(auto_demo=False)).grid(row=1, column=2, padx=(0, 10))

        ttk.Label(top, text="출력 폴더").grid(row=0, column=3, sticky="w")
        ttk.Entry(top, textvariable=self.output_dir_var).grid(row=1, column=3, sticky="ew", padx=(0, 6))
        ttk.Button(top, text="출력 폴더 선택", command=self.pick_output_dir).grid(row=1, column=4, padx=(0, 10))

        ttk.Label(top, text="MATLAB 경로").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.matlab_path_var).grid(row=3, column=0, columnspan=2, sticky="ew", padx=(0, 6))
        ttk.Button(top, text="MATLAB 선택", command=self.pick_matlab_path).grid(row=3, column=2, padx=(0, 10))
        ttk.Button(top, text="경로 재탐지", command=self.detect_matlab_path).grid(row=3, column=3, padx=(0, 10), sticky="w")

        ttk.Button(top, text="Demo 구성", command=self.auto_build_demo).grid(row=3, column=4, padx=(0, 10), sticky="w")
        ttk.Button(top, text="Graph 출력", command=self.run_matlab).grid(row=3, column=5, padx=(0, 10), sticky="w")

        top.columnconfigure(0, weight=2)
        top.columnconfigure(3, weight=2)

        body = ttk.Panedwindow(self.parent, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        left = ttk.Frame(body, padding=8)
        middle = ttk.Frame(body, padding=8)
        right = ttk.Frame(body, padding=8)
        body.add(left, weight=2)
        body.add(middle, weight=2)
        body.add(right, weight=3)

        self._build_series_panel(left)
        self._build_group_panel(middle)
        self._build_plot_panel(right)

        # how to use 버튼 (워터마크는 app.py에서 관리)
        bottom_bar = ttk.Frame(self.parent, padding=(10, 4))
        bottom_bar.pack(fill="x")
        ttk.Button(bottom_bar, text="how to use?", command=self.show_manual).pack(side="left")

    def _build_series_panel(self, parent: ttk.Frame):
        ttk.Label(parent, text="1) Series Browser").pack(anchor="w")
        self.series_tree = ttk.Treeview(parent, show="tree", height=20)
        self.series_tree.pack(fill="both", expand=True, pady=(6, 6))
        self.series_tree.bind("<<TreeviewSelect>>", lambda _event: self._update_series_summary())

        ttk.Button(parent, text="스테이징에 추가", command=self.add_selected_series_to_stage).pack(anchor="w")

        ttk.Label(parent, text="선택 요약").pack(anchor="w", pady=(10, 0))
        self.series_summary = tk.Text(parent, height=5, wrap="word")
        self.series_summary.pack(fill="both", expand=False)

    def _build_group_panel(self, parent: ttk.Frame):
        ttk.Label(parent, text="2) Group Builder").pack(anchor="w")

        name_frame = ttk.Frame(parent)
        name_frame.pack(fill="x", pady=(6, 6))
        ttk.Label(name_frame, text="Group 이름").pack(side="left")
        self.group_name_var = tk.StringVar()
        ttk.Entry(name_frame, textvariable=self.group_name_var).pack(side="left", fill="x", expand=True, padx=(8, 0))

        ttk.Label(parent, text="Group 내부 Series 순서").pack(anchor="w")
        stage_frame = ttk.Frame(parent)
        stage_frame.pack(fill="both", expand=True, pady=(4, 6))
        self.stage_listbox = tk.Listbox(stage_frame, exportselection=False)
        self.stage_listbox.pack(side="left", fill="both", expand=True)

        stage_buttons = ttk.Frame(stage_frame)
        stage_buttons.pack(side="left", padx=(6, 0), fill="y")
        ttk.Button(stage_buttons, text="▲", width=4, command=self.move_stage_up).pack(pady=(0, 4))
        ttk.Button(stage_buttons, text="▼", width=4, command=self.move_stage_down).pack(pady=(0, 4))
        ttk.Button(stage_buttons, text="삭제", width=6, command=self.remove_stage_item).pack(pady=(0, 4))
        ttk.Button(stage_buttons, text="비우기", width=6, command=self.clear_stage).pack()

        group_action_frame = ttk.Frame(parent)
        group_action_frame.pack(fill="x", pady=(0, 8))
        ttk.Button(group_action_frame, text="Group 저장/수정", command=self.save_group).pack(side="left")
        ttk.Button(group_action_frame, text="Group 삭제", command=self.delete_group).pack(side="left", padx=(6, 0))

        ttk.Label(parent, text="생성된 Group").pack(anchor="w")
        self.group_listbox = tk.Listbox(parent, height=12, exportselection=False)
        self.group_listbox.pack(fill="both", expand=True)
        self.group_listbox.bind("<<ListboxSelect>>", lambda _event: self.load_selected_group())

    def _build_plot_panel(self, parent: ttk.Frame):
        ttk.Label(parent, text="3) Plot Builder").pack(anchor="w")

        plot_name_frame = ttk.Frame(parent)
        plot_name_frame.pack(fill="x", pady=(6, 6))
        ttk.Label(plot_name_frame, text="Plot 이름").pack(side="left")
        self.plot_name_var = tk.StringVar()
        ttk.Entry(plot_name_frame, textvariable=self.plot_name_var).pack(side="left", fill="x", expand=True, padx=(8, 0))

        ttk.Label(parent, text="Plot에 포함할 Group").pack(anchor="w")
        self.plot_group_listbox = tk.Listbox(parent, selectmode="extended", height=12, exportselection=False)
        self.plot_group_listbox.pack(fill="both", expand=False, pady=(4, 6))

        plot_action_frame = ttk.Frame(parent)
        plot_action_frame.pack(fill="x", pady=(0, 8))
        ttk.Button(plot_action_frame, text="Plot 저장/수정", command=self.save_plot).pack(side="left")
        ttk.Button(plot_action_frame, text="Plot 삭제", command=self.delete_plot).pack(side="left", padx=(6, 0))

        ttk.Label(parent, text="생성된 Plot").pack(anchor="w")
        self.plot_listbox = tk.Listbox(parent, height=6, exportselection=False)
        self.plot_listbox.pack(fill="x", pady=(4, 8))
        self.plot_listbox.bind("<<ListboxSelect>>", lambda _event: self.load_selected_plot())

        option_frame = ttk.LabelFrame(parent, text="4) 축 옵션", padding=8)
        option_frame.pack(fill="x", pady=(4, 8))
        btn_row = ttk.Frame(option_frame)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="축 옵션 열기", command=self.open_output_options_dialog).pack(side="left")
        ttk.Button(btn_row, text="미리보기", command=self.show_preview_dialog).pack(side="left", padx=(8, 0))
        ttk.Label(option_frame, textvariable=self.output_option_summary_var, justify="left", wraplength=520).pack(anchor="w", pady=(8, 0))

        ttk.Label(parent, text="실행 로그").pack(anchor="w")
        self.log_text = tk.Text(parent, height=16, wrap="word")
        self.log_text.pack(fill="both", expand=True)

    # ── Actions ───────────────────────────────────────────────────────

    def show_manual(self):
        win = tk.Toplevel(self.parent)
        win.title("How to use")
        win.geometry("860x620")
        frame = ttk.Frame(win, padding=10)
        frame.pack(fill="both", expand=True)
        text = tk.Text(frame, wrap="word")
        text.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        scrollbar.pack(side="right", fill="y")
        text.configure(yscrollcommand=scrollbar.set)
        text.insert("1.0", self.MANUAL_TEXT)
        text.configure(state="disabled")

    def pick_raw_data(self):
        path = filedialog.askopenfilename(title="raw_data.csv 선택", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.raw_data_path_var.set(path)

    def pick_output_dir(self):
        path = filedialog.askdirectory(title="출력 폴더 선택")
        if path:
            self.output_dir_var.set(path)

    def pick_matlab_path(self):
        path = filedialog.askopenfilename(title="MATLAB 실행 파일 선택", filetypes=[("MATLAB executable", "matlab.exe"), ("All files", "*.*")])
        if path:
            self.matlab_path_var.set(path)

    def detect_matlab_path(self):
        path = self._default_matlab_path()
        self.matlab_path_var.set(path)
        self.log(f"MATLAB 경로 설정: {path}")

    def load_raw_data(self, auto_demo: bool):
        path = Path(self.raw_data_path_var.get().strip())
        if not path.exists():
            messagebox.showerror("로드 실패", f"파일이 없습니다.\n{path}")
            return

        try:
            with path.open("r", encoding="utf-8", newline="") as file_obj:
                reader = csv.DictReader(file_obj)
                self.available_columns = list(reader.fieldnames or [])
                rows = list(reader)
        except Exception as exc:
            messagebox.showerror("로드 실패", str(exc))
            return

        seen: set[tuple[str, str]] = set()
        series_refs: list[SeriesRef] = []
        for row in rows:
            source_file = (row.get("source_file") or "").strip()
            setup_title = (row.get("setup_title") or "").strip()
            if not source_file or not setup_title:
                continue
            key = (source_file, setup_title)
            if key in seen:
                continue
            seen.add(key)
            series_refs.append(SeriesRef(source_file=source_file, setup_title=setup_title))

        self.series_catalog = series_refs
        self.raw_rows = rows
        self.numeric_columns = self._infer_numeric_columns(rows)
        self._render_series_tree()
        self._refresh_axis_selectors()

        self.series_summary.delete("1.0", tk.END)
        self.series_summary.insert(tk.END, f"Series {len(self.series_catalog)}개 로드\n")
        self.series_summary.insert(tk.END, f"Numeric columns: {', '.join(self.numeric_columns)}")
        self.log(f"raw_data 로드 완료: {path}")
        self.log(f"숫자형 컬럼: {', '.join(self.numeric_columns)}")

        if auto_demo and not self.groups and not self.plots:
            self.auto_build_demo()

    def _infer_numeric_columns(self, rows: list[dict[str, str]]) -> list[str]:
        numeric_columns: list[str] = []
        for column in self.available_columns:
            if column in {"source_file", "setup_title", "source_line", "index"}:
                continue
            checked = 0
            valid = 0
            for row in rows:
                value = (row.get(column) or "").strip()
                if not value:
                    continue
                checked += 1
                try:
                    float(value)
                    valid += 1
                except ValueError:
                    break
                if checked >= 60:
                    break
            if checked > 0 and checked == valid:
                numeric_columns.append(column)
        return numeric_columns or [col for col in self.available_columns if col not in {"source_file", "setup_title", "source_line", "index"}]

    def _refresh_axis_selectors(self):
        default_x = self._pick_default_column(preferred=["V1", "Vg", "GateV", "DrainV", "Vd"])
        default_y = self._pick_default_column(preferred=["Abs_Id", "DrainI", "I1", "Id", "I3"], exclude={default_x})

        if not self.x_column_var.get() or self.x_column_var.get() not in self.numeric_columns:
            self.x_column_var.set(default_x)
        if not self.y_column_var.get() or self.y_column_var.get() not in self.numeric_columns:
            self.y_column_var.set(default_y)

        if not self.x_label_var.get().strip():
            self.x_label_var.set(self.x_column_var.get())
        if not self.y_label_var.get().strip():
            self.y_label_var.set(self.y_column_var.get())
        self._refresh_output_option_summary()

    def _scale_display(self, var: tk.StringVar) -> str:
        return var.get() or "Linear"

    def _matlab_scale(self, display: str) -> str:
        return self.SCALE_TO_MATLAB.get(display, "linear")

    def _is_log_display(self, display: str) -> bool:
        return display == "Log10"

    def _refresh_output_option_summary(self):
        x_range = f"{self.x_min_var.get().strip() or 'auto'} ~ {self.x_max_var.get().strip() or 'auto'}"
        y_range = f"{self.y_min_var.get().strip() or 'auto'} ~ {self.y_max_var.get().strip() or 'auto'}"
        self.output_option_summary_var.set(
            " / ".join(
                [
                    f"X:{self.x_column_var.get() or '-'} ({self._scale_display(self.x_scale_var)})",
                    f"Y:{self.y_column_var.get() or '-'} ({self._scale_display(self.y_scale_var)})",
                    f"X범위:{x_range}",
                    f"Y범위:{y_range}",
                    f"X주:{self.x_major_tick_var.get().strip() or 'auto'} 보조:{self.x_minor_tick_var.get().strip() or 'auto'}",
                    f"Y주:{self.y_major_tick_var.get().strip() or 'auto'} 보조:{self.y_minor_tick_var.get().strip() or 'auto'}",
                ]
            )
        )

    def open_output_options_dialog(self):
        win = tk.Toplevel(self.parent)
        win.title("축 옵션")
        win.geometry("520x580")
        win.resizable(False, False)

        notebook = ttk.Notebook(win)
        notebook.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        x_tab = ttk.Frame(notebook, padding=12)
        notebook.add(x_tab, text="  X 축 (Horizontal)  ")
        self._build_axis_tab(x_tab, axis="x")

        y_tab = ttk.Frame(notebook, padding=12)
        notebook.add(y_tab, text="  Y 축 (Vertical)  ")
        self._build_axis_tab(y_tab, axis="y")

        action_row = ttk.Frame(win, padding=(10, 10))
        action_row.pack(fill="x")
        ttk.Button(action_row, text="미리보기", command=self.show_preview_dialog).pack(side="right", padx=(8, 0))
        ttk.Button(action_row, text="적용", command=lambda: (self._refresh_output_option_summary(), win.destroy())).pack(side="right")

    def _build_axis_tab(self, parent: ttk.Frame, axis: str):
        if axis == "x":
            col_var, label_var = self.x_column_var, self.x_label_var
            scale_var = self.x_scale_var
            min_var, max_var = self.x_min_var, self.x_max_var
            major_var, minor_var = self.x_major_tick_var, self.x_minor_tick_var
            reverse_var = self.x_reverse_var
        else:
            col_var, label_var = self.y_column_var, self.y_label_var
            scale_var = self.y_scale_var
            min_var, max_var = self.y_min_var, self.y_max_var
            major_var, minor_var = self.y_major_tick_var, self.y_minor_tick_var
            reverse_var = self.y_reverse_var

        row = 0
        ttk.Label(parent, text="컬럼").grid(row=row, column=0, sticky="w")
        ttk.Combobox(parent, textvariable=col_var, values=self.numeric_columns,
                     state="readonly", width=22).grid(row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(parent, text="축 이름").grid(row=row, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(parent, textvariable=label_var, width=24).grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        row += 1
        scale_frame = ttk.LabelFrame(parent, text="Scale", padding=10)
        scale_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(12, 0))

        ttk.Label(scale_frame, text="From").grid(row=0, column=0, sticky="w")
        ttk.Entry(scale_frame, textvariable=min_var, width=14).grid(row=0, column=1, sticky="w", padx=(8, 16))
        ttk.Label(scale_frame, text="To").grid(row=0, column=2, sticky="w")
        ttk.Entry(scale_frame, textvariable=max_var, width=14).grid(row=0, column=3, sticky="w", padx=(8, 0))

        ttk.Label(scale_frame, text="Type").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(scale_frame, textvariable=scale_var, values=self.SCALE_TYPES,
                     state="readonly", width=12).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Checkbutton(scale_frame, text="Reverse", variable=reverse_var
                        ).grid(row=1, column=2, columnspan=2, sticky="w", padx=(16, 0), pady=(8, 0))

        row += 1
        major_frame = ttk.LabelFrame(parent, text="Major Ticks", padding=10)
        major_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Label(major_frame, text="Value (간격)").grid(row=0, column=0, sticky="w")
        ttk.Entry(major_frame, textvariable=major_var, width=14).grid(row=0, column=1, sticky="w", padx=(8, 0))

        row += 1
        minor_frame = ttk.LabelFrame(parent, text="Minor Ticks", padding=10)
        minor_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Label(minor_frame, text="Count (개수)").grid(row=0, column=0, sticky="w")
        ttk.Entry(minor_frame, textvariable=minor_var, width=14).grid(row=0, column=1, sticky="w", padx=(8, 0))

        parent.columnconfigure(1, weight=1)

    def _to_float_or_none(self, text: str) -> float | None:
        value = text.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def show_preview_dialog(self):
        if not self.raw_rows:
            messagebox.showwarning("알림", "raw_data를 먼저 로드하세요.")
            return
        if not self.groups or not self.plots:
            messagebox.showwarning("알림", "Group과 Plot을 먼저 구성하세요.")
            return

        x_col = self.x_column_var.get().strip()
        y_col = self.y_column_var.get().strip()
        if not x_col or not y_col:
            messagebox.showwarning("알림", "X / Y 컬럼을 먼저 선택하세요.")
            return

        curves = self._collect_preview_curves(x_col, y_col)
        if not curves:
            messagebox.showwarning("알림", "미리보기 가능한 데이터가 없습니다.")
            return

        all_x = [x for _name, xs, _ys in curves for x in xs]
        all_y = [y for _name, _xs, ys in curves for y in ys]
        if not all_x or not all_y:
            messagebox.showwarning("알림", "유효한 포인트가 없습니다.")
            return

        x_min = self._to_float_or_none(self.x_min_var.get())
        x_max = self._to_float_or_none(self.x_max_var.get())
        y_min = self._to_float_or_none(self.y_min_var.get())
        y_max = self._to_float_or_none(self.y_max_var.get())

        x_lo = min(all_x) if x_min is None else x_min
        x_hi = max(all_x) if x_max is None else x_max
        y_lo = min(all_y) if y_min is None else y_min
        y_hi = max(all_y) if y_max is None else y_max
        if x_hi <= x_lo or y_hi <= y_lo:
            messagebox.showerror("미리보기 실패", "축 범위를 확인하세요. 최소값은 최대값보다 작아야 합니다.")
            return

        preview = tk.Toplevel(self.parent)
        preview.title("미리보기")
        preview.geometry("980x560")
        canvas = tk.Canvas(preview, bg="black", highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        preview.update_idletasks()
        width = max(canvas.winfo_width(), 900)
        height = max(canvas.winfo_height(), 500)
        left, right, top, bottom = 70, 240, 40, 90
        plot_w = width - left - right
        plot_h = height - top - bottom

        colors = ["#FF5559", "#FFAA55", "#55FF55", "#55AAFF", "#AA66FF", "#FFD54F", "#4DD0E1"]

        def map_x(value: float) -> float:
            return left + (value - x_lo) / (x_hi - x_lo) * plot_w

        is_log = self._is_log_display(self.y_scale_var.get())
        y_lo_safe = max(y_lo, 1e-300) if is_log else y_lo
        y_hi_safe = max(y_hi, y_lo_safe * 10.0) if is_log else y_hi

        def map_y(value: float) -> float:
            if is_log:
                lv = math.log10(max(value, y_lo_safe))
                l0 = math.log10(y_lo_safe)
                l1 = math.log10(y_hi_safe)
                ratio = 0.0 if l1 == l0 else (lv - l0) / (l1 - l0)
            else:
                ratio = (value - y_lo) / (y_hi - y_lo)
            return top + (1.0 - ratio) * plot_h

        canvas.create_rectangle(left, top, left + plot_w, top + plot_h, outline="#D0D0D0", width=1)

        def _nice_ticks_linear(lo: float, hi: float, n: int = 5) -> list:
            if hi <= lo:
                return [lo, hi]
            span = hi - lo
            raw_step = span / n
            mag = 10 ** math.floor(math.log10(max(abs(raw_step), 1e-300)))
            nice = min([1, 2, 5, 10], key=lambda s: abs(s * mag - raw_step))
            step = nice * mag
            start = math.ceil(lo / step - 1e-9) * step
            ticks, v = [], start
            while v <= hi + step * 1e-6:
                if lo - step * 0.01 <= v <= hi + step * 0.01:
                    ticks.append(round(v, 10))
                v += step
            return ticks

        def _nice_ticks_log(lo: float, hi: float) -> list:
            lo_exp = math.floor(math.log10(max(lo, 1e-300)))
            hi_exp = math.ceil(math.log10(max(hi, 1e-300)))
            ticks = [10.0 ** e for e in range(int(lo_exp), int(hi_exp) + 1)
                     if lo * 0.999 <= 10.0 ** e <= hi * 1.001]
            return ticks if ticks else [lo, hi]

        def _fmt_tick_linear(val: float) -> str:
            if val == 0:
                return "0"
            if abs(val) >= 1e4 or (0 < abs(val) < 1e-2):
                return f"{val:.2e}"
            if abs(val) >= 100:
                return f"{val:.0f}"
            return f"{val:.3g}"

        def _fmt_tick_log(val: float) -> str:
            if val == 0:
                return "0"
            exp = round(math.log10(max(abs(val), 1e-300)))
            return f"10$^{{{exp}}}$" if abs(exp) < 30 else f"1e{exp:+d}"

        fmt_x = _fmt_tick_linear
        fmt_y = _fmt_tick_log if is_log else _fmt_tick_linear

        x_ticks = _nice_ticks_linear(x_lo, x_hi, 5)
        y_ticks = (_nice_ticks_log(y_lo_safe, y_hi_safe) if is_log
                   else _nice_ticks_linear(y_lo, y_hi, 5))

        tick_len = 5
        for xv in x_ticks:
            gx = map_x(xv)
            if left - 1 <= gx <= left + plot_w + 1:
                canvas.create_line(gx, top, gx, top + plot_h, fill="#2D2D2D")
                canvas.create_line(gx, top + plot_h, gx, top + plot_h + tick_len, fill="#C0C0C0", width=1)
                canvas.create_text(gx, top + plot_h + tick_len + 3,
                                   text=fmt_x(xv), fill="#C0C0C0",
                                   anchor="n", font=("Segoe UI", 9))

        for yv in y_ticks:
            gy = map_y(yv)
            if top - 1 <= gy <= top + plot_h + 1:
                canvas.create_line(left, gy, left + plot_w, gy, fill="#2D2D2D")
                canvas.create_line(left - tick_len, gy, left, gy, fill="#C0C0C0", width=1)
                canvas.create_text(left - tick_len - 4, gy,
                                   text=fmt_y(yv), fill="#C0C0C0",
                                   anchor="e", font=("Segoe UI", 9))

        for idx, (name, xs, ys) in enumerate(curves):
            points: list[float] = []
            for xv, yv in zip(xs, ys):
                if xv < x_lo or xv > x_hi or yv < y_lo or yv > y_hi:
                    continue
                points.extend([map_x(xv), map_y(yv)])
            if len(points) >= 4:
                canvas.create_line(*points, fill=colors[idx % len(colors)], width=2, smooth=False)

        canvas.create_text(left + plot_w / 2, top + plot_h + 40, text=self.x_label_var.get().strip() or x_col, fill="white", font=("Segoe UI", 13, "bold"))
        canvas.create_text(25, top + plot_h / 2, text=self.y_label_var.get().strip() or y_col, fill="white", angle=90, font=("Segoe UI", 13, "bold"))
        canvas.create_text(width / 2, 18, text=f"Preview | {y_col} - {x_col}", fill="white", font=("Segoe UI", 14, "bold"))

        legend_x = left + plot_w + 20
        legend_y = top + 20
        for idx, (name, _xs, _ys) in enumerate(curves):
            y_pos = legend_y + idx * 24
            color = colors[idx % len(colors)]
            canvas.create_line(legend_x, y_pos, legend_x + 26, y_pos, fill=color, width=3)
            canvas.create_text(legend_x + 32, y_pos, text=name, fill="white", anchor="w", font=("Segoe UI", 11))

    def _collect_preview_curves(self, x_col: str, y_col: str) -> list[tuple[str, list[float], list[float]]]:
        plot_name = self.plot_name_var.get().strip() or (self.plot_order[0] if self.plot_order else "")
        if plot_name not in self.plots:
            return []

        y_is_log = self._is_log_display(self.y_scale_var.get())
        curves: list[tuple[str, list[float], list[float]]] = []
        for group_name in self.plots[plot_name].group_names:
            group_def = self.groups.get(group_name)
            if not group_def:
                continue
            xs: list[float] = []
            ys: list[float] = []
            for series in group_def.series_refs:
                rows = [row for row in self.raw_rows if (row.get("source_file") or "").strip() == series.source_file and (row.get("setup_title") or "").strip() == series.setup_title]
                rows.sort(key=lambda row: float((row.get("source_line") or "0").strip() or "0"))
                for row in rows:
                    try:
                        xv = float((row.get(x_col) or "").strip())
                        yv = float((row.get(y_col) or "").strip())
                    except ValueError:
                        continue
                    if y_is_log and yv <= 0:
                        continue
                    xs.append(xv)
                    ys.append(yv)
            if len(xs) >= 2:
                curves.append((group_name, xs, ys))
        return curves

    def _pick_default_column(self, preferred: list[str], exclude: set[str] | None = None) -> str:
        exclude = exclude or set()
        for item in preferred:
            if item in self.numeric_columns and item not in exclude:
                return item
        for item in self.numeric_columns:
            if item not in exclude:
                return item
        return ""

    def _render_series_tree(self):
        self.series_tree.delete(*self.series_tree.get_children())
        self.series_iid_map.clear()

        by_file: dict[str, list[SeriesRef]] = {}
        for ref in self.series_catalog:
            by_file.setdefault(ref.source_file, []).append(ref)

        for source_file in sorted(by_file):
            parent_id = self.series_tree.insert("", "end", text=source_file, open=True)
            for ref in sorted(by_file[source_file], key=lambda item: item.setup_title):
                iid = f"{ref.source_file}|{ref.setup_title}"
                self.series_iid_map[iid] = ref
                self.series_tree.insert(parent_id, "end", iid=iid, text=ref.setup_title)

    def _update_series_summary(self):
        selected = self._selected_series_refs()
        self.series_summary.delete("1.0", tk.END)
        if not selected:
            self.series_summary.insert(tk.END, "선택된 Series 없음")
            return
        for ref in selected:
            self.series_summary.insert(tk.END, f"{ref.source_file} | {ref.setup_title}\n")

    def _selected_series_refs(self) -> list[SeriesRef]:
        refs: list[SeriesRef] = []
        for item_id in self.series_tree.selection():
            if item_id in self.series_iid_map:
                refs.append(self.series_iid_map[item_id])
        return refs

    def add_selected_series_to_stage(self):
        selected = self._selected_series_refs()
        if not selected:
            messagebox.showwarning("알림", "setup_title 항목을 선택하세요.")
            return
        for ref in selected:
            if ref not in self.staged_series:
                self.staged_series.append(ref)
        self._refresh_stage_listbox()

    def move_stage_up(self):
        selection = self.stage_listbox.curselection()
        if not selection or selection[0] <= 0:
            return
        index = selection[0]
        self.staged_series[index - 1], self.staged_series[index] = self.staged_series[index], self.staged_series[index - 1]
        self._refresh_stage_listbox(index - 1)

    def move_stage_down(self):
        selection = self.stage_listbox.curselection()
        if not selection or selection[0] >= len(self.staged_series) - 1:
            return
        index = selection[0]
        self.staged_series[index + 1], self.staged_series[index] = self.staged_series[index], self.staged_series[index + 1]
        self._refresh_stage_listbox(index + 1)

    def remove_stage_item(self):
        selection = self.stage_listbox.curselection()
        if not selection:
            return
        del self.staged_series[selection[0]]
        self._refresh_stage_listbox()

    def clear_stage(self):
        self.staged_series = []
        self._refresh_stage_listbox()

    def _refresh_stage_listbox(self, selected_index: int | None = None):
        self.stage_listbox.delete(0, tk.END)
        for ref in self.staged_series:
            self.stage_listbox.insert(tk.END, f"{ref.source_file} | {ref.setup_title}")
        if selected_index is not None and 0 <= selected_index < len(self.staged_series):
            self.stage_listbox.selection_set(selected_index)

    def save_group(self):
        name = self.group_name_var.get().strip()
        if not name:
            messagebox.showwarning("알림", "Group 이름을 입력하세요.")
            return
        if not self.staged_series:
            messagebox.showwarning("알림", "최소 1개 이상의 Series가 필요합니다.")
            return

        self.groups[name] = GroupDefinition(name=name, series_refs=list(self.staged_series))
        if name not in self.group_order:
            self.group_order.append(name)
        self._refresh_group_listbox(select_name=name)
        self._refresh_plot_group_listbox()
        self.log(f"Group 저장: {name} ({len(self.staged_series)} series)")

    def delete_group(self):
        selection = self.group_listbox.curselection()
        if not selection:
            return
        name = self.group_order[selection[0]]
        self.groups.pop(name, None)
        self.group_order.remove(name)
        for plot_name in list(self.plot_order):
            plot_def = self.plots[plot_name]
            if name in plot_def.group_names:
                filtered = [item for item in plot_def.group_names if item != name]
                self.plots[plot_name] = PlotDefinition(name=plot_name, group_names=filtered)
        self._refresh_group_listbox()
        self._refresh_plot_group_listbox()
        self._refresh_plot_listbox()
        self.group_name_var.set("")
        self.clear_stage()
        self.log(f"Group 삭제: {name}")

    def load_selected_group(self):
        selection = self.group_listbox.curselection()
        if not selection:
            return
        name = self.group_order[selection[0]]
        group_def = self.groups[name]
        self.group_name_var.set(group_def.name)
        self.staged_series = list(group_def.series_refs)
        self._refresh_stage_listbox()

    def _refresh_group_listbox(self, select_name: str | None = None):
        self.group_listbox.delete(0, tk.END)
        selected_index = None
        for index, name in enumerate(self.group_order):
            group_def = self.groups[name]
            self.group_listbox.insert(tk.END, f"{name} ({len(group_def.series_refs)} series)")
            if name == select_name:
                selected_index = index
        if selected_index is not None:
            self.group_listbox.selection_set(selected_index)

    def _refresh_plot_group_listbox(self, selected_names: list[str] | None = None):
        selected_names = selected_names or []
        self.plot_group_listbox.delete(0, tk.END)
        for index, name in enumerate(self.group_order):
            self.plot_group_listbox.insert(tk.END, name)
            if name in selected_names:
                self.plot_group_listbox.selection_set(index)

    def save_plot(self):
        name = self.plot_name_var.get().strip()
        if not name:
            messagebox.showwarning("알림", "Plot 이름을 입력하세요.")
            return
        selected_indices = self.plot_group_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("알림", "최소 1개 이상의 Group을 선택하세요.")
            return
        group_names = [self.group_order[index] for index in selected_indices]
        self.plots[name] = PlotDefinition(name=name, group_names=group_names)
        if name not in self.plot_order:
            self.plot_order.append(name)
        self._refresh_plot_listbox(select_name=name)
        self.log(f"Plot 저장: {name} ({len(group_names)} groups)")

    def delete_plot(self):
        selection = self.plot_listbox.curselection()
        if not selection:
            return
        name = self.plot_order[selection[0]]
        self.plots.pop(name, None)
        self.plot_order.remove(name)
        self._refresh_plot_listbox()
        self.plot_name_var.set("")
        self._refresh_plot_group_listbox()
        self.log(f"Plot 삭제: {name}")

    def load_selected_plot(self):
        selection = self.plot_listbox.curselection()
        if not selection:
            return
        name = self.plot_order[selection[0]]
        plot_def = self.plots[name]
        self.plot_name_var.set(plot_def.name)
        self._refresh_plot_group_listbox(selected_names=plot_def.group_names)

    def _refresh_plot_listbox(self, select_name: str | None = None):
        self.plot_listbox.delete(0, tk.END)
        selected_index = None
        for index, name in enumerate(self.plot_order):
            plot_def = self.plots[name]
            self.plot_listbox.insert(tk.END, f"{name} ({len(plot_def.group_names)} groups)")
            if name == select_name:
                selected_index = index
        if selected_index is not None:
            self.plot_listbox.selection_set(selected_index)

    def auto_build_demo(self):
        if not self.series_catalog:
            self.load_raw_data(auto_demo=False)
            if not self.series_catalog:
                return

        groups, plots = detect_demo_groups(self.series_catalog)
        self.groups = {group.name: group for group in groups}
        self.group_order = [group.name for group in groups]
        self.plots = {plot.name: plot for plot in plots}
        self.plot_order = [plot.name for plot in plots]

        self._refresh_group_listbox(select_name=self.group_order[0] if self.group_order else None)
        default_group_names = self.plots[self.plot_order[0]].group_names if self.plot_order else []
        self._refresh_plot_group_listbox(selected_names=default_group_names)
        self._refresh_plot_listbox(select_name=self.plot_order[0] if self.plot_order else None)

        if self.group_order:
            first_group = self.groups[self.group_order[0]]
            self.group_name_var.set(first_group.name)
            self.staged_series = list(first_group.series_refs)
            self._refresh_stage_listbox()
        if self.plot_order:
            self.plot_name_var.set(self.plot_order[0])

        self.log(f"Demo 구성 완료: {len(groups)} groups, {len(plots)} plots")

    def selected_output_formats(self) -> list[str]:
        return [name for name, var in self.output_format_vars.items() if var.get()]

    def generate_script(self) -> str | None:
        if not self.groups or not self.plots:
            messagebox.showwarning("알림", "Group과 Plot을 먼저 구성하세요.")
            return None

        x_column = self.x_column_var.get().strip()
        y_column = self.y_column_var.get().strip()
        x_label = self.x_label_var.get().strip()
        y_label = self.y_label_var.get().strip()
        if not x_column or not y_column:
            messagebox.showwarning("알림", "X / Y 컬럼을 선택하세요.")
            return None
        if x_column == y_column:
            messagebox.showwarning("알림", "X 컬럼과 Y 컬럼은 서로 달라야 합니다.")
            return None
        if not x_label:
            x_label = x_column
            self.x_label_var.set(x_label)
        if not y_label:
            y_label = y_column
            self.y_label_var.set(y_label)

        output_formats = ["png"]

        output_dir = Path(self.output_dir_var.get().strip())
        output_dir.mkdir(parents=True, exist_ok=True)
        script_path = output_dir / "generated_plotting_demo.m"

        script_text = build_matlab_script(
            raw_data_path=self.raw_data_path_var.get().strip(),
            output_dir=str(output_dir),
            groups=[self.groups[name] for name in self.group_order],
            plots=[self.plots[name] for name in self.plot_order],
            x_column=x_column,
            y_column=y_column,
            x_scale=self._matlab_scale(self.x_scale_var.get()),
            y_scale=self._matlab_scale(self.y_scale_var.get()),
            x_label=x_label,
            y_label=y_label,
            x_min=self.x_min_var.get().strip(),
            x_max=self.x_max_var.get().strip(),
            y_min=self.y_min_var.get().strip(),
            y_max=self.y_max_var.get().strip(),
            x_major_tick=self.x_major_tick_var.get().strip(),
            x_minor_tick=self.x_minor_tick_var.get().strip(),
            y_major_tick=self.y_major_tick_var.get().strip(),
            y_minor_tick=self.y_minor_tick_var.get().strip(),
            x_reverse=self.x_reverse_var.get(),
            y_reverse=self.y_reverse_var.get(),
            output_formats=output_formats,
        )

        script_path.write_text(script_text, encoding="utf-8")
        self.generated_script_var.set(str(script_path))
        self._refresh_output_option_summary()
        self.log(f"MATLAB 스크립트 생성: {script_path}")
        self.log(
            "설정: "
            f"X={x_column} ({self._scale_display(self.x_scale_var)}), "
            f"Y={y_column} ({self._scale_display(self.y_scale_var)}), "
            f"xlabel={x_label}, ylabel={y_label}, "
            f"xRange=[{self.x_min_var.get() or 'auto'}, {self.x_max_var.get() or 'auto'}], "
            f"yRange=[{self.y_min_var.get() or 'auto'}, {self.y_max_var.get() or 'auto'}], "
            f"xTick={self.x_major_tick_var.get() or 'auto'}/{self.x_minor_tick_var.get() or 'auto'}, "
            f"yTick={self.y_major_tick_var.get() or 'auto'}/{self.y_minor_tick_var.get() or 'auto'}"
        )
        return str(script_path)

    def run_matlab(self):
        if self.running_process:
            messagebox.showinfo("알림", "MATLAB 실행이 이미 진행 중입니다.")
            return

        matlab_path = Path(self.matlab_path_var.get().strip())
        if not matlab_path.exists():
            messagebox.showerror("실행 실패", f"MATLAB 실행 파일이 없습니다.\n{matlab_path}")
            return

        script_path = self.generate_script()
        if not script_path:
            return

        self.running_process = True
        self.log("MATLAB 실행 시작")

        worker = threading.Thread(target=self._run_matlab_worker, args=(str(matlab_path), script_path), daemon=True)
        worker.start()

    def _run_matlab_worker(self, matlab_path: str, script_path: str):
        script_arg = script_path.replace("\\", "/")
        command = [matlab_path, "-batch", f"run('{script_arg}')"]
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                self.log_queue.put(line.rstrip())
            code = process.wait()
            self.log_queue.put(f"MATLAB 종료 코드: {code}")
        except Exception as exc:
            self.log_queue.put(f"MATLAB 실행 실패: {exc}")
        finally:
            self.log_queue.put("__MATLAB_DONE__")

    def _poll_log_queue(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                if message == "__MATLAB_DONE__":
                    self.running_process = False
                    self.log("MATLAB 실행 종료")
                    messagebox.showinfo("완료", "MATLAB 실행이 끝났습니다. 로그를 확인하세요.")
                    continue
                self.log(message)
        except queue.Empty:
            pass
        self.parent.after(200, self._poll_log_queue)

    def log(self, message: str):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def _log_plot_shadow_notice(self):
        plot_file = self.base_dir / "plot.m"
        if plot_file.exists():
            self.log(f"참고: {plot_file.name} 이 MATLAB 내장 plot 함수를 가릴 수 있습니다. 생성기는 builtin 호출로 우회합니다.")

    @staticmethod
    def _default_matlab_path() -> str:
        candidates = [
            Path(r"C:\Program Files\MATLAB\R2025a\bin\matlab.exe"),
            Path(r"C:\Program Files\MATLAB\R2024b\bin\matlab.exe"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0])
