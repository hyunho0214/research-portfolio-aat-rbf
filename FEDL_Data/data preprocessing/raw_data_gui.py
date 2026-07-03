import csv
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import xlrd
except Exception:
    xlrd = None

try:
    import openpyxl
except Exception:
    openpyxl = None


class RawDataExtractorApp:
    MANUAL_TEXT = """[RawDataExtractor 사용 매뉴얼]

1. 파일 유형 선택
- 상단에서 파일 유형을 선택합니다.
- Agilent(CSV): .csv 파일 분석
- Keithley(Excel): .xls/.xlsx 파일 분석

2. 파일 선택
- [파일 선택] 버튼으로 해당 유형 파일을 1개 이상 선택합니다.

3. 분석
- [분석] 버튼을 누르면 파일 구조를 읽고 추출 가능한 컬럼을 구성합니다.

4. 저장 경로/파일명 설정
- [raw_data 저장 경로]를 직접 입력하거나 [경로 선택] 버튼으로 지정합니다.

5. 컬럼 선택
- 왼쪽 체크박스에서 필요한 컬럼만 선택합니다.

6. 컬럼 순서 변경
- "선택된 컬럼 순서" 목록에서 컬럼을 고른 뒤 [▲ 위로], [▼ 아래로]로 순서를 바꿉니다.

7. 미리보기 확인
- [미리보기 갱신]을 눌러 오른쪽 [HEAD]/[TAIL] 미리보기를 확인합니다.

8. 추출
- [추출] 버튼을 눌러 설정한 경로로 CSV를 저장합니다.

[권장 순서]
파일 선택 -> 분석 -> 저장 경로 설정 -> 컬럼 선택 -> 순서 조정 -> 미리보기 -> 추출
"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Raw Data Extractor")
        self.root.geometry("1200x760")

        self.selected_files = []
        self.available_columns = []
        self.analyzed_rows = []
        self.output_path_var = tk.StringVar(value=os.path.join(os.getcwd(), "raw_data.csv"))
        self.file_type_var = tk.StringVar(value="Agilent(CSV)")

        # 전체 컬럼 체크 상태 (각 항목은 (col_name, BooleanVar) 튜플)
        self.column_list = []
        # 체크된 컬럼만 별도로 순서 관리
        self.selected_order = []

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="1) 파일 유형").grid(row=0, column=0, sticky="w")
        type_combo = ttk.Combobox(
            top,
            textvariable=self.file_type_var,
            values=["Agilent(CSV)", "Keithley(Excel)"],
            state="readonly",
            width=22,
        )
        type_combo.grid(row=1, column=0, sticky="w", padx=(0, 8))

        ttk.Label(top, text="2) 파일 선택").grid(row=0, column=1, sticky="w")
        self.file_entry = ttk.Entry(top)
        self.file_entry.grid(row=1, column=1, sticky="ew", padx=(0, 8))

        btn_pick = ttk.Button(top, text="파일 선택", command=self.pick_files)
        btn_pick.grid(row=1, column=2, padx=(0, 8))

        btn_analyze = ttk.Button(top, text="분석", command=self.analyze_files)
        btn_analyze.grid(row=1, column=3)

        ttk.Label(top, text="3) raw_data 저장 경로").grid(row=2, column=0, sticky="w", pady=(10, 0))
        out_entry = ttk.Entry(top, textvariable=self.output_path_var)
        out_entry.grid(row=3, column=0, columnspan=2, sticky="ew", padx=(0, 8))
        btn_out = ttk.Button(top, text="경로 선택", command=self.pick_output_path)
        btn_out.grid(row=3, column=2, padx=(0, 8))

        top.columnconfigure(1, weight=1)

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        left = ttk.Frame(body, padding=8)
        right = ttk.Frame(body, padding=8)
        body.add(left, weight=1)
        body.add(right, weight=2)

        ttk.Label(left, text="3) 추출 컬럼 선택").pack(anchor="w")

        self.canvas = tk.Canvas(left, borderwidth=0, height=450)
        self.chk_frame = ttk.Frame(self.canvas)
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set)

        scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((0, 0), window=self.chk_frame, anchor="nw")
        self.chk_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        action_box = ttk.Frame(left)
        action_box.pack(fill="x", pady=(8, 0))

        ttk.Button(action_box, text="미리보기 갱신", command=self.update_preview).pack(side="left")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=(10, 6))
        ttk.Label(left, text="선택된 컬럼 순서 (체크된 컬럼만)").pack(anchor="w")

        order_frame = ttk.Frame(left)
        order_frame.pack(fill="x", pady=(4, 0))

        self.order_listbox = tk.Listbox(order_frame, height=8, exportselection=False)
        self.order_listbox.pack(side="left", fill="x", expand=True)

        order_btns = ttk.Frame(order_frame)
        order_btns.pack(side="left", padx=(6, 0))
        ttk.Button(order_btns, text="▲ 위로", command=self._move_selected_up).pack(fill="x", pady=(0, 4))
        ttk.Button(order_btns, text="▼ 아래로", command=self._move_selected_down).pack(fill="x")

        extract_box = ttk.Frame(left)
        extract_box.pack(fill="x", pady=(8, 0))
        ttk.Button(extract_box, text="추출", command=self.export_raw_data).pack(side="left")

        ttk.Label(right, text="4) 예상 raw_data 미리보기 (HEAD / TAIL)").pack(anchor="w")

        self.preview = tk.Text(right, wrap="none")
        self.preview.pack(fill="both", expand=True)

        x_scroll = ttk.Scrollbar(right, orient="horizontal", command=self.preview.xview)
        y_scroll = ttk.Scrollbar(right, orient="vertical", command=self.preview.yview)
        self.preview.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        x_scroll.pack(fill="x")
        y_scroll.pack(side="right", fill="y")

        # Fixed watermark at the real bottom-left corner of the window.
        self.watermark = ttk.Frame(self.root, padding=(10, 6))
        self.watermark.place(relx=0.0, rely=1.0, anchor="sw")

        wm_text = ttk.Frame(self.watermark)
        wm_text.pack(side="left")
        ttk.Label(wm_text, text="ASNL_HYUNHO.JANG", foreground="#6b6b6b").pack(anchor="w")
        ttk.Label(wm_text, text="무단 배포 금지", foreground="#6b6b6b").pack(anchor="w")
        ttk.Label(wm_text, text="문의: hyunho0214@naver.com", foreground="#6b6b6b").pack(anchor="w")

        ttk.Button(self.watermark, text="how to use?", command=self.show_manual).pack(side="left", padx=(10, 0), anchor="s")

    def show_manual(self):
        win = tk.Toplevel(self.root)
        win.title("How to use")
        win.geometry("760x560")

        frame = ttk.Frame(win, padding=10)
        frame.pack(fill="both", expand=True)

        txt = tk.Text(frame, wrap="word")
        txt.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=txt.yview)
        scroll.pack(side="right", fill="y")
        txt.configure(yscrollcommand=scroll.set)

        txt.insert("1.0", self.MANUAL_TEXT)
        txt.configure(state="disabled")

    def pick_files(self):
        file_type = self.file_type_var.get()
        if file_type == "Keithley(Excel)":
            filters = [("Excel files", "*.xls *.xlsx"), ("All files", "*.*")]
        else:
            filters = [("CSV files", "*.csv"), ("All files", "*.*")]

        files = filedialog.askopenfilenames(
            title="분석 파일 선택",
            filetypes=filters,
        )
        if not files:
            return

        self.selected_files = list(files)
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, f"{len(self.selected_files)}개 파일 선택됨")

    def pick_output_path(self):
        out_path = filedialog.asksaveasfilename(
            title="raw_data 저장 경로 선택",
            defaultextension=".csv",
            initialfile="raw_data.csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if out_path:
            self.output_path_var.set(out_path)

    def analyze_files(self):
        if not self.selected_files:
            messagebox.showwarning("알림", "먼저 분석 파일을 선택하세요.")
            return

        try:
            if self.file_type_var.get() == "Keithley(Excel)":
                rows, cols = self._extract_rows_keithley(self.selected_files)
            else:
                rows, cols = self._extract_rows(self.selected_files)
        except Exception as exc:
            messagebox.showerror("분석 실패", str(exc))
            return

        self.analyzed_rows = rows
        self.available_columns = cols

        # 분석 시 column_list를 초기화
        default_cols = {"source_file", "source_line", "setup_title", "V1", "Abs_Id"}
        self.column_list = [(c, tk.BooleanVar(value=(c in default_cols))) for c in cols]
        self.selected_order = [c for c, v in self.column_list if v.get() and c != "index"]

        self._render_column_checkboxes()
        self._refresh_selected_order_listbox()
        self.update_preview()

        messagebox.showinfo("완료", f"분석 완료: 총 {len(self.analyzed_rows)} 행\n컬럼 수: {len(self.available_columns)}")

    def _render_column_checkboxes(self, default_columns=None):
        for child in self.chk_frame.winfo_children():
            child.destroy()

        # 전체 컬럼 체크박스만 렌더링 (순서 조정은 별도 리스트에서 수행)
        for col, var in self.column_list:

            row_frame = ttk.Frame(self.chk_frame)
            row_frame.pack(anchor="w", fill="x", pady=2)

            cb = ttk.Checkbutton(row_frame, text=col, variable=var, command=lambda c=col: self._on_column_toggle(c))
            cb.pack(side="left", fill="x", expand=True)

    def _on_column_toggle(self, col):
        var_map = {c: v for c, v in self.column_list}
        is_checked = bool(var_map[col].get())

        if is_checked:
            if col != "index" and col not in self.selected_order:
                self.selected_order.append(col)
        else:
            if col in self.selected_order:
                self.selected_order.remove(col)

        self._refresh_selected_order_listbox()
        self.update_preview()

    def _refresh_selected_order_listbox(self):
        self.order_listbox.delete(0, tk.END)
        for col in self.selected_order:
            self.order_listbox.insert(tk.END, col)

    def _move_selected_up(self):
        sel = self.order_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx <= 0:
            return

        self.selected_order[idx - 1], self.selected_order[idx] = self.selected_order[idx], self.selected_order[idx - 1]
        self._refresh_selected_order_listbox()
        self.order_listbox.selection_set(idx - 1)
        self.update_preview()

    def _move_selected_down(self):
        sel = self.order_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self.selected_order) - 1:
            return

        self.selected_order[idx + 1], self.selected_order[idx] = self.selected_order[idx], self.selected_order[idx + 1]
        self._refresh_selected_order_listbox()
        self.order_listbox.selection_set(idx + 1)
        self.update_preview()

    def _selected_columns(self):
        """현재 선택된 컬럼들을 순서대로 반환"""
        return ["index"] + list(self.selected_order)

    def update_preview(self):
        self.preview.delete("1.0", tk.END)

        if not self.analyzed_rows:
            self.preview.insert(tk.END, "분석된 데이터가 없습니다.\n")
            return

        selected = self._selected_columns()
        data = self._project_rows(self.analyzed_rows, selected)

        head_n = 10
        tail_n = 10
        head = data[:head_n]
        tail = data[-tail_n:] if len(data) > tail_n else []

        self.preview.insert(tk.END, "[HEAD]\n")
        self.preview.insert(tk.END, self._format_table(head, selected) + "\n")

        if tail:
            self.preview.insert(tk.END, "\n[TAIL]\n")
            self.preview.insert(tk.END, self._format_table(tail, selected))

    def export_raw_data(self):
        if not self.analyzed_rows:
            messagebox.showwarning("알림", "먼저 분석을 수행하세요.")
            return

        selected = self._selected_columns()
        if len(selected) <= 1:
            messagebox.showwarning("알림", "최소 1개 이상의 컬럼을 선택하세요.")
            return

        out_path = self.output_path_var.get().strip()
        if not out_path:
            out_path = filedialog.asksaveasfilename(
                title="저장할 raw_data 경로 선택",
                defaultextension=".csv",
                initialfile="raw_data.csv",
                filetypes=[("CSV files", "*.csv")],
            )
        if not out_path:
            return

        try:
            data = self._project_rows(self.analyzed_rows, selected)
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=selected)
                writer.writeheader()
                writer.writerows(data)
        except Exception as exc:
            messagebox.showerror("저장 실패", str(exc))
            return

        messagebox.showinfo("완료", f"추출 완료\n{out_path}")

    @staticmethod
    def _project_rows(rows, columns):
        projected = []
        for r in rows:
            out = {}
            for c in columns:
                out[c] = r.get(c, "")
            projected.append(out)
        return projected

    @staticmethod
    def _format_table(rows, columns):
        if not rows:
            return "(empty)"

        widths = {c: len(c) for c in columns}
        for row in rows:
            for c in columns:
                widths[c] = max(widths[c], len(str(row.get(c, ""))))

        header = " | ".join(c.ljust(widths[c]) for c in columns)
        sep = "-+-".join("-" * widths[c] for c in columns)
        lines = [header, sep]

        for row in rows:
            lines.append(" | ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))

        return "\n".join(lines)

    def _extract_rows(self, file_paths):
        # Agilent(CSV) parser
        all_blocks = []
        all_columns = {"index", "source_file", "source_line", "setup_title"}

        for file_path in sorted(file_paths):
            source_file = os.path.basename(file_path)
            lines = self._read_lines(file_path)
            blocks = self._parse_blocks(lines, source_file)
            all_blocks.extend(blocks)

            for b in blocks:
                for c in b["columns"]:
                    all_columns.add(c)

        # Key rule: lower setup_title block in file is measured earlier.
        all_blocks.sort(key=lambda b: (b["source_file"], -b["setup_title_line"], -b["data_name_line"]))

        out_rows = []
        row_index = 0

        for bi, block in enumerate(all_blocks):
            # Within same setup_title block: source_line increasing.
            data_rows = sorted(block["rows"], key=lambda r: r["source_line"])

            for r in data_rows:
                row_index += 1
                out = {"index": str(row_index)}
                out.update(r)
                out_rows.append(out)

            # Add blank separator row between setup_title blocks.
            if bi < len(all_blocks) - 1:
                out_rows.append({"index": ""})

        preferred = ["index", "source_file", "source_line", "setup_title", "V1", "Abs_Id"]
        others = sorted(c for c in all_columns if c not in preferred)
        ordered_columns = preferred + others

        return out_rows, ordered_columns

    def _extract_rows_keithley(self, file_paths):
        # Keithley(Excel) parser
        all_blocks = []
        all_columns = {"index", "source_file", "source_line", "setup_title"}

        for file_path in sorted(file_paths):
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in {".xls", ".xlsx"}:
                continue

            source_file = os.path.basename(file_path)
            if ext == ".xls":
                if xlrd is None:
                    raise RuntimeError(".xls 분석에는 xlrd 모듈이 필요합니다.")

                wb = xlrd.open_workbook(file_path)
                for sh in wb.sheets():
                    header_row_idx = self._find_header_row_xls(sh)
                    if header_row_idx is None:
                        continue

                    headers = [str(sh.cell_value(header_row_idx, c)).strip() for c in range(sh.ncols)]
                    if not any(headers) or (not self._is_keithley_data_header(headers)):
                        continue

                    rows = []
                    for r in range(header_row_idx + 1, sh.nrows):
                        vals = [sh.cell_value(r, c) for c in range(sh.ncols)]
                        if all(str(v).strip() == "" for v in vals):
                            continue

                        row = {
                            "source_file": source_file,
                            "source_line": str(r + 1),
                            "setup_title": sh.name,
                        }

                        for c, h in enumerate(headers):
                            col_name = h if h else f"Column_{c+1}"
                            row[col_name] = self._excel_cell_to_text(vals[c])
                            all_columns.add(col_name)

                        rows.append(row)

                    if rows:
                        all_blocks.append(
                            {
                                "source_file": source_file,
                                "setup_title": sh.name,
                                "setup_title_line": header_row_idx + 1,
                                "data_name_line": header_row_idx + 1,
                                "rows": rows,
                            }
                        )

            elif ext == ".xlsx":
                if openpyxl is None:
                    raise RuntimeError(".xlsx 분석에는 openpyxl 모듈이 필요합니다.")

                wb = openpyxl.load_workbook(file_path, data_only=True)
                for sh in wb.worksheets:
                    rows_raw = list(sh.iter_rows(values_only=True))
                    if not rows_raw:
                        continue

                    header_row_idx = self._find_header_row_xlsx(rows_raw)
                    if header_row_idx is None:
                        continue

                    headers = [str(v).strip() if v is not None else "" for v in rows_raw[header_row_idx]]
                    if not any(headers) or (not self._is_keithley_data_header(headers)):
                        continue

                    rows = []
                    for r in range(header_row_idx + 1, len(rows_raw)):
                        vals = list(rows_raw[r])
                        if all((v is None) or (str(v).strip() == "") for v in vals):
                            continue

                        row = {
                            "source_file": source_file,
                            "source_line": str(r + 1),
                            "setup_title": sh.title,
                        }

                        if len(vals) < len(headers):
                            vals += [None] * (len(headers) - len(vals))

                        for c, h in enumerate(headers):
                            col_name = h if h else f"Column_{c+1}"
                            row[col_name] = self._excel_cell_to_text(vals[c])
                            all_columns.add(col_name)

                        rows.append(row)

                    if rows:
                        all_blocks.append(
                            {
                                "source_file": source_file,
                                "setup_title": sh.title,
                                "setup_title_line": header_row_idx + 1,
                                "data_name_line": header_row_idx + 1,
                                "rows": rows,
                            }
                        )

        # Preserve workbook sheet order per file.
        all_blocks.sort(key=lambda b: (b["source_file"], b["setup_title_line"]))

        out_rows = []
        row_index = 0
        for bi, block in enumerate(all_blocks):
            for r in block["rows"]:
                row_index += 1
                out = {"index": str(row_index)}
                out.update(r)
                out_rows.append(out)

            if bi < len(all_blocks) - 1:
                out_rows.append({"index": ""})

        preferred = ["index", "source_file", "source_line", "setup_title", "V1", "Abs_Id", "Time", "DrainI", "DrainV", "GateI", "GateV"]
        others = sorted(c for c in all_columns if c not in preferred)
        ordered_columns = [c for c in preferred if c in all_columns] + others

        return out_rows, ordered_columns

    @staticmethod
    def _find_header_row_xls(sheet):
        # Pick the first row that looks like a header: mostly non-empty strings.
        for r in range(min(40, sheet.nrows)):
            vals = [sheet.cell_value(r, c) for c in range(sheet.ncols)]
            non_empty = [v for v in vals if str(v).strip() != ""]
            if len(non_empty) < 2:
                continue

            str_like = 0
            for v in non_empty:
                if isinstance(v, str):
                    str_like += 1
                elif isinstance(v, float):
                    # Header rows are usually not mostly numeric.
                    pass

            if str_like >= max(2, len(non_empty) // 2):
                return r
        return None

    @staticmethod
    def _find_header_row_xlsx(rows):
        for r in range(min(40, len(rows))):
            vals = list(rows[r])
            non_empty = [v for v in vals if (v is not None) and (str(v).strip() != "")]
            if len(non_empty) < 2:
                continue

            str_like = sum(1 for v in non_empty if isinstance(v, str))
            if str_like >= max(2, len(non_empty) // 2):
                return r
        return None

    @staticmethod
    def _is_keithley_data_header(headers):
        keys = {"time", "draini", "drainv", "gatei", "gatev", "id", "vd", "vg"}
        low = {str(h).strip().lower() for h in headers if str(h).strip() != ""}
        match_count = len(low & keys)
        return match_count >= 2

    @staticmethod
    def _excel_cell_to_text(value):
        if value is None:
            return ""
        if isinstance(value, float):
            return format(value, ".15g")
        return str(value)

    @staticmethod
    def _read_lines(file_path):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return [line.rstrip("\n").rstrip("\r") for line in f]

    @staticmethod
    def _parse_blocks(lines, source_file):
        blocks = []
        current_title = ""
        current_title_line = 0

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("SetupTitle,"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    current_title = parts[1]
                    current_title_line = i + 1
                i += 1
                continue

            if line.startswith("AnalysisSetup, Analysis.Setup.Title,"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    current_title = parts[2]
                    current_title_line = i + 1
                i += 1
                continue

            if line.startswith("DataName,"):
                header_parts = [p.strip() for p in line.split(",")]
                columns = header_parts[1:]

                rows = []
                j = i + 1
                while j < len(lines):
                    dl = lines[j].strip()
                    if not dl.startswith("DataValue,"):
                        break

                    value_parts = [p.strip() for p in dl.split(",")][1:]
                    if len(value_parts) < len(columns):
                        value_parts += [""] * (len(columns) - len(value_parts))
                    elif len(value_parts) > len(columns):
                        value_parts = value_parts[: len(columns)]

                    row = {
                        "source_file": source_file,
                        "source_line": str(j + 1),
                        "setup_title": current_title,
                    }
                    for idx, col in enumerate(columns):
                        row[col] = value_parts[idx]

                    rows.append(row)
                    j += 1

                blocks.append(
                    {
                        "source_file": source_file,
                        "setup_title": current_title,
                        "setup_title_line": current_title_line,
                        "data_name_line": i + 1,
                        "columns": columns,
                        "rows": rows,
                    }
                )

                i = j
                continue

            i += 1

        return blocks


def main():
    root = tk.Tk()
    app = RawDataExtractorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
