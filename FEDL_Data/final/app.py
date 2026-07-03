"""FEDL Tool – 통합 앱 (데이터 전처리 + Plotting)"""

import tkinter as tk
from tkinter import messagebox, ttk

from preprocessing_tab import PreprocessingTab
from plotting_tab import PlottingTab


class FEDLApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("FEDL Tool")
        self.root.geometry("1500x940")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=(6, 0))

        # 탭 1: 데이터 전처리
        prep_frame = ttk.Frame(self.notebook)
        self.notebook.add(prep_frame, text="  데이터 전처리  ")
        self.prep_tab = PreprocessingTab(prep_frame, on_export_callback=self._on_export_done)

        # 탭 2: Plotting
        plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(plot_frame, text="  Plotting  ")
        self.plot_tab = PlottingTab(plot_frame)

        # 공통 footer
        footer = ttk.Frame(root, padding=(10, 4))
        footer.pack(fill="x")
        wm = ttk.Frame(footer)
        wm.pack(side="left")
        ttk.Label(wm, text="ASNL_HYUNHO.JANG", foreground="#666666").pack(anchor="w")
        ttk.Label(wm, text="무단 배포 금지", foreground="#666666").pack(anchor="w")
        ttk.Label(wm, text="문의: hyunho0214@naver.com", foreground="#666666").pack(anchor="w")

    def _on_export_done(self, csv_path: str):
        """전처리 탭에서 추출 완료 시 호출"""
        answer = messagebox.askyesno(
            "Plotting 연계",
            f"추출 완료:\n{csv_path}\n\nPlotting 탭으로 이동하여 데이터를 로드할까요?",
        )
        if answer:
            self.plot_tab.set_raw_data_and_load(csv_path)
            self.notebook.select(1)


def main():
    root = tk.Tk()
    FEDLApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
