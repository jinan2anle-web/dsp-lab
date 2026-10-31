"""实验三的图形界面：选一条录音 → 看波形/端点/MFCC/各模板距离 → 出识别结果 → 播放。

启动：

    python -m dsp.app

界面只用标准库 tkinter 加 matplotlib 的 TkAgg 后端，播放用 winsound（Windows 自带），
不引入任何额外的 GUI 依赖。模板库第一次运行时现场建（约 20 s），之后读缓存。

署名固定在窗口底部：没有名字的界面在验收时不算数。
"""

from __future__ import annotations

import sys
import traceback

AUTHORS = "梁博强（A，实验二）· 师瑞颖（B，实验三）· 肖雨乐（C，工程与统稿）"


def _ensure_libs():
    from dsp.env import add_pylibs

    add_pylibs()


def _setup_fonts():
    """matplotlib 默认的 DejaVu Sans 没有汉字，必须换成系统里的中文字体，
    否则界面图里的中文全是方块（验收明确不许）。"""
    from matplotlib import font_manager, rcParams

    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
                 "SimSun", "DengXian"):
        if cand in names:
            rcParams["font.sans-serif"] = [cand]
            rcParams["font.family"] = "sans-serif"
            rcParams["axes.unicode_minus"] = False
            return cand
    rcParams["axes.unicode_minus"] = False
    return None


def build_app():
    import tkinter as tk
    from tkinter import messagebox, ttk

    import numpy as np
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    from dsp import config, endpoint, io_wav, mfcc, templates

    class App:
        def __init__(self, root):
            self.root = root
            root.title("数字信号处理 · 实验三：DTW 孤立字识别")
            root.geometry("1180x760")

            self.paths = config.wav_paths()
            self.table = None          # {label: [mfcc,...]}
            self.current = None        # 当前 wav 路径
            self.signal = None
            self.fs = config.FS
            self.feat = None

            # ---------- 左侧控制区 ----------
            foot = ttk.Label(root, text=AUTHORS, anchor="center", foreground="#444444")
            foot.pack(side=tk.BOTTOM, fill=tk.X, pady=3)

            left = ttk.Frame(root, padding=8)
            left.pack(side=tk.LEFT, fill=tk.Y)

            ttk.Label(left, text="录音（data/A，100 条）").pack(anchor="w")
            self.listbox = tk.Listbox(left, width=26, height=16, exportselection=False)
            for p in self.paths:
                self.listbox.insert(tk.END, p.name)
            self.listbox.pack(fill=tk.BOTH, expand=True)
            self.listbox.bind("<<ListboxSelect>>", lambda e: self.on_pick())

            btns = ttk.Frame(left)
            btns.pack(fill=tk.X, pady=6)
            ttk.Button(btns, text="识别", command=self.on_recognize).pack(fill=tk.X, pady=2)
            ttk.Button(btns, text="播放", command=self.on_play).pack(fill=tk.X, pady=2)
            ttk.Button(btns, text="重建模板库", command=self.on_build).pack(fill=tk.X, pady=2)

            ttk.Label(left, text="识别结果").pack(anchor="w", pady=(8, 0))
            self.result = tk.StringVar(value="（先选一条录音，再点「识别」）")
            ttk.Label(left, textvariable=self.result, wraplength=190,
                      justify="left", font=("Microsoft YaHei", 10)).pack(anchor="w")

            self.status = tk.StringVar(value="模板库：未加载")
            ttk.Label(left, textvariable=self.status, wraplength=190, foreground="#555577",
                      justify="left").pack(anchor="w", pady=(8, 0))

            # ---------- 右侧绘图区 ----------
            right = ttk.Frame(root)
            right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            self.fig = Figure(figsize=(8.6, 6.4), dpi=100, layout="constrained")
            self.ax_wave = self.fig.add_subplot(311)
            self.ax_mfcc = self.fig.add_subplot(312)
            self.ax_dist = self.fig.add_subplot(313)
            self._idle()

            self.canvas = FigureCanvasTkAgg(self.fig, master=right)
            # Windows 150% 缩放时 Tk 报告的 DPI 会让 matplotlib 按 1.5 倍缓冲渲染，
            # 而 Tk 控件按 1 倍显示，右半边图会被裁掉。强制 1:1。
            try:
                self.canvas.device_pixel_ratio = 1.0
            except Exception:
                pass
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            root.protocol("WM_DELETE_WINDOW", self.on_close)

        # ---------- 绘图辅助 ----------
        def _idle(self):
            for ax in (self.ax_wave, self.ax_mfcc, self.ax_dist):
                ax.clear()
                ax.grid(alpha=0.3)
            self.ax_wave.set_title("波形与端点检测", fontsize=9)
            self.ax_mfcc.set_title("MFCC $[T,13]$", fontsize=9)
            self.ax_dist.set_title("各类别最近模板距离", fontsize=9)

        def _redraw(self):
            # 版面用 constrained layout 引擎，窗口缩放时自动重排，不需要手动 tight_layout
            self.canvas.draw_idle()

        # ---------- 事件 ----------
        def on_pick(self):
            sel = self.listbox.curselection()
            if not sel:
                return
            self.current = self.paths[sel[0]]
            x, fs = io_wav.read_wav(self.current)
            self.signal, self.fs = x, fs
            self.feat = mfcc.extract(x, fs)
            seg = endpoint.detect(x, fs)

            ax = self.ax_wave
            ax.clear()
            t = np.arange(len(x)) / fs
            ax.plot(t, x, lw=0.5, color="#2E5C9A")
            if seg is not None:
                ax.axvspan(seg[0] / fs, seg[1] / fs, color="#FF6B6B", alpha=0.22,
                           label=f"端点段 {(seg[1]-seg[0])/fs:.3f} s")
                ax.legend(fontsize=7, loc="upper right")
            ax.set_title(f"{self.current.name}  波形与端点检测", fontsize=9)
            ax.set_xlabel("时间 / s", fontsize=8)
            ax.grid(alpha=0.3)

            ax = self.ax_mfcc
            ax.clear()
            im = ax.imshow(self.feat.T, aspect="auto", origin="lower", cmap="coolwarm")
            ax.set_title(f"MFCC $[T,13]$，T={self.feat.shape[0]}（含 $c_0$）", fontsize=9)
            ax.set_ylabel("阶 n", fontsize=8)
            ax.set_xlabel("帧号", fontsize=8)

            ax = self.ax_dist
            ax.clear()
            if self.table is not None:
                self._draw_distances(templates.distances(self.feat, self.table))
            ax.set_title("各类别最近模板距离", fontsize=9)
            ax.grid(alpha=0.3)

            self._redraw()

        def _draw_distances(self, d, best=None):
            ax = self.ax_dist
            labels = sorted(d)
            vals = [d[k] for k in labels]
            colors = ["#C0392B" if k == best else "#5B8FF9" for k in labels]
            ax.bar([str(k) for k in labels], vals, color=colors)
            ax.set_ylabel("DTW 距离", fontsize=8)

        def on_recognize(self):
            if self.current is None:
                messagebox.showinfo("提示", "先在左边选一条录音")
                return
            if self.table is None:
                self.on_build()
                if self.table is None:
                    return
            try:
                d = templates.distances(self.feat, self.table)
                best = min(d, key=d.get)
                true = config.label_of(self.current)
                ok = "✓ 正确" if best == true else "✗ 错误"
                self.result.set(
                    f"预测：{best}\n真实：{true}　{ok}\n"
                    f"最近距离 {d[best]:.4f}\n次近 "
                    + ", ".join(f"{k}:{d[k]:.4f}" for k in sorted(d, key=d.get)[1:4])
                )
                ax = self.ax_dist
                ax.clear()
                self._draw_distances(d, best)
                ax.set_title(f"各类别最近模板距离（预测 {best}，真实 {true}）", fontsize=9)
                ax.grid(alpha=0.3)
                self._redraw()
            except Exception:
                messagebox.showerror("识别失败", traceback.format_exc())

        def on_build(self):
            try:
                self.status.set("模板库：正在计算 50 条训练录音…")
                self.root.update_idletasks()
                self.table = templates.build("train", normalize="none")
                self.status.set(f"模板库：{len(self.table)} 类 / "
                                f"{sum(len(v) for v in self.table.values())} 条模板")
                self.on_pick()
            except Exception:
                self.status.set("模板库：建立失败")
                messagebox.showerror("建模板失败", traceback.format_exc())

        def on_play(self):
            if self.current is None:
                return
            import winsound

            winsound.PlaySound(str(self.current), winsound.SND_FILENAME | winsound.SND_ASYNC)

        def on_close(self):
            try:
                import winsound

                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
            self.root.destroy()

    return App


def main(argv=None):
    _ensure_libs()
    _setup_fonts()
    import tkinter as tk

    App = build_app()
    root = tk.Tk()
    try:
        root.iconbitmap(default="")
    except Exception:
        pass
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
