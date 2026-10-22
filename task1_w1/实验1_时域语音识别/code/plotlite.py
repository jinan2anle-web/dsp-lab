# -*- coding: utf-8 -*-
"""轻量绘图工具：仅依赖 Pillow + numpy，用于生成实验报告插图（支持中文）"""
import os
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]
_fc = {}

W_BG = (255, 255, 255)
C_AXIS = (70, 70, 70)
C_GRID = (225, 228, 232)
PALETTE = [(31, 79, 156), (214, 39, 40), (44, 160, 44), (255, 127, 14),
           (148, 103, 189), (140, 86, 75), (0, 150, 150), (200, 60, 160)]


def font(size):
    if size not in _fc:
        for p in FONT_CANDIDATES:
            if os.path.exists(p):
                try:
                    _fc[size] = ImageFont.truetype(p, size)
                    break
                except Exception:
                    continue
        else:
            _fc[size] = ImageFont.load_default()
    return _fc[size]


SS = 2  # 超采样倍率


def nice_ceil(v):
    """把坐标轴上界取整到"好看"的刻度（1/2/2.5/5 × 10^k）"""
    import math
    if v <= 0:
        return 1.0
    e = math.floor(math.log10(v))
    base = 10.0 ** e
    m = v / base
    for s in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if m <= s + 1e-12:
            return s * base
    return 10.0 * base


class Panel:
    """一个坐标轴面板：负责数据坐标 -> 图像坐标的映射与绘制"""

    def __init__(self, draw, box, xlim, ylim, xlabel='', ylabel='', title='',
                 xticks=None, yticks=None, xfmt='{:.2f}', yfmt='{:.2f}',
                 ymax_ticks=5, xmax_ticks=8, grid=True, fontscale=1.0):
        self.d, self.x0, self.y0, self.x1, self.y1 = draw, *box
        self.img = getattr(draw, '_image', None)
        self.xlim, self.ylim = xlim, ylim
        self.fs = fontscale
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.grid = grid
        self.xticks = xticks
        self.yticks = yticks
        self.xfmt, self.yfmt = xfmt, yfmt
        self.ymax_ticks, self.xmax_ticks = ymax_ticks, xmax_ticks
        self.legend = []
        self.prepare()

    def prepare(self):
        d = self.d
        d.rectangle([self.x0, self.y0, self.x1, self.y1], fill=W_BG,
                    outline=C_AXIS, width=int(1.2 * SS))

    def mx(self, x):
        a, b = self.xlim
        return self.x0 + (x - a) / (b - a + 1e-12) * (self.x1 - self.x0)

    def my(self, y):
        a, b = self.ylim
        return self.y1 - (y - a) / (b - a + 1e-12) * (self.y1 - self.y0)

    # ---------------- 网格与刻度 ----------------
    def draw_axes(self):
        import math
        d = self.d
        xt = self.xticks if self.xticks is not None else self._auto(self.xlim, self.xmax_ticks)
        yt = self.yticks if self.yticks is not None else self._auto(self.ylim, self.ymax_ticks)
        xf = self.xfmt if self.xticks is None else '{}'
        yf = self.yfmt
        # 按刻度间隔自动决定小数位，避免出现 0.19000000000000003 之类的浮点噪声
        if self.ylim[1] != self.ylim[0]:
            step = (self.ylim[1] - self.ylim[0]) / max(1, len(yt) - 1)
            dec = max(0, int(math.ceil(-math.log10(step + 1e-12))) + 1)
            yf = '{:.' + str(dec) + 'f}'
        f = font(int(15 * SS * self.fs))
        if self.grid:
            for x in xt:
                d.line([self.mx(x), self.y0, self.mx(x), self.y1], fill=C_GRID, width=SS)
            for y in yt:
                d.line([self.x0, self.my(y), self.x1, self.my(y)], fill=C_GRID, width=SS)
        for x in xt:
            d.line([self.mx(x), self.y1, self.mx(x), self.y1 + 5 * SS], fill=C_AXIS, width=SS)
            d.text((self.mx(x), self.y1 + 8 * SS), xf.format(x), font=f,
                   fill=C_AXIS, anchor='ma')
        for y in yt:
            d.line([self.x0 - 5 * SS, self.my(y), self.x0, self.my(y)], fill=C_AXIS, width=SS)
            d.text((self.x0 - 8 * SS, self.my(y)), yf.format(y), font=f,
                   fill=C_AXIS, anchor='rm')
        if self.xlabel:
            d.text(((self.x0 + self.x1) / 2, self.y1 + 40 * SS * self.fs), self.xlabel,
                   font=font(int(16 * SS * self.fs)), fill=C_AXIS, anchor='ma')
        if self.ylabel:
            self._draw_rotated_label(self.ylabel, self.x0 - 78 * SS, (self.y0 + self.y1) / 2)
        if self.title:
            d.text(((self.x0 + self.x1) / 2, self.y0 - 34 * SS * self.fs), self.title,
                   font=font(int(18 * SS * self.fs)), fill=(20, 20, 20), anchor='ma')

    def _draw_rotated_label(self, s, cx, cy):
        """绘制竖排 y 轴标签，按文字实际大小精确定位，避免被裁剪或压住刻度"""
        f = font(int(16 * SS * self.fs))
        box = self.d.textbbox((0, 0), s, font=f)
        w, h = box[2] - box[0] + 8 * SS, box[3] - box[1] + 8 * SS
        tmp = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(tmp).text((w / 2, h / 2), s, font=f, fill=C_AXIS, anchor='mm')
        tmp = tmp.rotate(90, expand=True)
        self.img.paste(tmp, (max(2, int(cx - tmp.width / 2)), int(cy - tmp.height / 2)), tmp)

    @staticmethod
    def _auto(lim, n):
        import numpy as np
        a, b = lim
        step = (b - a) / n
        ticks = []
        v = a
        while v <= b + 1e-9:
            ticks.append(round(v, 6))
            v += step
        return ticks

    # ---------------- 基本图元 ----------------
    def line(self, xs, ys, color=None, lw=1.4, label=None, alpha=255):
        import numpy as np
        color = color or PALETTE[len(self.legend) % len(PALETTE)]
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        if len(xs) == 0:
            return
        pts = [(self.mx(px), self.my(py)) for px, py in zip(xs, ys)]
        self.d.line(pts, fill=color + (alpha,) if len(color) == 3 else color,
                    width=max(1, int(lw * SS)), joint='curve')
        if label:
            self.legend.append((label, color))

    def vline(self, x, color=(214, 39, 40), lw=1.0, dash=True):
        if dash:
            y = self.y0
            while y < self.y1:
                self.d.line([self.mx(x), y, self.mx(x), min(y + 8 * SS, self.y1)],
                            fill=color, width=max(1, int(lw * SS)))
                y += 16 * SS
        else:
            self.d.line([self.mx(x), self.y0, self.mx(x), self.y1], fill=color, width=int(lw * SS))

    def hline(self, y, color=(214, 39, 40), lw=1.0, dash=True):
        if dash:
            x = self.x0
            while x < self.x1:
                self.d.line([x, self.my(y), min(x + 8 * SS, self.x1), self.my(y)],
                            fill=color, width=max(1, int(lw * SS)))
                x += 16 * SS
        else:
            self.d.line([self.x0, self.my(y), self.x1, self.my(y)], fill=color, width=int(lw * SS))

    def text_at(self, x, y, s, size=15, color=(30, 30, 30), anchor='mm'):
        self.d.text((self.mx(x), self.my(y)), s, font=font(int(size * SS * self.fs)),
                    fill=color, anchor=anchor)

    def draw_legend(self, loc='upper right', dy=0):
        if not self.legend:
            return
        f = font(int(15 * SS * self.fs))
        items = self.legend
        wmax = max(self.d.textlength(t, font=f) for t, _ in items)
        hh = 24 * SS * self.fs
        bw, bh = wmax + 52 * SS, hh * len(items) + 12 * SS
        if loc == 'upper right':
            bx, by = self.x1 - bw - 8 * SS, self.y0 + 8 * SS + int(dy * SS)
        elif loc == 'lower left':
            bx, by = self.x0 + 8 * SS, self.y1 - bh - 8 * SS - int(dy * SS)
        else:
            bx, by = self.x0 + 8 * SS, self.y0 + 8 * SS + int(dy * SS)
        self.d.rectangle([bx, by, bx + bw, by + bh], fill=(255, 255, 255, 235),
                         outline=(200, 200, 200), width=SS)
        for i, (t, c) in enumerate(items):
            yy = by + 10 * SS + i * hh
            self.d.line([bx + 10 * SS, yy + 6 * SS, bx + 40 * SS, yy + 6 * SS], fill=c, width=3 * SS)
            self.d.text((bx + 46 * SS, yy + 6 * SS), t, font=f, fill=(40, 40, 40), anchor='lm')


class Figure:
    def __init__(self, w, h):
        self.W, self.H = int(w * SS), int(h * SS)
        self.img = Image.new('RGBA', (self.W, self.H), (255, 255, 255, 255))
        self.d = ImageDraw.Draw(self.img)
        self.d._image = self.img

    def panel(self, box, spec, ylim, **kw):
        """box=(x0,y0,x1,y1) 逻辑坐标; spec=(xmin,xmax); ylim=(ymin,ymax)"""
        p = Panel(self.d, [b * SS for b in box], spec, ylim, **kw)
        return p

    def save(self, path):
        out = self.img.convert('RGB').resize((self.W // SS, self.H // SS), Image.LANCZOS)
        out.save(path)
        return path


def confusion_figure(path, labels, conf, title, cmap='Blues'):
    """绘制混淆矩阵（表格 + 色块）"""
    import numpy as np
    n = len(labels)
    cell = 92
    left, top = 150, 120
    W = left + cell * n + 150
    H = top + cell * n + 130
    fig = Figure(W, H)
    d = fig.d
    conf = np.asarray(conf)
    mx = max(conf.max(), 1)
    for i in range(n):
        for j in range(n):
            v = conf[i, j] / mx
            col = (int(247 - 190 * v), int(251 - 160 * v), int(255 - 90 * v))
            x0, y0 = left + j * cell, top + i * cell
            d.rectangle([x0 * SS, y0 * SS, (x0 + cell) * SS, (y0 + cell) * SS],
                        fill=col, outline=(255, 255, 255), width=2 * SS)
            d.text(((x0 + cell / 2) * SS, (y0 + cell / 2) * SS), str(conf[i, j]),
                   font=font(int(26 * SS)), fill=(255, 255, 255) if v > 0.6 else (30, 30, 30),
                   anchor='mm')
    for j, l in enumerate(labels):
        d.text(((left + j * cell + cell / 2) * SS, (top - 22) * SS), f'识别为"{l}"',
               font=font(int(15 * SS)), fill=(60, 60, 60), anchor='mm')
    for i, l in enumerate(labels):
        d.text(((left - 18) * SS, (top + i * cell + cell / 2) * SS), f'真实"{l}"',
               font=font(int(15 * SS)), fill=(60, 60, 60), anchor='rm')
    d.text((W / 2 * SS, 46 * SS), title, font=font(int(19 * SS)), fill=(20, 20, 20), anchor='mm')
    fig.save(path)
    return path


def grouped_bar_figure(path, groups, series, title, ylabel='', ylim=None, yfmt='{:.2f}'):
    """groups: x 轴标签; series: {name: [values]}"""
    import numpy as np
    fig = Figure(880, 420)
    vals = np.array(list(series.values()), dtype=float)
    ymax = ylim[1] if ylim else float(vals.max()) * 1.18
    ymin = ylim[0] if ylim else 0.0
    p = fig.panel((118, 70, 830, 340), (0, len(groups)), (ymin, ymax),
                  xlabel='', ylabel=ylabel, title=title,
                  xticks=[], grid=True)
    p.draw_axes()
    f = font(int(15 * SS))
    for i, g in enumerate(groups):
        p.d.text((p.mx(i + 0.5), p.y1 + 30 * SS), g, font=f, fill=C_AXIS, anchor='ma')
    n = len(series)
    bw = 0.72 / n
    for k, (name, ys) in enumerate(series.items()):
        c = PALETTE[k % len(PALETTE)]
        for i, v in enumerate(ys):
            x0 = p.mx(i + 0.14 + k * bw)
            x1 = p.mx(i + 0.14 + (k + 1) * bw)
            p.d.rectangle([x0, p.my(v), x1, p.my(ymin)], fill=c, outline=(255, 255, 255), width=SS)
            p.d.text(((x0 + x1) / 2, p.my(v) - 12 * SS), yfmt.format(v), font=font(int(12 * SS)),
                     fill=(40, 40, 40), anchor='mm')
        p.legend.append((name, c))
    p.draw_legend('upper left', dy=16)
    fig.save(path)
    return path
