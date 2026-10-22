# -*- coding: utf-8 -*-
"""
按《周报模板.pptx》生成第一周周报。

用法：
    python tools/build_weekly_report.py --date "04/26～05/02" --reporter "~~~，~~~"

不传 --date / --reporter 时保留模板原有文字（标题日期、报告人占位）。
输出：task1_w1/第一周周报.pptx
"""
import argparse
import os
import shutil
import sys

from pptx import Presentation
from pptx.util import Pt

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # task1_w1/接口冻结
BASE = os.path.dirname(HERE)                                          # task1_w1
TPL = os.path.join(BASE, '周报模板.pptx')
OUT = os.path.join(BASE, '第一周周报.pptx')

# ---------------------------------------------------------------- 周报正文
# 仅保留结论与待办；未提到的责任人不臆测，用"待确认"标注
CELLS = {
    # 项目名称（第 1 列，竖排）
    0: ['实验1 基于时域\n分析技术的\n语音识别'],

    # 当前项目所处阶段以及目标
    1: ['第1阶段：准备与\n接口冻结（第1—2天）',
        '目标：完成录音入库、\nWAV 字段解析，\n冻结组内协作接口。'],

    # 本周工作计划
    2: ['1. 录制"0"~"9"\n各10组以上 wav',
        '2. 解析 WAV 格式\n并读取关键字段',
        '3. 完成接口约定\n与联调（D2）'],

    # 本周工作情况
    3: ['【录音与入库】完成数字0~9各10组共100条录音，'
        '命名 0_01.wav~9_10.wav。\n'
        '【WAV解析】实现 read_wav() 逐字段读取，'
        '输出采样率/通道数/位深等关键字段对照表。\n'
        '【质量审核】100条全部 16 kHz/单声道/16 bit/'
        '1.500 s，无削波、无直流偏置。\n'
        '【接口冻结】冻结7个函数签名与输出结构，'
        '覆盖数据/预处理/特征/端点检测/DTW 五层；'
        '产出《接口冻结说明》v1.0。\n'
        '【算法跑通】分帧加窗、短时能量/平均幅度/'
        '过零率、双门限端点检测、时域特征归一化、'
        'DTW距离与最近邻分类器均跑通。\n'
        '【自检】建立28项断言的接口自检脚本，'
        '当前 28/28 全部通过。'],

    # 项目中遇到的问题
    4: ['1. read_wav 我原先按 (x, fs, info) 写接口，'
        '实际代码是四元组，调用方曾按字典取字段。\n'
        '→ 已定：冻结清单改为四元组，'
        '以现有代码为准。\n'
        '2. 录音文件名只有数字+序号（0_01.wav），'
        '未含说话人标识。\n'
        '→ 已定：以实际文件为准，'
        '改为按说话人建文件夹区分。\n'
        '3. wav_parse.py 读奇数长度子块'
        '（LIST/fact）会错位；公共模块已修正，'
        '待回同步（本批无此块，未暴露）。\n'
        '4. 冻结清单是我单方整理的，'
        '问题1、2 的口径尚未与组内当面对齐，'
        '需在下周联调前确认。'],

    # 下周工作计划
    5: ['1. 组内对齐接口冻结\n清单并确认签字。',
        '2. 整库重跑接口自检\n（出口判据复查）。',
        '3. 实现识别主流程：\n特征提取—模板库—\n最近邻分类。',
        '4. 在真实录音上做\n窗型/帧长对比实验。',
        '5. 把 read_wav 的奇数\n块填充修复回同步\n到 wav_parse.py。'],
}

TITLE_DATE = None      # 由命令行 --date 覆盖
REPORTER = None        # 由命令行 --reporter 覆盖


def set_east_asian_font(run, typeface='宋体'):
    """设置 a:ea（中文字体）。a:ea 必须排在 a:latin 之后，否则 rPr 子元素顺序非法。"""
    from pptx.oxml.ns import qn
    rPr = run._r.get_or_add_rPr()
    for tag in ('a:sym', 'a:cs', 'a:ea'):
        for el in rPr.findall(qn(tag)):
            rPr.remove(el)
    ea = rPr.makeelement(qn('a:ea'), {'typeface': typeface})
    latin = rPr.find(qn('a:latin'))
    if latin is not None:
        latin.addnext(ea)
    else:
        rPr.append(ea)


def fill_cell(cell, lines, size=11, line_spacing=1.0):
    tf = cell.text_frame
    tf.word_wrap = True
    tf.clear()
    for i, text in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.name = '宋体'
        set_east_asian_font(run, '宋体')


def set_title_date(slide, date_text):
    """标题文本框形如 '第一周周报 04/26～ 05/02'
    date_text 为 None 时保留模板原文；为空串时清掉日期只留标题；否则整体重写。"""
    for sh in slide.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip().startswith('第一周周报'):
            if date_text is None:
                return True
            tf = sh.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = '第一周周报  ' + date_text if date_text else '第一周周报'
            run.font.size = Pt(13)
            run.font.name = '+mj-lt'
            return True
    return False


def set_reporter(slide, reporter_text):
    """报告人文本框形如 '报告人：~~~，~~~'
    reporter_text 为 None 时保留模板原文；为空串时只留 "报告人："；否则替换其后的内容。"""
    for sh in slide.shapes:
        if sh.has_text_frame and sh.text_frame.text.startswith('报告人'):
            if reporter_text is None:
                return True
            tf = sh.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            for seg in ['报告人：', reporter_text]:
                if not seg:
                    continue
                run = p.add_run()
                run.text = seg
                run.font.size = Pt(14)
                run.font.name = 'Arial'
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=None,
                    help='周报日期，如 04/26～05/02；传空串 "" 则清空日期留待手填')
    ap.add_argument('--reporter', default=None,
                    help='报告人，如 张三，李四；传空串 "" 则只留"报告人："待手填')
    args = ap.parse_args()

    shutil.copyfile(TPL, OUT)
    prs = Presentation(OUT)
    slide = prs.slides[0]

    table = None
    for sh in slide.shapes:
        if sh.has_table and len(sh.table.rows) == 2 and len(sh.table.columns) == 6:
            table = sh.table
            break
    if table is None:
        print('未找到周报表格，模板结构可能已变更。')
        return 1

    for col, lines in CELLS.items():
        fill_cell(table.cell(1, col), lines,
                  size=11 if col != 0 else 12,
                  line_spacing=0.95 if col == 0 else 1.0)

    if args.date is not None:
        set_title_date(slide, args.date)
    if args.reporter is not None:
        set_reporter(slide, args.reporter)

    prs.save(OUT)
    print('已生成：%s' % OUT)
    print('日期：%s' % (args.date if args.date else
                      ('已清空，待手填' if args.date == '' else '保留模板原文 04/26～ 05/02')))
    print('报告人：%s' % (args.reporter if args.reporter else
                       ('已清空，待手填' if args.reporter == '' else '保留模板占位 ~~~，~~~')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
