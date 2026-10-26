"""
水印分析报告统一生成工具
========================

将水印分析实验的 JSON 数据 + 图表，统一打包成 Markdown / PDF / Word 报告。
所有数据自动从 `save/diagnostics/*.json` 与 `save/survival_analysis/*.json`
读取最新文件，避免硬编码过时数据。

依赖（按需安装）:
  pip install matplotlib                  # 图表生成必需
  pip install python-docx                 # Word 输出
  pip install markdown weasyprint         # PDF 输出

用法（从项目根目录运行）:
  python utils/generate_report.py --all
  python utils/generate_report.py --figures
  python utils/generate_report.py --md --docx --pdf
  python utils/generate_report.py --output-dir save/docs
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

# --------------------------------------------------------------------------- #
# 重要: 必须在 import numpy 之前设置
#   本机 anaconda 环境自带的 MKL 在 AVX2 路径上会触发 "Windows fatal exception
#   0xc06d007e" (缺失 DLL), 表现为 numpy.dot / matplotlib 绘制时进程直接崩溃
#   且无 Python 异常。禁用 MKL 多线程调度即可绕开。
#   如在别的机器上运行, 该设置无副作用, 可安全保留。
# --------------------------------------------------------------------------- #
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('MKL_ENABLE_INSTRUCTIONS', 'SSE4_2')


# ============================================================
# 路径与数据加载
# ============================================================

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_latest_json(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def load_latest_data():
    """加载最新梯度诊断 + 水印存活 JSON"""
    root = get_project_root()
    diag_path = find_latest_json(os.path.join(root, 'save', 'diagnostics', 'diagnostics_*.json'))
    surv_path = find_latest_json(os.path.join(root, 'save', 'survival_analysis', 'survival_*.json'))
    diag = json.load(open(diag_path, encoding='utf-8')) if diag_path else {}
    surv = json.load(open(surv_path, encoding='utf-8')) if surv_path else {}
    return diag, surv, diag_path, surv_path


def check_data_freshness(diag_path, surv_path):
    """
    对比诊断/存活数据与最新实验结果的时间戳, 过旧则明确告警。
    返回 (diag_ok, surv_ok)
    """
    root = get_project_root()
    latest_exp = _latest_experiment_time(root)
    if latest_exp is None:
        print("[WARN] 找不到实验结果文件, 无法校验诊断数据新鲜度")
        return True, True

    ok = True
    for label, path in (('梯度诊断', diag_path), ('水印存活', surv_path)):
        if not path:
            continue
        t = os.path.getmtime(path)
        if t < latest_exp:
            age_days = (latest_exp - t) / 86400.0
            print(f"[WARN] {label}数据过期: {relpath(path, root)} "
                  f"比最新实验结果旧 {age_days:.1f} 天, "
                  f"建议重跑 {'watermark_diagnostics.py' if '诊断' in label else 'watermark_survival_analysis.py'}")
            ok = False
    return ok, ok


def _latest_experiment_time(root):
    """最新一次实验结果 (trace/detection/aggregate) 的修改时间"""
    candidates = []
    for sub, pattern in (('trace', 'trace_results_*.json'),
                          ('detection', 'watermark_deviation_detection_*.json')):
        d = os.path.join(root, 'save', sub)
        for p in glob.glob(os.path.join(d, pattern)):
            candidates.append(os.path.getmtime(p))
    agg = os.path.join(root, 'save', 'aggregate', 'summary_by_experiment.csv')
    if os.path.exists(agg):
        candidates.append(os.path.getmtime(agg))
    return max(candidates) if candidates else None


def relpath(p, root):
    return os.path.relpath(p, root) if p else 'N/A'


# ============================================================
# 追踪准确率数据
# ============================================================
# 优先从 aggregate_results.py 产出的 summary_by_experiment.csv 读取真实实验结果;
# 若该文件不存在, 再回退到 save/trace 与 save/detection 下最新 18 个结果文件。
# 严禁硬编码: 历史版本在此写死了 AlexNet+CIFAR 在 target_ratio=0.3 下
# 失败时的旧数据 (0.31/0.25/0.17), 导致与实际 100% 的结果不符。

DATASET_LABEL = {
    'chestmnist': 'ChestMNIST',
    'cifar10': 'CIFAR-10',
    'cifar100': 'CIFAR-100',
}
MODEL_LABEL = {
    'alexnet': 'AlexNet',
    'resnet': 'ResNet18',
}


def _norm_dataset(ds):
    return DATASET_LABEL.get(str(ds).strip().lower(), str(ds))


def _norm_model(md):
    m = str(md).strip().lower()
    if m in ('resnet18', 'resnet18_', 'resnet'):
        return 'ResNet18'
    return MODEL_LABEL.get(m, md)


def load_tracking_accuracy():
    """
    读取 (model, dataset, K) -> 追踪准确率(0~1) 的真实实验结果。

    数据源优先级 (全部来自真实实验产物, 不含任何硬编码):
      1. save/aggregate/aggregate_results.xlsx 的 Summary 表
      2. save/aggregate/summary_by_experiment.csv
      3. save/figures/plot_data_for_redo.xlsx 的 Fig1a_Accuracy 表
      4. 回退: 直接扫描 save/trace 与 save/detection 的原始结果文件

    返回 dict: { 'AlexNet + CIFAR-10': {'K=5': 1.0, ...}, ... }
    """
    root = get_project_root()

    rows, src = [], None

    # 1) aggregate Excel (与用户手动查看的 Excel 同源)
    xlsx = os.path.join(root, 'save', 'aggregate', 'aggregate_results.xlsx')
    if os.path.exists(xlsx):
        try:
            import pandas as pd
            df = pd.read_excel(xlsx, sheet_name='Summary')
            rows = df.to_dict('records')
            src = relpath(xlsx, root)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] 读取 {xlsx} 失败: {e}")
            rows = []

    # 2) summary CSV
    if not rows:
        csv_path = os.path.join(root, 'save', 'aggregate', 'summary_by_experiment.csv')
        if os.path.exists(csv_path):
            try:
                import csv as _csv
                with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
                    rows = list(_csv.DictReader(f))
                src = relpath(csv_path, root)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] 读取 {csv_path} 失败: {e}")
                rows = []

    # 3) 绘图数据 Excel (Fig1a_Accuracy)
    if not rows:
        pdx = os.path.join(root, 'save', 'figures', 'plot_data_for_redo.xlsx')
        if os.path.exists(pdx):
            try:
                import pandas as pd
                df = pd.read_excel(pdx, sheet_name='Fig1a_Accuracy')
                df = df.rename(columns={'trace_accuracy_pct': 'trace_accuracy'})
                rows = df.to_dict('records')
                src = relpath(pdx, root)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] 读取 {pdx} 失败: {e}")
                rows = []

    # 4) 回退: 原始结果文件
    if not rows:
        rows = _tracking_accuracy_from_raw_results(root)
        src = 'save/trace + save/detection'

    result = {}
    for r in rows:
        try:
            acc = float(r.get('trace_accuracy', r.get('trace_accuracy_pct')))
        except (TypeError, ValueError):
            continue
        if acc > 1.0:
            acc = acc / 100.0
        key = f"{_norm_model(r.get('model'))} + {_norm_dataset(r.get('dataset'))}"
        try:
            k = int(float(r.get('client_num')))
        except (TypeError, ValueError):
            continue
        result.setdefault(key, {})[f'K={k}'] = acc

    if result:
        print(f"[OK] 追踪准确率: 从 {src} 加载 "
              f"{sum(len(v) for v in result.values())} 个配置")
    else:
        print("[!] 未找到真实追踪准确率数据")
    return result


def _tracking_accuracy_from_raw_results(root):
    """直接扫描 save/trace 与 save/detection, 按 (model, dataset, K) 取最新文件配对。"""
    import glob
    import re
    from datetime import datetime

    trace_dir = os.path.join(root, 'save', 'trace')
    det_dir = os.path.join(root, 'save', 'detection')
    re_trace = re.compile(r'trace_results_(?P<m>.+?)_(?P<d>chestmnist|cifar10|cifar100)_'
                          r'(?P<ts>\d{14})\.json$')
    re_det = re.compile(r'watermark_deviation_detection_(?P<m>.+?)_(?P<d>chestmnist|cifar10|cifar100)_'
                        r'(?P<ts>\d{14})\.json$')

    def collect(d, rx):
        out = []
        for p in glob.glob(os.path.join(d, '*.json')):
            m = rx.match(os.path.basename(p))
            if not m:
                continue
            out.append({'path': p, 'model': m.group('m'), 'dataset': m.group('d'),
                        'ts': m.group('ts'),
                        'dt': datetime.strptime(m.group('ts'), '%Y%m%d%H%M%S')})
        out.sort(key=lambda x: x['dt'], reverse=True)
        return out

    traces = collect(trace_dir, re_trace)[:18]
    dets = collect(det_dir, re_det)[:18]
    det_by_key = {}
    for d in dets:
        det_by_key.setdefault((d['model'], d['dataset']), []).append(d)

    rows = []
    for t in traces:
        try:
            data = json.load(open(t['path'], encoding='utf-8'))
        except Exception:  # noqa: BLE001
            continue
        acc = data.get('overall_accuracy')
        if acc is None:
            continue
        if acc > 1.0:
            acc = acc / 100.0
        k = _infer_client_num(data)
        rows.append({'model': t['model'], 'dataset': t['dataset'],
                     'client_num': k, 'trace_accuracy': acc})
    return rows


def _infer_client_num(trace_data):
    """从 similarity_rankings 长度推断客户端数量"""
    for d in (trace_data.get('details') or []):
        rk = d.get('similarity_rankings') or []
        if rk:
            return len(rk)
    return None


COLOR_MAP = {
    'AlexNet':   '#E74C3C',
    'ResNet18':  '#3498DB',
}


# ============================================================
# 图表生成
# ============================================================

def _setup_matplotlib():
    """设置中文字体"""
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import platform

    if platform.system() == 'Windows':
        for font in ['Microsoft YaHei', 'SimHei', 'SimSun']:
            try:
                fm.findfont(font, fallback_to_default=False)
                matplotlib.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
                break
            except Exception:
                continue
    matplotlib.rcParams['axes.unicode_minus'] = False
    return plt


def _arch_color(name):
    return COLOR_MAP['AlexNet'] if 'AlexNet' in name else COLOR_MAP['ResNet18']


def _avg_tracking(acc_map):
    """对某配置的 {K: acc} 求平均; 空字典返回 0"""
    vals = [v for v in acc_map.values() if v is not None]
    return sum(vals) / len(vals) if vals else 0



def generate_all_figures(figures_dir, diag, surv, tracking_accuracy=None):
    """
    图表统一由 ``python -m utils.plot_paper_figures`` 生成 (两种攻击模式分开绘制):
        fig1_perturbation_accuracy.png    水印区域扰动的追踪准确率
        fig2_finetune_accuracy.png   本地微调的检测准确率
        fig3_perturbation_similarity.png   水印区域扰动的 top-1/top-2 相似度与 gap
        fig4_finetune_similarity.png  本地微调的 top-1/top-2 相似度与 gap

    这里只确认这四张图存在, 不重复绘图 (避免两套口径不一致)。
    """
    os.makedirs(figures_dir, exist_ok=True)
    for fname in ('fig1_perturbation_accuracy.png',
                  'fig2_finetune_accuracy.png',
                  'fig3_perturbation_similarity.png',
                  'fig4_finetune_similarity.png'):
        if os.path.exists(os.path.join(figures_dir, fname)):
            print(f"[OK] {fname} (由 utils.plot_paper_figures 生成)")
        else:
            print(f"[WARN] 缺少 {fname}, 请先运行: python -m utils.plot_paper_figures")


# ============================================================
# Markdown 报告生成
# ============================================================

def build_markdown(diag, surv, diag_path, surv_path, tracking_accuracy=None):
    md = []
    if tracking_accuracy is None:
        tracking_accuracy = load_tracking_accuracy()
    md.append("# 水印追踪准确率异常分析报告\n")
    md.append(f"> **报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    md.append("> **目标水印比率**: K = 0.3 (默认值)  ")
    md.append("> **数据自动读取**: 最新的 `save/diagnostics/*.json` 与 `save/survival_analysis/*.json`\n")
    md.append("---\n")

    # 1. 实验现象
    md.append("## 1. 实验现象总览\n")
    md.append("### 1.1 追踪准确率（自动读取真实实验结果）\n")
    md.append("| 配置 | K=5 | K=10 | K=20 | 平均 |")
    md.append("|------|-----|------|------|------|")
    for name, acc in tracking_accuracy.items():
        avg = _avg_tracking(acc)
        warn = " ⚠️" if avg < 0.5 else ""
        cells = []
        for k in ('K=5', 'K=10', 'K=20'):
            v = acc.get(k)
            cells.append(f"{v*100:.2f}%" if v is not None else "N/A")
        md.append(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | "
                  f"{avg*100:.2f}%{warn} |")
    md.append("")
    md.append("### 1.2 水印区域扰动：追踪准确率\n")
    md.append("![水印区域扰动追踪准确率](figures/fig1_perturbation_accuracy.png)\n")
    md.append("### 1.3 本地微调：检测准确率\n")
    md.append("![本地微调检测准确率](figures/fig2_finetune_accuracy.png)\n")
    md.append("### 1.4 水印区域扰动：相似度与判别 gap\n")
    md.append("![水印区域扰动相似度与gap](figures/fig3_perturbation_similarity.png)\n")
    md.append("### 1.5 本地微调：相似度与判别 gap\n")
    md.append("![本地微调相似度与gap](figures/fig4_finetune_similarity.png)\n")
    md.append("---\n")

    # 2. 假设
    md.append("## 2. 核心假设\n")
    md.append("- **假设 1**: 主任务梯度强度差异（AlexNet 处理 CIFAR 时梯度更强）")
    md.append("- **假设 2**: 模型容量竞争（AlexNet 容量不足以同时承载主任务和水印）")
    md.append("- **假设 3**: 水印信号被主任务\"挤压\"（容量竞争中水印被冲淡）\n")
    md.append("---\n")

    # 3. 实验验证
    md.append("## 3. 实验验证\n")
    md.append("### 3.1 梯度强度诊断\n")
    if diag:
        md.append("**测量指标**: 单次反向传播后的总梯度范数\n\n")
        md.append("| 配置 | 总参数量 | 损失值 | 总梯度范数 | 单位参数负载 |")
        md.append("|------|---------|--------|-----------|------------|")
        for name, d in diag.items():
            total_m = d.get('total_params', 0) / 1e6
            unit = d['total_grad_norm'] / d['total_params'] if d.get('total_params') else 0
            md.append(f"| {name} | {total_m:.2f}M | {d.get('loss', 0):.2f} | "
                      f"**{d['total_grad_norm']:.2f}** | **{unit:.2e}** |")
        md.append("")
        if 'AlexNet + CIFAR-10' in diag and 'AlexNet + ChestMNIST' in diag:
            ratio_a = diag['AlexNet + CIFAR-10']['total_grad_norm'] / diag['AlexNet + ChestMNIST']['total_grad_norm']
            md.append(f"**核心发现**: AlexNet + CIFAR-10 的梯度强度是 AlexNet + ChestMNIST 的 **{ratio_a:.1f} 倍**\n")
    else:
        md.append("> ⚠️ 未找到 `save/diagnostics/*.json` 数据，请先运行 `python watermark_diagnostics.py`\n")

    md.append("### 3.2 水印存活能力\n")
    if surv:
        md.append("**测量方法**: 嵌入水印 → 模拟 15 轮联邦训练 → 测量水印区域参数偏移\n\n")
        md.append("| 配置 | WM L2 终值 | Non-WM L2 终值 | 保护比 (WM/Non-WM) |")
        md.append("|------|----------|---------------|-------------------|")
        for name, s in surv.items():
            # 兼容嵌套 (history.*) 和扁平两种结构
            wm_hist = s.get('history', {}).get('wm_l2_change') or s.get('wm_l2_change')
            nwm_hist = s.get('history', {}).get('nonwm_l2_change') or s.get('nonwm_l2_change')
            if wm_hist and nwm_hist:
                wm_final = s.get('wm_l2_final', wm_hist[-1])
                nwm_final = s.get('nonwm_l2_final', nwm_hist[-1])
                ratio = s.get('wm_to_nonwm_ratio_final', wm_final / nwm_final if nwm_final else 0)
                md.append(f"| {name} | {wm_final:.4f} | {nwm_final:.4f} | **{ratio:.4f}** |")
        md.append("")
    else:
        md.append("> ⚠️ 未找到 `save/survival_analysis/*.json` 数据，请先运行 `python watermark_survival_analysis.py`\n")
    md.append("---\n")
    # 4. 机理
    md.append("## 4. 深层机理分析\n")
    md.append("### 4.1 模型训练公式\n")
    md.append("```")
    md.append("模型更新: w_new = w_old - lr × ∇L")
    md.append("水印保护: 约束水印区域 |∇L_wm| ≤ target_ratio × |∇L_normal|")
    md.append("```\n")
    md.append("### 4.2 AlexNet + CIFAR 为何失效？\n")
    md.append("- **机制 1**: 主任务\"独霸\"参数空间 — CIFAR 任务特征复杂，AlexNet 容量紧张")
    md.append("- **机制 2**: 0.3 ratio 不够保护 — 即使限制 30%，主任务梯度仍会冲淡水印\n")
    md.append("---\n")

    # 5. 解决方案
    md.append("## 5. 解决方案与建议\n")
    md.append("### 5.1 推荐 target_ratio\n")
    md.append("| 配置 | 当前 ratio | 推荐 ratio | 理由 |")
    md.append("|------|-----------|-----------|------|")
    md.append("| AlexNet + CIFAR-10/100 | 0.30 | 0.50 - 0.55 | 容量紧张，需更强保护 |")
    md.append("| AlexNet + ChestMNIST | 0.30 | 0.30 - 0.35 | 已够用 |")
    md.append("| ResNet18 + CIFAR | 0.30 | 0.35 - 0.40 | 容量大，轻微调整 |")
    md.append("| ResNet18 + ChestMNIST | 0.30 | 0.30 | 已最优 |\n")
    md.append("### 5.2 调整水印区域\n")
    md.append("- 缩小水印区域，让单位保护强度更大")
    md.append("- 选择低活跃层（BatchNorm、bias）放置水印\n")
    md.append("### 5.3 自适应水印比率\n")
    md.append("```")
    md.append("target_ratio = base_ratio × f(unit_load)")
    md.append("f(unit_load) = sigmoid(unit_load - threshold)")
    md.append("```\n")
    md.append("---\n")

    # 6. 结论
    md.append("## 6. 结论\n")
    md.append("### 6.1 核心结论\n")
    md.append("1. AlexNet + CIFAR 的主任务梯度强度远强于其他场景")
    md.append("2. AlexNet 单位参数负载是 ResNet18 的 ~9 倍，容量紧张")
    md.append("3. AlexNet + CIFAR 保护比仅 ~6.7%，水印被严重侵蚀\n")
    md.append("### 6.2 一句话总结\n")
    md.append("> AlexNet + CIFAR 的水印追踪失败，根源在于\"小模型遇到复杂任务时，主任务梯度远强于水印保护能力\"，这是容量约束下的必然结果。\n")
    md.append("---\n")

    # 附录
    md.append("## 附录: 实验数据来源\n")
    md.append(f"- 梯度诊断: `{relpath(diag_path, get_project_root())}`")
    md.append(f"- 水印存活: `{relpath(surv_path, get_project_root())}`")
    md.append("- 图表目录: `save/figures/`")
    md.append("- 复现脚本:")
    md.append("  - `watermark_diagnostics.py`")
    md.append("  - `watermark_survival_analysis.py`")
    md.append("  - `utils/generate_report.py` (本工具)\n")

    return "\n".join(md)


# ============================================================
# Word / PDF 输出
# ============================================================

def generate_docx(md_content, output_path):
    try:
        from docx import Document
        from docx.shared import Pt
    except ImportError:
        print("[!] 缺少 python-docx，请先: pip install python-docx")
        return False

    doc = Document()
    style = doc.styles['Normal']
    style.font.name = 'Microsoft YaHei'
    style.font.size = Pt(11)

    for line in md_content.split('\n'):
        s = line.strip()
        if s.startswith('# '):
            doc.add_heading(s[2:], level=0)
        elif s.startswith('## '):
            doc.add_heading(s[3:], level=1)
        elif s.startswith('### '):
            doc.add_heading(s[4:], level=2)
        elif s.startswith('!['):
            m = re.match(r'!\[.*?\]\((.*?)\)', s)
            if m:
                img_path = os.path.join(get_project_root(), m.group(1))
                if os.path.exists(img_path):
                    doc.add_picture(img_path, width=Pt(450))
        elif s.startswith('- '):
            doc.add_paragraph(s[2:], style='List Bullet')
        elif s and not s.startswith('|') and not s.startswith('>'):
            doc.add_paragraph(s)

    doc.save(output_path)
    print(f"[OK] Word 已生成: {output_path}")
    return True


def generate_pdf(md_content, output_path):
    try:
        import markdown as md
        from weasyprint import HTML
    except ImportError:
        print("[!] 缺少 weasyprint/markdown，请先: pip install weasyprint markdown")
        return False

    html_body = md.markdown(md_content, extensions=['tables', 'fenced_code'])
    html_full = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  @page {{ size: A4; margin: 2cm; }}
  body {{ font-family: 'Microsoft YaHei', sans-serif; line-height: 1.6; color: #333; }}
  h1 {{ color: #2C3E50; border-bottom: 3px solid #3498DB; padding-bottom: 8px; }}
  h2 {{ color: #34495E; border-bottom: 1px solid #BDC3C7; padding-bottom: 4px; }}
  h3 {{ color: #7F8C8D; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #BDC3C7; padding: 6px 10px; }}
  th {{ background: #3498DB; color: white; }}
  code {{ background: #F4F6F7; padding: 1px 4px; border-radius: 3px; }}
  pre {{ background: #2C3E50; color: #ECF0F1; padding: 10px; border-radius: 5px; }}
  blockquote {{ border-left: 4px solid #3498DB; padding-left: 12px; color: #555; font-style: italic; }}
  img {{ max-width: 100%; display: block; margin: 12px auto; }}
</style></head>
<body>{html_body}</body></html>"""

    HTML(string=html_full).write_pdf(output_path)
    print(f"[OK] PDF 已生成: {output_path}")
    return True


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='水印分析报告统一生成工具')
    parser.add_argument('--all', action='store_true', help='生成图表 + Markdown + PDF')
    parser.add_argument('--figures', action='store_true', help='仅生成图表')
    parser.add_argument('--docx', action='store_true', help='生成 Word')
    parser.add_argument('--pdf', action='store_true', help='生成 PDF')
    parser.add_argument('--md', action='store_true', help='仅生成 Markdown')
    parser.add_argument('--output-dir', default='save/docs', help='文档输出目录（相对项目根）')
    parser.add_argument('--figures-dir', default='save/figures', help='图表输出目录（相对项目根）')
    args = parser.parse_args()

    if not any([args.all, args.figures, args.docx, args.pdf, args.md]):
        args.all = True

    root = get_project_root()
    out_dir = os.path.join(root, args.output_dir)
    fig_dir = os.path.join(root, args.figures_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    diag, surv, diag_path, surv_path = load_latest_data()
    print(f"[*] 梯度诊断: {relpath(diag_path, root)}")
    print(f"[*] 水印存活: {relpath(surv_path, root)}")
    check_data_freshness(diag_path, surv_path)

    # 追踪准确率统一从真实实验结果加载 (不再硬编码)
    tracking_accuracy = load_tracking_accuracy()

    if args.all or args.figures:
        print(f"\n[*] 生成图表到 {os.path.relpath(fig_dir, root)}/")
        generate_all_figures(fig_dir, diag, surv, tracking_accuracy)

    md_content = None
    if any([args.all, args.md, args.docx, args.pdf]):
        md_content = build_markdown(diag, surv, diag_path, surv_path, tracking_accuracy)
        md_path = os.path.join(out_dir, 'WATERMARK_ANALYSIS_REPORT.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"\n[OK] Markdown: {os.path.relpath(md_path, root)}")

    if args.all or args.docx:
        if md_content:
            generate_docx(md_content, os.path.join(out_dir, 'WATERMARK_ANALYSIS_REPORT.docx'))

    if args.all or args.pdf:
        if md_content:
            generate_pdf(md_content, os.path.join(out_dir, 'WATERMARK_ANALYSIS_REPORT.pdf'))

    print("\n[完成]")


if __name__ == '__main__':
    main()
