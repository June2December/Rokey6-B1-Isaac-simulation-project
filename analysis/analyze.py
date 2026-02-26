"""
WandB CSV 분석 및 시각화 스크립트 (Step vs Loss)

같은 메트릭이 여러 CSV에 걸쳐 있으면 Step을 이어붙여 하나의 그래프로 출력합니다.

사용법:
    python3 analyze.py <csv_directory>      # 디렉터리 안 CSV 전체 처리
    python3 analyze.py <file.csv>           # 단일 파일 처리
    python3 analyze.py <csv_directory> 20   # 스무딩 윈도우 지정 (기본 15)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import glob
import re
from datetime import datetime

# ─────────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────────
SMOOTHING_WINDOW = 15
STEP_COL = "Step"
DPI = 300

# ─────────────────────────────────────────────
# CLI 인자 처리
# ─────────────────────────────────────────────
if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

target = sys.argv[1]
if len(sys.argv) >= 3:
    try:
        SMOOTHING_WINDOW = int(sys.argv[2])
    except ValueError:
        print(f"[경고] smoothing_window 값이 올바르지 않습니다. 기본값({SMOOTHING_WINDOW}) 사용.")

# 스크립트 위치 기준으로 charts 디렉터리 고정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(SCRIPT_DIR, "charts")

if os.path.isdir(target):
    csv_files = sorted(glob.glob(os.path.join(target, "*.csv")))
elif os.path.isfile(target):
    csv_files = [target]
else:
    print(f"[오류] 경로를 찾을 수 없습니다: {target}")
    sys.exit(1)

if not csv_files:
    print(f"[오류] CSV 파일이 없습니다: {target}")
    sys.exit(1)

os.makedirs(out_dir, exist_ok=True)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"\n{'='*60}")
print(f"  대상: {target}")
print(f"  CSV 파일 수: {len(csv_files)}개")
print(f"  스무딩 윈도우: {SMOOTHING_WINDOW}")
print(f"  출력 디렉터리: {out_dir}")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────
def parse_col(col: str) -> tuple[str, str]:
    """'RunName - Loss / Value loss' → ('RunName', 'Loss / Value loss')"""
    match = re.match(r"^(.+?)\s+-\s+(.+)$", col)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", col.strip()

def safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-_. ]", "_", name).strip()

# ─────────────────────────────────────────────
# Step 1: 모든 CSV를 읽어 메트릭별로 데이터 수집
#   metric_data[metric_name] = [
#       {"steps": ndarray, "values": ndarray, "run_label": str, "file": str},
#       ...
#   ]  (CSV 파일 순서대로)
# ─────────────────────────────────────────────
metric_data: dict[str, list[dict]] = {}

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    fname = os.path.basename(csv_path)

    # Step 컬럼 탐색
    if STEP_COL in df.columns:
        step_col = STEP_COL
    else:
        candidates = [c for c in df.columns if c.lower() == "step"]
        step_col = candidates[0] if candidates else df.columns[0]

    # __MIN / __MAX 제외한 수치형 메트릭 컬럼
    metric_cols = [
        c for c in df.columns
        if c != step_col
        and not c.endswith("__MIN")
        and not c.endswith("__MAX")
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    print(f"[수집] {fname}  →  메트릭 {len(metric_cols)}개: {[parse_col(c)[1] for c in metric_cols]}")

    for col in metric_cols:
        run_label, metric_name = parse_col(col)
        series = df[col].dropna()
        if series.empty:
            continue
        steps = df.loc[series.index, step_col].values

        if metric_name not in metric_data:
            metric_data[metric_name] = []
        metric_data[metric_name].append({
            "steps":     steps,
            "values":    series.values,
            "run_label": run_label,
            "file":      fname,
        })

if not metric_data:
    print("[오류] 분석할 메트릭이 없습니다.")
    sys.exit(1)

print(f"\n총 메트릭 종류: {list(metric_data.keys())}\n")

# ─────────────────────────────────────────────
# Step 2: 메트릭별로 세그먼트를 이어붙여 그래프 생성
# ─────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
palette = sns.color_palette("tab10", len(metric_data))

for m_idx, (metric_name, segments) in enumerate(metric_data.items()):
    color = palette[m_idx % len(palette)]

    # 세그먼트를 Step 기준으로 이어붙이기
    # 각 CSV의 Step을 직전 CSV 마지막 Step 뒤에 이어지도록 offset 부여
    all_steps  = []
    all_values = []
    boundaries = []   # 각 세그먼트 시작 Step (그래프에 구분선 표시용)

    step_offset = 0
    for seg in segments:
        raw_steps = seg["steps"].astype(float)
        offset_steps = raw_steps - raw_steps[0] + step_offset  # 이전 끝에서 이어붙임

        boundaries.append(offset_steps[0])
        all_steps.append(offset_steps)
        all_values.append(seg["values"])

        # 다음 세그먼트의 시작점 = 이번 세그먼트 마지막 Step + 평균 간격
        if len(raw_steps) > 1:
            avg_gap = float(np.median(np.diff(raw_steps)))
        else:
            avg_gap = 1.0
        step_offset = offset_steps[-1] + avg_gap

    all_steps  = np.concatenate(all_steps)
    all_values = np.concatenate(all_values)
    smoothed   = pd.Series(all_values).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean().values

    # ── 그래프 그리기 ──
    fig, ax = plt.subplots(figsize=(12, 6))

    # 원본 (연하게)
    ax.plot(all_steps, all_values,
            color="gray", alpha=0.35, linewidth=1, label="Raw")

    # 추세선 (스무딩)
    ax.plot(all_steps, smoothed,
            color=color, linewidth=2.5,
            label=f"Smoothed (window={SMOOTHING_WINDOW})")

    # CSV 경계 구분선 (첫 번째 제외)
    for i, bx in enumerate(boundaries[1:], start=1):
        ax.axvline(x=bx, color="red", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(bx, ax.get_ylim()[1], f" run {i+1}",
                color="red", fontsize=8, va="top", ha="left")

    # run 수가 여러 개면 제목에 표기
    n_runs = len(segments)
    title_suffix = f"  ({n_runs} runs concatenated)" if n_runs > 1 else ""
    ax.set_title(f"{metric_name}{title_suffix}", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Step", fontsize=13, fontweight="bold")
    ax.set_ylabel(metric_name, fontsize=13, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=11, loc="best")
    plt.tight_layout()

    # 파일명 생성: PPO_value_loss_20260226_143052
    first_run_label = segments[0]["run_label"]
    algo = first_run_label.split("_")[-1] if first_run_label else "unknown"
    metric_short = metric_name.split("/")[-1].strip().lower().replace(" ", "_")
    save_name = f"{algo}_{metric_short}_{RUN_TIMESTAMP}"

    plt.savefig(os.path.join(out_dir, f"{save_name}.png"), dpi=DPI)
    plt.savefig(os.path.join(out_dir, f"{save_name}.svg"), format="svg")
    plt.close(fig)
    print(f"[저장] {save_name}.png  ({n_runs}개 run, 총 {len(all_steps)}포인트)")

print(f"\n{'='*60}")
print(f"  완료! 출력 디렉터리: {out_dir}")
print(f"{'='*60}\n")
