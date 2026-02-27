"""
mission_analysis.py — mission_events CSV 분석 및 결과 출력

사용법:
  # 최신 CSV 자동 검색
  python3 scripts/mission_analysis.py

  # 파일 직접 지정
  python3 scripts/mission_analysis.py mission_events_XXXX.csv
"""
import glob
import json
import os
import sys

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── 한글 폰트 설정 ─────────────────────────────────────────────────────────────
_FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False   # 마이너스 기호 깨짐 방지

ROBOT_COLORS = ["#2196F3", "#FF5722"]   # Robot0: 파랑, Robot1: 주황

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mission_data")


def find_latest(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ── 오탐 제거: 시작 직후(5초 이내) basecamp_return 은 초기화 노이즈 ──────────
    noise_mask = (df["event_type"] == "basecamp_return") & (df["elapsed_s"] < 5.0)
    removed = noise_mask.sum()
    if removed:
        print(f"[Filter] 초기화 노이즈 {removed}건 제거 (elapsed < 5s basecamp_return)")
    df = df[~noise_mask].reset_index(drop=True)
    return df


def analyze(df: pd.DataFrame) -> dict:
    results = {}
    total_elapsed = df["elapsed_s"].max()

    minerals = df[df["event_type"] == "mineral_collect"]
    returns  = df[df["event_type"] == "basecamp_return"]

    # ── 로봇별 분석 ─────────────────────────────────────────────────────────────
    robot_stats = {}
    for rid in sorted(df["robot_id"].unique()):
        rob_min = minerals[minerals["robot_id"] == rid].copy()
        rob_ret = returns[returns["robot_id"] == rid].copy()

        # 라운드별 수집 시간 간격
        round_durations = []
        ret_times = rob_ret["elapsed_s"].tolist()
        if rob_min.empty:
            round_durations = []
        else:
            first_mineral_time = rob_min["elapsed_s"].min()
            prev_t = first_mineral_time
            for t in ret_times:
                if t > prev_t:
                    round_durations.append(t - prev_t)
                    prev_t = t

        # 광물 수집 간 평균 시간
        if len(rob_min) > 1:
            mineral_times = rob_min["elapsed_s"].values
            avg_mineral_interval = float((mineral_times[-1] - mineral_times[0]) / (len(rob_min) - 1))
        else:
            avg_mineral_interval = None

        robot_stats[int(rid)] = {
            "mineral_count":        len(rob_min),
            "round_count":          len(rob_ret),
            "rounds_completed":     sorted(rob_ret["round"].tolist()),
            "avg_round_duration_s": round(sum(round_durations) / len(round_durations), 1) if round_durations else None,
            "avg_mineral_interval_s": round(avg_mineral_interval, 1) if avg_mineral_interval else None,
            "minerals_per_min":     round(len(rob_min) / (total_elapsed / 60), 2) if total_elapsed > 0 else 0,
            "collection_coords":    rob_min[["round", "mineral_num", "x", "y"]].to_dict("records"),
        }

    # ── 전체 통계 ────────────────────────────────────────────────────────────────
    results["session"] = {
        "total_elapsed_s":   round(total_elapsed, 1),
        "total_elapsed_min": round(total_elapsed / 60, 2),
        "total_minerals":    len(minerals),
        "total_returns":     len(returns),
        "overall_minerals_per_min": round(len(minerals) / (total_elapsed / 60), 2) if total_elapsed > 0 else 0,
    }
    results["robots"] = robot_stats
    return results


def print_report(results: dict):
    sess = results["session"]
    sep  = "=" * 58

    print(f"\n{sep}")
    print("         미션 분석 결과")
    print(sep)
    elapsed_min = int(sess["total_elapsed_s"] // 60)
    elapsed_sec = int(sess["total_elapsed_s"] % 60)
    print(f"  기록 시간   : {elapsed_min}분 {elapsed_sec}초")
    print(f"  전체 수집   : {sess['total_minerals']}개")
    print(f"  전체 복귀   : {sess['total_returns']}회")
    print(f"  전체 효율   : {sess['overall_minerals_per_min']:.2f} 개/min")

    for rid, st in results["robots"].items():
        print(f"\n{'─'*58}")
        print(f"  Robot {rid}")
        print(f"{'─'*58}")
        print(f"  수집 광물 수         : {st['mineral_count']}개")
        print(f"  완료 라운드 수       : {st['round_count']}회  {st['rounds_completed']}")
        print(f"  수집 효율           : {st['minerals_per_min']:.2f} 개/min")

        if st["avg_round_duration_s"] is not None:
            m, s = divmod(int(st["avg_round_duration_s"]), 60)
            print(f"  평균 라운드 소요시간 : {m}분 {s}초")
        if st["avg_mineral_interval_s"] is not None:
            print(f"  광물 간 평균 간격   : {st['avg_mineral_interval_s']}초")

        print(f"\n  수집 좌표 목록:")
        print(f"  {'R-No':>5}  {'x':>10}  {'y':>10}")
        print(f"  {'─'*30}")
        for c in st["collection_coords"]:
            label = f"R{int(c['round'])}-{int(c['mineral_num'])}"
            print(f"  {label:>5}  {c['x']:>10.2f}  {c['y']:>10.2f}")

    print(f"\n{sep}\n")


def plot_graphs(event_df: pd.DataFrame, log_df: pd.DataFrame, results: dict, ts: str):
    """
    4개 패널 그래프 생성 및 저장.

    [1] 누적 수집량 vs 시간   [2] 속도 vs 시간
    [3] 목표까지 거리 vs 시간  [4] 라운드별 소요 시간
    """
    minerals = event_df[event_df["event_type"] == "mineral_collect"]
    returns  = event_df[event_df["event_type"] == "basecamp_return"]
    total_s  = results["session"]["total_elapsed_s"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Mission Analysis", fontsize=14, fontweight="bold", color="black")
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("#f7f7f7")
        ax.tick_params(colors="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        ax.grid(True, color="#cccccc", alpha=0.6, linewidth=0.5)

    def fmt_min(x, _):
        return f"{int(x//60)}:{int(x%60):02d}"

    # ── [1] 누적 수집량 vs 시간 ────────────────────────────────────────────────
    ax1 = axes[0, 0]
    ax1.set_title("누적 수집량 vs 시간")
    ax1.set_xlabel("시간 (mm:ss)")
    ax1.set_ylabel("누적 광물 수 (개)")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_min))

    cumulative_total = []
    for rid, color in enumerate(ROBOT_COLORS):
        rob_min = minerals[minerals["robot_id"] == rid].sort_values("elapsed_s")
        if rob_min.empty:
            continue
        times  = [0.0] + rob_min["elapsed_s"].tolist() + [total_s]
        counts = list(range(len(rob_min) + 1)) + [len(rob_min)]
        ax1.step(times, counts, where="post", color=color, linewidth=1.8,
                 label=f"Robot {rid}")
        # 수집 이벤트 마커
        ax1.scatter(rob_min["elapsed_s"], range(1, len(rob_min) + 1),
                    color=color, s=40, zorder=5)
        cumulative_total.append((rob_min["elapsed_s"].tolist(), list(range(1, len(rob_min) + 1))))

    # 전체 합산 누적
    all_times  = minerals.sort_values("elapsed_s")["elapsed_s"].tolist()
    all_counts = list(range(1, len(all_times) + 1))
    if all_times:
        ax1.step([0.0] + all_times + [total_s],
                 [0] + all_counts + [all_counts[-1]],
                 where="post", color="#555555", linewidth=1.2,
                 linestyle="--", alpha=0.7, label="전체 합산")

    # 베이스캠프 복귀 시점 수직선
    for _, row in returns.iterrows():
        ax1.axvline(row["elapsed_s"], color=ROBOT_COLORS[int(row["robot_id"])],
                    linestyle=":", alpha=0.4, linewidth=1)

    ax1.legend(fontsize=8, facecolor="white", edgecolor="#cccccc",
               labelcolor="black", loc="upper left")

    # ── [2] 속도 vs 시간 ──────────────────────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.set_title("속도 vs 시간")
    ax2.set_xlabel("시간 (mm:ss)")
    ax2.set_ylabel("속도 (m/s)")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_min))

    if log_df is not None:
        for rid, color in enumerate(ROBOT_COLORS):
            col = f"r{rid}_speed"
            if col in log_df.columns:
                sub = log_df[["elapsed_s", col]].dropna()
                ax2.plot(sub["elapsed_s"], sub[col], color=color,
                         linewidth=0.8, alpha=0.85, label=f"Robot {rid}")
        # 수집 이벤트 마커
        for rid, color in enumerate(ROBOT_COLORS):
            rob_min = minerals[minerals["robot_id"] == rid]
            for t in rob_min["elapsed_s"]:
                ax2.axvline(t, color=color, linestyle=":", alpha=0.3, linewidth=0.8)
        ax2.legend(fontsize=8, facecolor="white", edgecolor="#cccccc",
                   labelcolor="black")
    else:
        ax2.text(0.5, 0.5, "mission_log 없음", transform=ax2.transAxes,
                 ha="center", va="center", color="#888888")

    # ── [3] 목표까지 거리 vs 시간 ─────────────────────────────────────────────
    ax3 = axes[1, 0]
    ax3.set_title("목표까지 거리 vs 시간")
    ax3.set_xlabel("시간 (mm:ss)")
    ax3.set_ylabel("거리 (m)")
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_min))

    if log_df is not None:
        for rid, color in enumerate(ROBOT_COLORS):
            col = f"r{rid}_distance"
            if col in log_df.columns:
                sub = log_df[["elapsed_s", col]].dropna()
                ax3.plot(sub["elapsed_s"], sub[col], color=color,
                         linewidth=0.8, alpha=0.85, label=f"Robot {rid}")
        ax3.legend(fontsize=8, facecolor="white", edgecolor="#cccccc",
                   labelcolor="black")
    else:
        ax3.text(0.5, 0.5, "mission_log 없음", transform=ax3.transAxes,
                 ha="center", va="center", color="#888888")

    # ── [4] 라운드별 소요 시간 ────────────────────────────────────────────────
    ax4 = axes[1, 1]
    ax4.set_title("라운드별 소요 시간")
    ax4.set_xlabel("라운드")
    ax4.set_ylabel("소요 시간 (초)")

    bar_x, bar_h, bar_c, bar_labels = [], [], [], []
    offset = 0
    for rid, color in enumerate(ROBOT_COLORS):
        rob_min = minerals[minerals["robot_id"] == rid].sort_values("elapsed_s")
        rob_ret = returns[returns["robot_id"] == rid].sort_values("elapsed_s")
        if rob_min.empty:
            continue
        prev_t = rob_min["elapsed_s"].min()
        for _, row in rob_ret.iterrows():
            if row["elapsed_s"] > prev_t:
                dur = row["elapsed_s"] - prev_t
                bar_x.append(offset)
                bar_h.append(dur)
                bar_c.append(color)
                bar_labels.append(f"R{int(row['round'])}\nRob{rid}")
                offset += 1
                prev_t = row["elapsed_s"]

    if bar_x:
        bars = ax4.bar(bar_x, bar_h, color=bar_c, edgecolor="#cccccc",
                       linewidth=0.5, width=0.6)
        ax4.set_xticks(bar_x)
        ax4.set_xticklabels(bar_labels, fontsize=7, color="black")
        for bar, h in zip(bars, bar_h):
            m, s = divmod(int(h), 60)
            ax4.text(bar.get_x() + bar.get_width() / 2, h + 1,
                     f"{m}:{s:02d}", ha="center", va="bottom",
                     fontsize=7, color="black")
        # 평균선
        avg_dur = np.mean(bar_h)
        ax4.axhline(avg_dur, color="#555555", linestyle="--", linewidth=1,
                    alpha=0.7, label=f"평균 {avg_dur:.0f}초")
        ax4.legend(fontsize=8, facecolor="white", edgecolor="#cccccc",
                   labelcolor="black")
    else:
        ax4.text(0.5, 0.5, "완료된 라운드 없음", transform=ax4.transAxes,
                 ha="center", va="center", color="#888888")

    # ── 저장 ──────────────────────────────────────────────────────────────────
    plt.tight_layout()
    out_path = os.path.join(DATA_DIR, f"mission_graph_{ts}.png")
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"[Graph] 저장 완료: {out_path}")
    plt.show()


def main():
    if len(sys.argv) >= 2:
        event_path = sys.argv[1]
    else:
        event_path = find_latest(os.path.join(DATA_DIR, "mission_events_*.csv"))

    if not event_path or not os.path.exists(event_path):
        print(f"[Error] mission_events_*.csv 파일을 찾을 수 없습니다. ({DATA_DIR})")
        sys.exit(1)

    print(f"[Analysis] 파일: {event_path}")

    df      = load_events(event_path)
    results = analyze(df)
    ts      = os.path.basename(event_path).replace("mission_events_", "").replace(".csv", "")

    print_report(results)

    # mission_log 자동 검색 (속도/거리 그래프용)
    log_path = os.path.join(DATA_DIR, f"mission_log_{ts}.csv")
    log_df   = pd.read_csv(log_path) if os.path.exists(log_path) else None
    if log_df is None:
        print("[Graph] mission_log_*.csv 없음 — 속도/거리 그래프 생략")

    plot_graphs(df, log_df, results, ts)

    # JSON으로도 저장
    json_path = os.path.join(DATA_DIR, f"mission_result_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Analysis] JSON 저장: {json_path}")


if __name__ == "__main__":
    main()
