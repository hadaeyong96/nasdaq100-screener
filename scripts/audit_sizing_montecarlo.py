"""KJB-1.1 이후 사이징 감사 — 수정된 cfg(account.total_krw == backtest.total_krw)로
B0.5 2016~2021 무작위 진입 500회를 다시 돌린다.

일회성 감사 스크립트(CLAUDE.md scripts/). 실제 몬테카를로 계산은 전부
scripts/p5_5_experiments.py의 monte_carlo_luck_test(및 그 안의 _mc_single_draw)를
그대로 재사용한다 — 새 무작위 진입 로직을 만들지 않는다. 이 스크립트는 그 함수를
25건씩 나눠 반복 호출하면서(이미 있는 체크포인트 덕분에 앞부분은 즉시 재사용됨)
진행 상황·예상 종료 시각을 로그 파일에 남기는 오케스트레이션만 한다.

실행(반드시 python -u로 — 진행 로그가 즉시 보이게):
    python -u -m scripts.audit_sizing_montecarlo --n-draws 10   시험 실행(10회)
    python -u -m scripts.audit_sizing_montecarlo                전체 500회(체크포인트에서 이어서)

체크포인트: outputs/backtest/p5_5_checkpoint/audit_fixed_2016_batchNN.pkl
           (scripts/p5_5_experiments.py의 기존 체크포인트 메커니즘 그대로 재사용 —
           죽었다 다시 실행해도 이미 끝난 배치는 다시 계산하지 않는다)
로그: outputs/backtest/sizing_audit/mc_fixed_2016.log (25건마다 진행·예상 종료 시각)
진행 확인:
    tail -f outputs/backtest/sizing_audit/mc_fixed_2016.log   (또는 그냥 tail)
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from datetime import date  # noqa: E402

from engine import backtest as bt  # noqa: E402
from scripts.p5_5_experiments import monte_carlo_luck_test  # noqa: E402

OUT_DIR = bt.OUTPUT_ROOT / "sizing_audit"
LOG_PATH = OUT_DIR / "mc_fixed_2016.log"
SEED_BASE = 6000  # P5-5 자체의 mc_full(2016~2021 "전체")과 같은 시드 규칙 — 비교 가능하게
CKPT_PREFIX = "audit_fixed_2016"
STEP = 25


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-draws", type=int, default=500)
    args = parser.parse_args()

    # n_draws < 500(시험 실행)은 반드시 다른 체크포인트 이름을 써야 한다 — 500과
    # 같은 이름을 쓰면 마지막 배치가 n_draws로 잘려서 저장되고, 나중에 진짜 500회
    # 실행이 그 잘린 배치를 재사용하려다 KeyError로 죽는다(실제로 겪은 버그 —
    # 10회 시험 실행의 batch00이 draws 0~9만 담고 있었는데, 500회 실행의 batch00은
    # draws 0~24를 기대해서 터졌다). 500일 때만 진짜 이름을 쓴다.
    ckpt_prefix = CKPT_PREFIX if args.n_draws == 500 else f"{CKPT_PREFIX}_test{args.n_draws}"

    cfg = bt.load_config()
    if cfg["account"]["total_krw"] != cfg["backtest"]["total_krw"]:
        raise SystemExit(
            "cfg[\"account\"][\"total_krw\"] != cfg[\"backtest\"][\"total_krw\"] — "
            "config.yaml이 아직 KJB-1.1 수정 전 상태입니다."
        )

    warmup_start = bt._parse_date(cfg["backtest"]["warmup_start"])
    train_end = bt._parse_date(cfg["backtest"]["train_end"])
    start = date(2016, 1, 1)

    log(f"=== 시작 (n_draws={args.n_draws}, seed_base={SEED_BASE}) ===")
    log("prepare_data ...")
    data = bt.prepare_data(cfg, warmup_start, train_end)

    log("수정된 B0.5 2016-2021 실제 시뮬레이션 ...")
    result = bt.simulate_portfolio(data, cfg, start, train_end, apply_costs=True, apply_tax=True, max_slots=5)
    result_pretax = bt.simulate_portfolio(data, cfg, start, train_end, apply_costs=True, apply_tax=False, max_slots=5)
    liquidated = bt.compute_liquidated_cagr(result, data, cfg, start, train_end)
    scenario = {"result": result, "result_pretax": result_pretax, "cagr_a_pct": liquidated.get("cagr_liquidated_pct")}
    log(f"실제 세후(a) = {scenario['cagr_a_pct']}%")

    n_draws = args.n_draws
    start_time = time.time()
    mc = None
    done = 0
    step = min(STEP, n_draws)
    target = step
    while True:
        mc = monte_carlo_luck_test(
            data, cfg, scenario, start, train_end, n_draws=target, seed_base=SEED_BASE,
            batch_size=STEP, ckpt_prefix=ckpt_prefix,
        )
        done = target
        elapsed = time.time() - start_time
        rate = elapsed / done if done else 0
        remaining = n_draws - done
        eta_seconds = rate * remaining
        eta_time = datetime.now() + timedelta(seconds=eta_seconds)
        log(
            f"진행 {done} / {n_draws} — 경과 {elapsed:.0f}초, "
            f"예상 종료 {eta_time.isoformat(timespec='seconds')} "
            f"(남은 {remaining}건, 건당 평균 {rate:.1f}초)"
        )
        if done >= n_draws:
            break
        target = min(target + step, n_draws)

    log(f"=== 완료 === {mc}")


if __name__ == "__main__":
    main()
