from __future__ import annotations

import contextlib
import io
import math
import multiprocessing
import os
import random
import subprocess
import sys
import tempfile
import traceback
from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from macro_place.benchmark import Benchmark


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_FULL_SRC = _HERE / "placer.cpp"
_FAST_SRC = _HERE / "placerFast.cpp"
_BUILD_DIR = _HERE / "build"
_FULL_BIN = _BUILD_DIR / "vibe_placer_full"
_FAST_BIN = _BUILD_DIR / "vibe_placer_fast"


def _build_binary(src: Path, binary: Path) -> Path:
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    needs_build = not binary.exists() or binary.stat().st_mtime < src.stat().st_mtime
    if not needs_build:
        return binary

    cxx = os.environ.get("CXX", "c++")
    cmd = [
        cxx,
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-Wall",
        "-Wextra",
        "-pthread",
        "-o",
        str(binary),
        str(src),
    ]
    try:
        subprocess.run(cmd, check=True, cwd=str(_HERE))
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"failed to build {src.name} with {' '.join(cmd)}"
        ) from exc
    return binary


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _analytical_warm_start(
    benchmark: Benchmark,
    seed: int,
    iter_idx: int = 1,
    official_net_weights=None,
    official_net_count=None,
    load_official_proxy: bool = True,
    knobs: dict | None = None,
) -> torch.Tensor:
    """Run vplacer17's analytical frontend, leaving C++ to legalize and run SA.

    ``knobs`` is an optional dict of per-run overrides for the Adam-phase
    hyperparameters (see ``_KNOB_DEFAULTS``). If ``None``, the module-level
    defaults are used (baseline behavior).
    """
    if not _env_flag("VIBECPP_ANALYTICAL_WARMSTART", True):
        return benchmark.macro_positions.detach().cpu().clone()

    try:
        # When running in parallel (8 workers), default to fewer threads per
        # worker to avoid CPU oversubscription.
        n_parallel_workers = _env_int("VIBECPP_SEED_SWEEP_COUNT", 8)
        analytical_threads = _env_int(
            "VIBECPP_ANALYTICAL_THREADS",
            max(1, min(8, (os.cpu_count() or 1) // max(1, n_parallel_workers))),
        )
        placer = VibePlacer(
            seed=int(seed),
            num_threads=analytical_threads,
            deterministic=True,
        )

        torch.use_deterministic_algorithms(True)
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        random.seed(int(seed))
        torch.set_num_threads(max(1, int(analytical_threads)))
        try:
            torch.set_num_interop_threads(min(2, max(1, int(analytical_threads))))
        except RuntimeError:
            pass

        n_hard = int(benchmark.num_hard_macros)
        n_total = int(benchmark.num_macros)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes_np = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64)
        fixed_np = benchmark.macro_fixed.detach().cpu().numpy().astype(bool)
        original_pos = (
            benchmark.macro_positions.detach().cpu().numpy().astype(np.float64).copy()
        )
        half_w = sizes_np[:, 0] / 2.0
        half_h = sizes_np[:, 1] / 2.0

        threshold = LARGE_MACRO_AREA_FRAC * cw * ch
        large_mask = np.zeros(n_total, dtype=bool)
        for i in range(n_hard):
            if not fixed_np[i] and (sizes_np[i, 0] * sizes_np[i, 1]) > threshold:
                large_mask[i] = True
        large_indices = np.where(large_mask)[0]

        if load_official_proxy and official_net_weights is None:
            official_plc = _load_official_proxy_plc(benchmark)
            official_net_weights, official_net_count = _official_proxy_params_from_plc(
                benchmark,
                official_plc,
                len(benchmark.net_pin_nodes),
            )

        print(
            "[vibeCpp] analytical warm-start enabled | "
            f"threads={analytical_threads}",
            flush=True,
        )
        positions = original_pos.copy()
        positions = placer._step1_center_init(
            positions, fixed_np, half_w, half_h, n_total, cw, ch
        )
        _dump_global_step_snapshot(
            benchmark,
            positions,
            sizes_np,
            fixed_np,
            iter_idx,
            1,
            "step1_center_init",
            official_net_weights,
        )
        positions = placer._step3_adam_large(
            benchmark,
            positions,
            large_mask,
            fixed_np,
            sizes_np,
            cw,
            ch,
            n_total,
            official_net_weights_np=official_net_weights,
            knobs=knobs,
        )
        _dump_global_step_snapshot(
            benchmark,
            positions,
            sizes_np,
            fixed_np,
            iter_idx,
            2,
            "step3_adam_large",
            official_net_weights,
        )
        positions = _legalize_to_boundary(
            positions,
            large_indices,
            sizes_np,
            cw,
            ch,
            gap=GAP,
        )
        _dump_global_step_snapshot(
            benchmark,
            positions,
            sizes_np,
            fixed_np,
            iter_idx,
            3,
            "step4_legalize_to_boundary",
            official_net_weights,
        )
        positions = placer._step6_adam_small_and_exact(
            benchmark,
            positions,
            large_mask,
            fixed_np,
            sizes_np,
            cw,
            ch,
            n_total,
            official_net_weights_np=official_net_weights,
            knobs=knobs,
        )
        _dump_global_step_snapshot(
            benchmark,
            positions,
            sizes_np,
            fixed_np,
            iter_idx,
            4,
            "step6_adam_small_and_exact",
            official_net_weights,
        )
        positions[:n_hard] = _legalize_large_then_small(
            positions[:n_hard],
            sizes_np[:n_hard],
            fixed_np[:n_hard],
            large_mask[:n_hard],
            cw,
            ch,
            gap=GAP,
        )
        _dump_global_step_snapshot(
            benchmark,
            positions,
            sizes_np,
            fixed_np,
            iter_idx,
            5,
            "step7_legalize_large_then_small",
            official_net_weights,
        )
        positions = placer._global_analytical_place(
            benchmark,
            positions,
            sizes_np,
            fixed_np,
            large_mask,
            n_hard,
            n_total,
            cw,
            ch,
            official_net_weights_np=official_net_weights,
            official_wl_net_count=official_net_count,
            knobs=knobs,
        )
        _dump_global_step_snapshot(
            benchmark,
            positions,
            sizes_np,
            fixed_np,
            iter_idx,
            6,
            "step8_global_analytical_place",
            official_net_weights,
        )
        positions = placer._finalize_positions(
            positions, half_w, half_h, fixed_np, original_pos, cw, ch
        )
        _dump_global_step_snapshot(
            benchmark,
            positions,
            sizes_np,
            fixed_np,
            iter_idx,
            7,
            "step9_finalize_positions",
            official_net_weights,
        )
        print("[vibeCpp] analytical warm-start done", flush=True)
        return torch.tensor(positions, dtype=torch.float32)
    except Exception as exc:
        if _env_flag("VIBECPP_ANALYTICAL_STRICT", False):
            raise
        print(
            f"[vibeCpp] analytical warm-start failed; using input placement: {exc}",
            flush=True,
        )
        return benchmark.macro_positions.detach().cpu().clone()


def _pin_rows(benchmark: Benchmark) -> List[Tuple[int, float, float, int]]:
    n_hard = int(benchmark.num_hard_macros)
    n_total = int(benchmark.num_macros)

    pin_offsets_per_macro = []
    for i in range(n_hard):
        if i < len(benchmark.macro_pin_offsets):
            offsets = benchmark.macro_pin_offsets[i].detach().cpu()
            pin_offsets_per_macro.append(offsets)
        else:
            pin_offsets_per_macro.append(torch.zeros(0, 2))

    rows: List[Tuple[int, float, float, int]] = []
    for net_i, pins in enumerate(benchmark.net_pin_nodes):
        if pins.numel() == 0 or pins.shape[0] < 2:
            continue
        for owner_t, slot_t in pins.detach().cpu().tolist():
            owner = int(owner_t)
            slot = int(slot_t)
            off_x = 0.0
            off_y = 0.0
            if 0 <= owner < n_hard:
                offsets = pin_offsets_per_macro[owner]
                if 0 <= slot < offsets.shape[0]:
                    off_x = float(offsets[slot, 0])
                    off_y = float(offsets[slot, 1])
            elif owner < n_total:
                off_x = 0.0
                off_y = 0.0
            rows.append((owner, off_x, off_y, int(net_i)))
    return rows


def _official_proxy_params(benchmark: Benchmark) -> Tuple[List[float] | None, float | None]:
    try:
        from macro_place.loader import load_benchmark_from_dir
    except Exception:
        return None, None

    for bench_dir in _benchmark_dir_candidates(benchmark):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _unused_benchmark, plc = load_benchmark_from_dir(str(bench_dir))
            weights = []
            for driver in plc.nets.keys():
                mod_idx = plc.mod_name_to_indices[driver]
                weights.append(float(plc.modules_w_pins[mod_idx].get_weight()))
            if len(weights) != int(benchmark.num_nets):
                continue
            net_count = float(getattr(plc, "net_cnt", 0.0) or 0.0)
            if net_count <= 0.0:
                net_count = float(sum(weights))
            return weights, net_count
        except Exception:
            continue
    return None, None


def _write_input(
    path: Path,
    benchmark: Benchmark,
    positions_override: torch.Tensor | None = None,
    net_weights_override=None,
    wl_net_count_override: float | None = None,
) -> None:
    positions = (
        positions_override.detach().cpu()
        if positions_override is not None
        else benchmark.macro_positions.detach().cpu()
    )
    sizes = benchmark.macro_sizes.detach().cpu()
    fixed = benchmark.macro_fixed.detach().cpu()
    ports = benchmark.port_positions.detach().cpu()
    pins = _pin_rows(benchmark)
    if net_weights_override is None:
        official_weights, official_net_count = _official_proxy_params(benchmark)
        net_weights = (
            official_weights
            if official_weights is not None
            else benchmark.net_weights.detach().cpu().tolist()
        )
        net_count = (
            float(official_net_count)
            if official_net_count is not None
            else float(max(int(benchmark.num_nets), 1))
        )
    else:
        net_weights = net_weights_override
        net_count = (
            float(wl_net_count_override)
            if wl_net_count_override is not None
            else float(max(int(benchmark.num_nets), 1))
        )

    def f(value: float) -> str:
        return f"{float(value):.17g}"

    with path.open("w", encoding="utf-8") as out:
        out.write("VIBECPP1\n")
        out.write(f"name {benchmark.name}\n")
        out.write(f"canvas {f(benchmark.canvas_width)} {f(benchmark.canvas_height)}\n")
        out.write(
            "counts "
            f"{int(benchmark.num_macros)} "
            f"{int(benchmark.num_hard_macros)} "
            f"{int(ports.shape[0])} "
            f"{int(benchmark.num_nets)} "
            f"{len(pins)} "
            f"{int(benchmark.grid_rows)} "
            f"{int(benchmark.grid_cols)} "
            f"{f(getattr(benchmark, 'hroutes_per_micron', 0.0))} "
            f"{f(getattr(benchmark, 'vroutes_per_micron', 0.0))}\n"
        )
        out.write(f"net_count {f(net_count)}\n")
        out.write("macros\n")
        for i in range(int(benchmark.num_macros)):
            out.write(
                f"{f(positions[i, 0])} {f(positions[i, 1])} "
                f"{f(sizes[i, 0])} {f(sizes[i, 1])} {1 if bool(fixed[i]) else 0}\n"
            )
        out.write("ports\n")
        for i in range(int(ports.shape[0])):
            out.write(f"{f(ports[i, 0])} {f(ports[i, 1])}\n")
        out.write("pins\n")
        for owner, off_x, off_y, net_i in pins:
            out.write(f"{owner} {f(off_x)} {f(off_y)} {net_i}\n")
        out.write("net_weights\n")
        for i in range(int(benchmark.num_nets)):
            weight = float(net_weights[i]) if i < len(net_weights) else 1.0
            out.write(f"{f(weight)}\n")
        out.write("END\n")


def _read_output(path: Path, n_total: int) -> torch.Tensor:
    rows = []
    with path.open("r", encoding="utf-8") as inp:
        for line in inp:
            stripped = line.strip()
            if not stripped:
                continue
            x_s, y_s = stripped.split()[:2]
            rows.append((float(x_s), float(y_s)))
    if len(rows) != n_total:
        raise RuntimeError(f"C++ placer returned {len(rows)} rows; expected {n_total}")
    
    return torch.tensor(rows, dtype=torch.float32)


def _seed_sweep_worker(
    benchmark: Benchmark,
    seed: int,
    idx: int,
    official_net_weights,
    official_net_count,
    write_net_weights,
    write_net_count,
    knobs: dict,
    in_path: Path,
    out_path: Path,
    err_path: Path,
    log_path: Path,
    fast_binary: Path,
) -> None:
    """Run one analytical warm-start + fast C++ engine in a child process.

    All stdout/stderr output (Python prints and C++ subprocess output) is
    redirected to ``log_path`` so the parent can replay logs sequentially.
    The result placement is written to ``out_path``.  If anything goes wrong
    the traceback is written to ``err_path`` so the parent can report it.
    """
    # Redirect stdout and stderr to a per-worker log file so output from
    # parallel workers does not interleave.
    log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
    sys.stdout = log_file
    sys.stderr = log_file
    try:
        warm_positions = _analytical_warm_start(
            benchmark,
            seed,
            iter_idx=idx,
            official_net_weights=official_net_weights,
            official_net_count=official_net_count,
            load_official_proxy=False,
            knobs=knobs,
        )
        _write_input(
            in_path,
            benchmark,
            positions_override=warm_positions,
            net_weights_override=write_net_weights,
            wl_net_count_override=write_net_count,
        )
        subprocess.run(
            [
                str(fast_binary),
                "--input",
                str(in_path),
                "--output",
                str(out_path),
                "--seed",
                str(seed),
            ],
            check=True,
            cwd=str(_HERE),
            stdout=log_file,
            stderr=log_file,
        )
    except Exception:
        err_path.write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        log_file.flush()
        log_file.close()


class VibeCppPlacer:
    def __init__(self, seed: int = 7):
        self.seed = int(seed)
        self._full_binary = _build_binary(_FULL_SRC, _FULL_BIN)
        self._fast_binary = _build_binary(_FAST_SRC, _FAST_BIN)
        self._seed_sweep_count = max(1, _env_int("VIBECPP_SEED_SWEEP_COUNT", 8))
        self._seed_sweep_stride = max(
            1, _env_int("VIBECPP_SEED_SWEEP_STRIDE", 1009)
        )
        self._final_topk = max(1, _env_int("VIBECPP_FINAL_TOPK", 2))

    def _sweep_seeds(self) -> List[int]:
        return [
            self.seed + idx * self._seed_sweep_stride
            for idx in range(self._seed_sweep_count)
        ]

    def _run_engine(
        self,
        binary: Path,
        in_path: Path,
        out_path: Path,
        seed: int,
    ) -> None:
        subprocess.run(
            [
                str(binary),
                "--input",
                str(in_path),
                "--output",
                str(out_path),
                "--seed",
                str(seed),
            ],
            check=True,
            cwd=str(_HERE),
        )

    def _score_positions(
        self,
        benchmark: Benchmark,
        positions: torch.Tensor,
        official_proxy_plc,
        official_net_weights,
        official_net_count,
    ):
        sizes_np = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64)
        pos_np = positions.detach().cpu().numpy().astype(np.float64)
        return _proxy_components_for_acceptance(
            benchmark,
            pos_np,
            sizes_np,
            int(benchmark.num_macros),
            official_plc=official_proxy_plc,
            net_weights_np=official_net_weights,
            wl_net_count=official_net_count,
        )

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        official_proxy_plc = _load_official_proxy_plc(benchmark)
        official_net_weights, official_net_count = _official_proxy_params_from_plc(
            benchmark, official_proxy_plc, len(benchmark.net_pin_nodes)
        )
        write_net_weights = (
            official_net_weights
            if official_net_weights is not None
            else benchmark.net_weights.detach().cpu().tolist()
        )
        write_net_count = (
            float(official_net_count)
            if official_net_count is not None
            else float(max(int(benchmark.num_nets), 1))
        )
        seeds = self._sweep_seeds()
        with tempfile.TemporaryDirectory(prefix="vibecpp_") as tmpdir_s:
            tmpdir = Path(tmpdir_s)
            print(
                "[vibeCpp] parallel seed sweep enabled | "
                f"count={len(seeds)} seeds={','.join(str(s) for s in seeds)}",
                flush=True,
            )

            # ── Launch all analytical warm-starts + fast engines in parallel ──
            fast_binary = self._fast_binary
            n_macros = int(benchmark.num_macros)
            worker_args = []
            for idx, seed in enumerate(seeds, start=1):
                knobs = _sample_knobs(seed, baseline=(idx == 1))
                tag = "baseline" if idx == 1 else f"perturbed sigma={KNOB_PERTURB_REL_STD:.2f}"
                print(
                    f"[vibeCpp] launching worker {idx}/{len(seeds)} | "
                    f"seed={seed} | knobs={tag}",
                    flush=True,
                )
                print(f"[vibeCpp] knobs | {_format_knobs(knobs)}", flush=True)
                in_path = tmpdir / f"benchmark_fast_{idx}.txt"
                out_path = tmpdir / f"placement_fast_{idx}.txt"
                err_path = tmpdir / f"error_{idx}.txt"
                log_path = tmpdir / f"log_{idx}.txt"
                worker_args.append((
                    benchmark,
                    seed,
                    idx,
                    official_net_weights,
                    official_net_count,
                    write_net_weights,
                    write_net_count,
                    knobs,
                    in_path,
                    out_path,
                    err_path,
                    log_path,
                    fast_binary,
                ))

            # Use spawn context to avoid macOS PyTorch MPS fork crash.
            # Add _HERE to sys.path so the spawned child can import 'placer'
            if str(_HERE) not in sys.path:
                sys.path.insert(0, str(_HERE))
            ctx = multiprocessing.get_context("spawn")
            processes = []
            for args in worker_args:
                p = ctx.Process(
                    target=_seed_sweep_worker,
                    args=args,
                )
                p.start()
                processes.append(p)

            # Wait for all workers to finish.
            for p in processes:
                p.join()

            # ── Replay each worker's captured log sequentially ────────────────
            for idx, (seed, args) in enumerate(
                zip(seeds, worker_args), start=1
            ):
                log_path = args[11]  # log_path
                print(
                    f"\n{'═' * 72}\n"
                    f"[vibeCpp] ── worker {idx}/{len(seeds)} "
                    f"seed={seed} log ──\n"
                    f"{'═' * 72}",
                    flush=True,
                )
                if log_path.exists():
                    print(
                        log_path.read_text(encoding="utf-8", errors="replace"),
                        end="",
                        flush=True,
                    )
                else:
                    print("  (no log output)", flush=True)
            print(f"{'═' * 72}\n", flush=True)

            # ── Collect results from all workers ──────────────────────────────
            best_positions = None
            best_score = None
            best_seed = None
            candidates = []

            for idx, (seed, args) in enumerate(
                zip(seeds, worker_args), start=1
            ):
                out_path = args[9]  # out_path
                err_path = args[10]  # err_path
                if err_path.exists():
                    err_msg = err_path.read_text(encoding="utf-8").strip()
                    print(
                        f"[vibeCpp] seed sweep worker {idx} FAILED | "
                        f"seed={seed}: {err_msg}",
                        flush=True,
                    )
                    continue
                if not out_path.exists():
                    print(
                        f"[vibeCpp] seed sweep worker {idx} produced no output | "
                        f"seed={seed}",
                        flush=True,
                    )
                    continue
                candidate = _read_output(out_path, n_macros)
                score = self._score_positions(
                    benchmark,
                    candidate,
                    official_proxy_plc,
                    official_net_weights,
                    official_net_count,
                )
                print(
                    "[vibeCpp] seed sweep candidate | "
                    f"seed={seed} proxy={score['proxy']:.6f} "
                    f"WL={score['wirelength']:.6f} "
                    f"density={score['density']:.6f} "
                    f"congestion={score['congestion']:.6f} "
                    f"source={score['source']}",
                    flush=True,
                )
                candidates.append(
                    {
                        "seed": seed,
                        "positions": candidate,
                        "score": score,
                    }
                )
                if best_score is None or score["proxy"] < best_score["proxy"]:
                    best_positions = candidate
                    best_score = score
                    best_seed = seed

            if best_positions is None:
                raise RuntimeError(
                    "All seed sweep workers failed; cannot proceed."
                )
            assert best_score is not None
            assert best_seed is not None
            print(
                "[vibeCpp] seed sweep selected | "
                f"seed={best_seed} proxy={best_score['proxy']:.6f} "
                f"source={best_score['source']}",
                flush=True,
            )

            finalists = sorted(
                candidates,
                key=lambda item: item["score"]["proxy"],
            )[: min(self._final_topk, len(candidates))]
            print(
                "[vibeCpp] final SA push | "
                f"engine={_FULL_SRC.name} top_k={len(finalists)} "
                f"seeds={','.join(str(item['seed']) for item in finalists)}",
                flush=True,
            )
            best_final_positions = None
            best_final_score = None
            best_final_seed = None
            for final_idx, item in enumerate(finalists, start=1):
                final_seed = int(item["seed"])
                final_in_path = tmpdir / f"benchmark_final_{final_idx}.txt"
                final_out_path = tmpdir / f"placement_final_{final_idx}.txt"
                _write_input(
                    final_in_path,
                    benchmark,
                    positions_override=item["positions"],
                    net_weights_override=write_net_weights,
                    wl_net_count_override=write_net_count,
                )
                print(
                    "[vibeCpp] final SA candidate | "
                    f"{final_idx}/{len(finalists)} seed={final_seed} "
                    f"fast_proxy={item['score']['proxy']:.6f}",
                    flush=True,
                )
                self._run_engine(
                    self._full_binary,
                    final_in_path,
                    final_out_path,
                    final_seed,
                )
                final_positions = _read_output(
                    final_out_path,
                    int(benchmark.num_macros),
                )
                final_score = self._score_positions(
                    benchmark,
                    final_positions,
                    official_proxy_plc,
                    official_net_weights,
                    official_net_count,
                )
                print(
                    "[vibeCpp] final SA result | "
                    f"seed={final_seed} proxy={final_score['proxy']:.6f} "
                    f"WL={final_score['wirelength']:.6f} "
                    f"density={final_score['density']:.6f} "
                    f"congestion={final_score['congestion']:.6f} "
                    f"source={final_score['source']}",
                    flush=True,
                )
                if (
                    best_final_score is None
                    or final_score["proxy"] < best_final_score["proxy"]
                ):
                    best_final_positions = final_positions
                    best_final_score = final_score
                    best_final_seed = final_seed

            assert best_final_positions is not None
            assert best_final_score is not None
            assert best_final_seed is not None
            print(
                "[vibeCpp] final SA selected | "
                f"seed={best_final_seed} "
                f"proxy={best_final_score['proxy']:.6f} "
                f"source={best_final_score['source']}",
                flush=True,
            )
            positions = best_final_positions
            _dump_global_step_snapshot(
                benchmark,
                positions,
                benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64),
                benchmark.macro_fixed.detach().cpu().numpy(),
                100,
                8,
                "step10_final_sa",
                official_net_weights,
            )
            return positions


GAP = 0.0005
EPS = 1e-12
LEGALIZE_RING = 80

LARGE_MACRO_AREA_FRAC = 0.005

STEP3_ITERS = 400
STEP3_LR_FRAC = 0.01
STEP3_LAMBDA_START = 1.0
STEP3_LAMBDA_END = 1000.0

STEP6_ITERS = 360
STEP6_LR_FRAC = 0.01
STEP6_LAMBDA_START = 0.01
STEP6_LAMBDA_END = 200.0

EXACT_DENSITY_ITERS = 1600
EXACT_DENSITY_LR_FRAC = 0.0004
EXACT_DENSITY_LAMBDA_START = 100_000.0
EXACT_DENSITY_LAMBDA_END = 10_000_000.0
EXACT_DENSITY_TARGET = 0.65
EXACT_DENSITY_OVERFLOW_WEIGHT = 8.0

GLOBAL_PLACE_ITERS = 1200
GLOBAL_PLACE_LR_FRAC = 0.0015
GLOBAL_PLACE_LAMBDA_START = 20.0
GLOBAL_PLACE_LAMBDA_END = 3000.0
GLOBAL_PLACE_DENSITY_TARGET_SLACK = 0.85
# Original v13 value 0.35. ABU cong_pen is ~100x larger in magnitude than the
# legacy RUDY penalty, so the *effective* gradient contribution at 0.35 is
# already much stronger than v13. Keep at 0.35 to avoid drowning out density.
GLOBAL_PLACE_CONG_WEIGHT = 0.35
GLOBAL_PLACE_ACCEPT_TOL = 1e-6


# ── Per-seed knob perturbation ───────────────────────────────────────────────
# When the outer seed sweep runs N analytical warm-starts, we keep the first
# run on the baseline defaults and draw perturbed knobs for the rest. This
# gives more variation across runs without ever being worse than the tuned
# baseline. Each knob is multiplied by ``1 + N(0, KNOB_PERTURB_REL_STD)``
# and then clamped to a safe range determined by ``kind``:
#   "int"   → round to int, min 1
#   "float" → positive (>= 1e-12)
#   "frac"  → clamp into (0.05, 0.95) — for density targets in [0, 1]
#   "slack" → clamp into (0.05, 1.5)  — for density-target slack
KNOB_PERTURB_REL_STD = 0.20

_KNOB_DEFAULTS = {
    "step3_iters":                       (STEP3_ITERS, "int"),
    "step3_lr_frac":                     (STEP3_LR_FRAC, "float"),
    "step3_lambda_start":                (STEP3_LAMBDA_START, "float"),
    "step3_lambda_end":                  (STEP3_LAMBDA_END, "float"),
    "step6_iters":                       (STEP6_ITERS, "int"),
    "step6_lr_frac":                     (STEP6_LR_FRAC, "float"),
    "step6_lambda_start":                (STEP6_LAMBDA_START, "float"),
    "step6_lambda_end":                  (STEP6_LAMBDA_END, "float"),
    "exact_density_iters":               (EXACT_DENSITY_ITERS, "int"),
    "exact_density_lr_frac":             (EXACT_DENSITY_LR_FRAC, "float"),
    "exact_density_lambda_start":        (EXACT_DENSITY_LAMBDA_START, "float"),
    "exact_density_lambda_end":          (EXACT_DENSITY_LAMBDA_END, "float"),
    "exact_density_target":              (EXACT_DENSITY_TARGET, "frac"),
    "exact_density_overflow_weight":     (EXACT_DENSITY_OVERFLOW_WEIGHT, "float"),
    "global_place_iters":                (GLOBAL_PLACE_ITERS, "int"),
    "global_place_lr_frac":              (GLOBAL_PLACE_LR_FRAC, "float"),
    "global_place_lambda_start":         (GLOBAL_PLACE_LAMBDA_START, "float"),
    "global_place_lambda_end":           (GLOBAL_PLACE_LAMBDA_END, "float"),
    "global_place_density_target_slack": (GLOBAL_PLACE_DENSITY_TARGET_SLACK, "slack"),
    "global_place_cong_weight":          (GLOBAL_PLACE_CONG_WEIGHT, "float"),
}


def _clamp_knob(value: float, kind: str) -> float:
    if kind == "int":
        return max(1, int(round(value)))
    if kind == "frac":
        return float(min(0.95, max(0.05, value)))
    if kind == "slack":
        return float(min(1.5, max(0.05, value)))
    return float(max(1e-12, value))


def _baseline_knobs() -> dict:
    return {name: default for name, (default, _kind) in _KNOB_DEFAULTS.items()}


def _sample_knobs(seed: int, baseline: bool = False,
                  rel_std: float = KNOB_PERTURB_REL_STD) -> dict:
    """Return a dict of knob values for one analytical warm-start run.

    If ``baseline`` is True or the env flag ``VIBECPP_KNOB_PERTURB`` is
    disabled, return the un-perturbed defaults. Otherwise multiply each
    default by ``1 + N(0, rel_std)`` (deterministic RNG seeded from
    ``seed``) and clamp.
    """
    if baseline or not _env_flag("VIBECPP_KNOB_PERTURB", True):
        return _baseline_knobs()
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    out = {}
    for name, (default, kind) in _KNOB_DEFAULTS.items():
        factor = 1.0 + float(rng.normal(0.0, rel_std))
        out[name] = _clamp_knob(float(default) * factor, kind)
    return out


def _format_knobs(knobs: dict) -> str:
    parts = []
    for name in _KNOB_DEFAULTS:
        v = knobs[name]
        if isinstance(v, int):
            parts.append(f"{name}={v}")
        else:
            parts.append(f"{name}={v:.4g}")
    return " ".join(parts)


# IBM17/IBM18 are soft-macro dominated, but they do not want the same rescue:
# ibm18 needs a hard density push, while ibm17 recovers better when the spread
# phase remains close to v13 and lets SA pull wirelength back down.


# v13 used 600k; on ibm17/ibm18 the final density only converges to ~0.525
# with ~600k iters. At 100k the SA bails out at density ~0.65 and leaves
# cong stuck at 2.0+. Restore the larger budget; it remains within the
# 1-hour per-benchmark limit even on the biggest designs.


# Final dedicated congestion-focused polish (post all 9-stages)


# Macro-blockage gradient strength inside the analytical congestion penalty.
# 0.0 reproduces the legacy net-only RUDY-style penalty.
CONG_MACRO_BLOCKAGE_WEIGHT = 0.0
# Skip the analytical macro-blockage term in early Adam phases where macros
# are still bunched at the canvas center: at that point the blockage signal
# is mostly self-repulsion noise that traps macros into local minima.
CONG_MACRO_BLOCKAGE_EARLY_DISABLE = True
# Use the legacy single-grid RUDY cong penalty (matches vplacer13) instead of
# the ABU-style V/H separated penalty. The ABU penalty has ~100x larger
# magnitude than legacy, which over-weights congestion vs. density in Adam
# loss for large designs (ibm17/ibm18 show phase-separated layouts).
# True = use legacy `_routing_congestion_penalty` (safe default).
# False = use new `_routing_congestion_penalty_abu`.
USE_LEGACY_CONG_PENALTY = True


HROUTING_ALLOC = 30.304
VROUTING_ALLOC = 71.304
SMOOTH_RANGE = 2
CONGESTION_TOP_FRAC = 0.05
DENSITY_TOP_FRAC = 0.10


def _build_pin_arrays_np(
    benchmark: Benchmark,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cached = getattr(benchmark, "_vplacer_pin_arrays_np", None)
    if cached is not None:
        return cached

    n_hard = benchmark.num_hard_macros
    n_total = benchmark.num_macros
    n_ports = int(benchmark.port_positions.shape[0])
    port_pos_np = (
        benchmark.port_positions.detach().cpu().numpy().astype(np.float64)
        if n_ports > 0
        else np.zeros((0, 2), dtype=np.float64)
    )

    pin_offsets_per_macro: List[np.ndarray] = []
    for i in range(n_hard):
        po = (
            benchmark.macro_pin_offsets[i]
            if i < len(benchmark.macro_pin_offsets)
            else None
        )
        if po is None or po.numel() == 0:
            pin_offsets_per_macro.append(np.zeros((1, 2), dtype=np.float64))
        else:
            pin_offsets_per_macro.append(
                po.detach().cpu().numpy().astype(np.float64)
            )

    owners: List[int] = []
    offsets: List[Tuple[float, float]] = []
    nets: List[int] = []
    for net_i, npn in enumerate(benchmark.net_pin_nodes):
        if npn.numel() == 0:
            continue
        arr = npn.detach().cpu().numpy()
        if len(arr) < 2:
            continue
        for o, s in arr:
            o = int(o)
            s = int(s)
            owners.append(o)
            if o < n_hard:
                po = pin_offsets_per_macro[o]
                if 0 <= s < len(po):
                    offsets.append((float(po[s, 0]), float(po[s, 1])))
                else:
                    offsets.append((0.0, 0.0))
            else:
                offsets.append((0.0, 0.0))
            nets.append(net_i)

    if not owners:
        result = (
            np.zeros(0, dtype=np.int64),
            np.zeros((0, 2), dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            port_pos_np,
        )
        setattr(benchmark, "_vplacer_pin_arrays_np", result)
        return result
    result = (
        np.asarray(owners, dtype=np.int64),
        np.asarray(offsets, dtype=np.float64),
        np.asarray(nets, dtype=np.int64),
        port_pos_np,
    )
    setattr(benchmark, "_vplacer_pin_arrays_np", result)
    return result


def _net_pin_index_table(
    pin_net_np: np.ndarray, num_nets: int
) -> List[np.ndarray]:
    if num_nets == 0:
        return []
    order = np.argsort(pin_net_np, kind="stable")
    sorted_nets = pin_net_np[order]
    bounds = np.searchsorted(sorted_nets, np.arange(num_nets + 1))
    return [order[bounds[ni] : bounds[ni + 1]] for ni in range(num_nets)]

def _net_pin_index_table_cached(
    benchmark: Benchmark, pin_net_np: np.ndarray, num_nets: int
) -> List[np.ndarray]:
    cached = getattr(benchmark, "_vplacer_net_pin_index_table", None)
    if cached is not None and len(cached) == num_nets:
        return cached
    table = _net_pin_index_table(pin_net_np, num_nets)
    setattr(benchmark, "_vplacer_net_pin_index_table", table)
    return table

def _all_net_hpwls_vec(
    pin_owner_np: np.ndarray,
    pin_offset_np: np.ndarray,
    pin_net_np: np.ndarray,
    pos: np.ndarray,
    port_pos_np: np.ndarray,
    n_total: int,
    num_nets: int,
) -> np.ndarray:
    if len(pin_owner_np) == 0 or num_nets == 0:
        return np.zeros(num_nets, dtype=np.float64)

    n_ports = port_pos_np.shape[0]
    is_port = pin_owner_np >= n_total
    safe_macro = np.where(is_port, 0, pin_owner_np)
    safe_port = np.where(is_port, pin_owner_np - n_total, 0)

    macro_x = pos[safe_macro, 0] + pin_offset_np[:, 0]
    macro_y = pos[safe_macro, 1] + pin_offset_np[:, 1]
    if n_ports > 0:
        port_x = port_pos_np[safe_port, 0]
        port_y = port_pos_np[safe_port, 1]
    else:
        port_x = macro_x                                   
        port_y = macro_y

    pin_x = np.where(is_port, port_x, macro_x)
    pin_y = np.where(is_port, port_y, macro_y)

    max_x = np.full(num_nets, -np.inf, dtype=np.float64)
    min_x = np.full(num_nets, np.inf, dtype=np.float64)
    max_y = np.full(num_nets, -np.inf, dtype=np.float64)
    min_y = np.full(num_nets, np.inf, dtype=np.float64)
    np.maximum.at(max_x, pin_net_np, pin_x)
    np.minimum.at(min_x, pin_net_np, pin_x)
    np.maximum.at(max_y, pin_net_np, pin_y)
    np.minimum.at(min_y, pin_net_np, pin_y)

    out = np.zeros(num_nets, dtype=np.float64)
    has_pins = np.isfinite(max_x)
    out[has_pins] = (max_x[has_pins] - min_x[has_pins]) + (
        max_y[has_pins] - min_y[has_pins]
    )
    return out


class _IncrementalDensityCost:

    def __init__(self, benchmark, pos, sizes_np, top_frac=DENSITY_TOP_FRAC):
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.grid_col = max(1, int(benchmark.grid_cols))
        self.grid_row = max(1, int(benchmark.grid_rows))
        self.grid_w = self.cw / self.grid_col
        self.grid_h = self.ch / self.grid_row
        self.grid_area = max(self.grid_w * self.grid_h, EPS)

        self.sizes = sizes_np
        self.pos = pos.copy()
        self.occupied = np.zeros(
            (self.grid_row, self.grid_col), dtype=np.float64
        )

        for i in range(len(pos)):
            self._add_macro(i, pos[i, 0], pos[i, 1], 1.0)

        self.top_count = max(
            1, int(math.floor(self.grid_col * self.grid_row * top_frac))
        )
        self.current_cost = self._calc_cost()
        self.saved_occupied: np.ndarray | None = None
        self.saved_region: tuple | None = None


    def _add_macro(self, i, cx, cy, sign):
        w = float(self.sizes[i, 0])
        h = float(self.sizes[i, 1])
        if w <= 0.0 or h <= 0.0:
            return
        lx = float(cx - w / 2)
        ux = float(cx + w / 2)
        ly = float(cy - h / 2)
        uy = float(cy + h / 2)
        if ux <= 0.0 or uy <= 0.0 or lx >= self.cw or ly >= self.ch:
            return

        c0 = max(0, min(self.grid_col - 1, int(math.floor(lx / self.grid_w))))
        c1 = max(0, min(self.grid_col - 1, int(math.floor(ux / self.grid_w))))
        r0 = max(0, min(self.grid_row - 1, int(math.floor(ly / self.grid_h))))
        r1 = max(0, min(self.grid_row - 1, int(math.floor(uy / self.grid_h))))

        for r in range(r0, r1 + 1):
            y0 = r * self.grid_h
            y1 = y0 + self.grid_h
            oy = max(0.0, min(uy, y1) - max(ly, y0))
            if oy <= 0.0:
                continue
            for c in range(c0, c1 + 1):
                x0 = c * self.grid_w
                x1 = x0 + self.grid_w
                ox = max(0.0, min(ux, x1) - max(lx, x0))
                if ox > 0.0:
                    self.occupied[r, c] += sign * ox * oy

    def _calc_cost(self):
        density = self.occupied.flatten() / self.grid_area
        nonzero = np.sort(density[density > 1e-9])[::-1]
        if len(nonzero) == 0:
            return 0.0
        if len(density) < 10:
            return 0.5 * float(nonzero.mean())
        return 0.5 * float(
            nonzero[: min(self.top_count, len(nonzero))].sum() / self.top_count
        )


class _IncrementalCongestionCost:

    def __init__(
        self,
        benchmark: Benchmark,
        positions: np.ndarray,
        sizes_np: np.ndarray,
        pin_owner_np: np.ndarray,
        pin_offset_np: np.ndarray,
        nets_pin_indices: List[np.ndarray],
        port_pos_np: np.ndarray,
        hrouting_alloc: float = HROUTING_ALLOC,
        vrouting_alloc: float = VROUTING_ALLOC,
        smooth_range: int = SMOOTH_RANGE,
        top_frac: float = CONGESTION_TOP_FRAC,
        net_weights_np: np.ndarray | None = None,
    ):
        self.benchmark = benchmark
        self.pos = positions
        self.sizes = sizes_np
        self.pin_owner_np = pin_owner_np
        self.pin_offset_np = pin_offset_np
        self.nets_pin_indices = nets_pin_indices
        self.port_pos_np = port_pos_np
        self.n_total = benchmark.num_macros
        self.n_hard = benchmark.num_hard_macros
        self.grid_col = int(benchmark.grid_cols)
        self.grid_row = int(benchmark.grid_rows)
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.grid_w = self.cw / self.grid_col
        self.grid_h = self.ch / self.grid_row
        self.grid_v_routes = self.grid_w * float(benchmark.vroutes_per_micron)
        self.grid_h_routes = self.grid_h * float(benchmark.hroutes_per_micron)
        self.inv_grid_v_routes = 1.0 / max(self.grid_v_routes, EPS)
        self.inv_grid_h_routes = 1.0 / max(self.grid_h_routes, EPS)
        self.hrouting_alloc = float(hrouting_alloc)
        self.vrouting_alloc = float(vrouting_alloc)
        self.smooth_range = int(math.floor(smooth_range))
        self.top_frac = float(top_frac)
        self._v_smooth = []
        for col in range(self.grid_col):
            lo = max(0, col - self.smooth_range)
            hi = min(self.grid_col, col + self.smooth_range + 1)
            self._v_smooth.append((lo, hi, self.inv_grid_v_routes / (hi - lo)))
        self._h_smooth = []
        for row in range(self.grid_row):
            lo = max(0, row - self.smooth_range)
            hi = min(self.grid_row, row + self.smooth_range + 1)
            self._h_smooth.append((lo, hi, self.inv_grid_h_routes / (hi - lo)))

        if (
            net_weights_np is not None
            and len(net_weights_np) == len(nets_pin_indices)
        ):
            self.net_weights = np.asarray(net_weights_np, dtype=np.float64)
        elif benchmark.net_weights.numel() == len(nets_pin_indices):
            self.net_weights = (
                benchmark.net_weights.detach().cpu().numpy().astype(np.float64)
            )
        else:
            self.net_weights = np.ones(len(nets_pin_indices), dtype=np.float64)

        self.v_route = np.zeros((self.grid_row, self.grid_col), dtype=np.float64)
        self.h_route = np.zeros((self.grid_row, self.grid_col), dtype=np.float64)
        self.v_macro = np.zeros((self.grid_row, self.grid_col), dtype=np.float64)
        self.h_macro = np.zeros((self.grid_row, self.grid_col), dtype=np.float64)
        self.grids = (self.v_route, self.h_route, self.v_macro, self.h_macro)
        self.grid_size = self.grid_row * self.grid_col
        self.abu_top_count = int(
            math.floor(2 * self.grid_size * self.top_frac)
        )
        self._v_total = np.empty(
            (self.grid_row, self.grid_col), dtype=np.float64
        )
        self._h_total = np.empty(
            (self.grid_row, self.grid_col), dtype=np.float64
        )
        self._abu_total = np.empty(2 * self.grid_size, dtype=np.float64)
        self._v_route_delta = np.zeros(
            (self.grid_row, self.grid_col), dtype=np.float64
        )
        self._h_route_delta = np.zeros(
            (self.grid_row, self.grid_col), dtype=np.float64
        )

        self.net_contribs = []
        self.net_route_keys = []
        for net_i in range(len(self.nets_pin_indices)):
            contrib, route_key = self._net_contrib(net_i, return_key=True)
            self.net_contribs.append(contrib)
            self.net_route_keys.append(route_key)
            self._apply_net(contrib, 1.0)

        self.macro_contribs: List = [None] * self.n_hard
        for macro_i in range(self.n_hard):
            contrib = self._macro_contrib(macro_i)
            self.macro_contribs[macro_i] = contrib
            self._apply_macro(contrib, 1.0)

        self.current_cost = self._abu_cost()

    def _grid_cell(self, x, y):
        row = int(math.floor(y / self.grid_h))
        col = int(math.floor(x / self.grid_w))
        row = max(0, min(self.grid_row - 1, row))
        col = max(0, min(self.grid_col - 1, col))
        return row, col


    @staticmethod
    def _overlap_dist(ax0, ax1, ay0, ay1, bx0, bx1, by0, by1):
        x_diff = min(ax1, bx1) - max(ax0, bx0)
        y_diff = min(ay1, by1) - max(ay0, by0)
        if x_diff > 0 and y_diff > 0:
            return x_diff, y_diff
        return 0.0, 0.0

    def _add_v_route(self, contrib, row, col, weight):
        if row < 0 or row >= self.grid_row or col < 0 or col >= self.grid_col:
            return
        lo, hi, scale = self._v_smooth[col]
        contrib[0].append((row, lo, hi, weight * scale))

    def _add_h_route(self, contrib, row, col, weight):
        if row < 0 or row >= self.grid_row or col < 0 or col >= self.grid_col:
            return
        lo, hi, scale = self._h_smooth[row]
        contrib[1].append((lo, hi, col, weight * scale))

    def _route_two(self, contrib, source_gcell, node_gcells, weight):
        cells = list(node_gcells)
        sink_gcell = cells[1] if cells[0] == source_gcell else cells[0]
        row_min = min(sink_gcell[0], source_gcell[0])
        row_max = max(sink_gcell[0], source_gcell[0])
        col_min = min(sink_gcell[1], source_gcell[1])
        col_max = max(sink_gcell[1], source_gcell[1])
        for col in range(col_min, col_max):
            self._add_h_route(contrib, source_gcell[0], col, weight)
        for row in range(row_min, row_max):
            self._add_v_route(contrib, row, sink_gcell[1], weight)

    def _route_l(self, contrib, cells, weight):
        cells = sorted(cells, key=lambda x: (x[1], x[0]))
        y1, x1 = cells[0]
        y2, x2 = cells[1]
        y3, x3 = cells[2]
        for col in range(x1, x2):
            self._add_h_route(contrib, y1, col, weight)
        for col in range(x2, x3):
            self._add_h_route(contrib, y2, col, weight)
        for row in range(min(y1, y2), max(y1, y2)):
            self._add_v_route(contrib, row, x2, weight)
        for row in range(min(y2, y3), max(y2, y3)):
            self._add_v_route(contrib, row, x3, weight)

    def _route_t(self, contrib, cells, weight):
        cells = sorted(cells)
        y1, x1 = cells[0]
        y2, x2 = cells[1]
        y3, x3 = cells[2]
        xmin = min(x1, x2, x3)
        xmax = max(x1, x2, x3)
        for col in range(xmin, xmax):
            self._add_h_route(contrib, y2, col, weight)
        for row in range(min(y1, y2), max(y1, y2)):
            self._add_v_route(contrib, row, x1, weight)
        for row in range(min(y2, y3), max(y2, y3)):
            self._add_v_route(contrib, row, x3, weight)

    def _route_three(self, contrib, node_gcells, weight):
        cells = sorted(node_gcells, key=lambda x: (x[1], x[0]))
        y1, x1 = cells[0]
        y2, x2 = cells[1]
        y3, x3 = cells[2]
        if x1 < x2 and x2 < x3 and min(y1, y3) < y2 and max(y1, y3) > y2:
            self._route_l(contrib, cells, weight)
        elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
            for col in range(x1, x2):
                self._add_h_route(contrib, y1, col, weight)
            for row in range(y1, max(y2, y3)):
                self._add_v_route(contrib, row, x2, weight)
        elif y2 == y3:
            for col in range(x1, x2):
                self._add_h_route(contrib, y1, col, weight)
            for col in range(x2, x3):
                self._add_h_route(contrib, y2, col, weight)
            for row in range(min(y2, y1), max(y2, y1)):
                self._add_v_route(contrib, row, x2, weight)
        else:
            self._route_t(contrib, cells, weight)

    def _get_net_gcells(self, idx_arr):
        n_total = self.n_total
        grid_row_minus_1 = self.grid_row - 1
        grid_col_minus_1 = self.grid_col - 1

        pos = self.pos
        pin_owner_np = self.pin_owner_np
        pin_offset_np = self.pin_offset_np
        port_pos_np = self.port_pos_np

        node_gcells = set()
        source_gcell = None

        for i, pin_idx in enumerate(idx_arr):
            pin_idx = int(pin_idx)
            owner = int(pin_owner_np[pin_idx])
            if owner < n_total:
                px = pos[owner, 0] + pin_offset_np[pin_idx, 0]
                py = pos[owner, 1] + pin_offset_np[pin_idx, 1]
            else:
                port_idx = owner - n_total
                px = port_pos_np[port_idx, 0]
                py = port_pos_np[port_idx, 1]

            row = int(math.floor(py / self.grid_h))
            col = int(math.floor(px / self.grid_w))

            if row < 0: row = 0
            elif row > grid_row_minus_1: row = grid_row_minus_1
            if col < 0: col = 0
            elif col > grid_col_minus_1: col = grid_col_minus_1

            gcell = (row, col)
            node_gcells.add(gcell)
            if i == 0:
                source_gcell = gcell

        return source_gcell, node_gcells


    def _net_contrib_from_gcells(self, net_i, source_gcell, node_gcells):
        contrib = ([], [])
        weight = float(self.net_weights[net_i])
        if len(node_gcells) == 2:
            self._route_two(contrib, source_gcell, node_gcells, weight)
        elif len(node_gcells) == 3:
            self._route_three(contrib, node_gcells, weight)
        elif len(node_gcells) > 3:
            for gcell in node_gcells:
                if gcell != source_gcell:
                    self._route_two(
                        contrib, source_gcell, {source_gcell, gcell}, weight
                    )
        return contrib

    def _net_contrib(self, net_i, return_key=False):
        idx_arr = self.nets_pin_indices[net_i]
        if len(idx_arr) < 2:
            route_key = None
            contrib = ([], [])
            return (contrib, route_key) if return_key else contrib
        source_gcell, node_gcells = self._get_net_gcells(idx_arr)
        route_key = source_gcell, tuple(sorted(node_gcells))
        contrib = self._net_contrib_from_gcells(
            net_i, source_gcell, node_gcells
        )
        return (contrib, route_key) if return_key else contrib

    def _macro_contrib(self, macro_i):
        vpts = []
        hpts = []
        x = self.pos[macro_i, 0]
        y = self.pos[macro_i, 1]
        w = float(self.sizes[macro_i, 0])
        h = float(self.sizes[macro_i, 1])
        x0 = x - w / 2
        x1 = x + w / 2
        y0 = y - h / 2
        y1 = y + h / 2
        bl_row, bl_col = self._grid_cell(x0, y0)
        ur_row, ur_col = self._grid_cell(x1, y1)
        partial_vertical = False
        partial_horizontal = False

        for row in range(bl_row, ur_row + 1):
            for col in range(bl_col, ur_col + 1):
                gx0 = col * self.grid_w
                gx1 = (col + 1) * self.grid_w
                gy0 = row * self.grid_h
                gy1 = (row + 1) * self.grid_h
                x_dist, y_dist = self._overlap_dist(
                    x0, x1, y0, y1, gx0, gx1, gy0, gy1
                )
                if ur_row != bl_row and row in (bl_row, ur_row):
                    if abs(y_dist - self.grid_h) > 1e-5:
                        partial_vertical = True
                if ur_col != bl_col and col in (bl_col, ur_col):
                    if abs(x_dist - self.grid_w) > 1e-5:
                        partial_horizontal = True
                vpts.append(
                    (row, col, x_dist * self.vrouting_alloc * self.inv_grid_v_routes)
                )
                hpts.append(
                    (row, col, y_dist * self.hrouting_alloc * self.inv_grid_h_routes)
                )

        if partial_vertical:
            row = ur_row
            for col in range(bl_col, ur_col + 1):
                gx0 = col * self.grid_w
                gx1 = (col + 1) * self.grid_w
                gy0 = row * self.grid_h
                gy1 = (row + 1) * self.grid_h
                x_dist, _ = self._overlap_dist(
                    x0, x1, y0, y1, gx0, gx1, gy0, gy1
                )
                vpts.append(
                    (row, col, -x_dist * self.vrouting_alloc * self.inv_grid_v_routes)
                )

        if partial_horizontal:
            col = ur_col
            for row in range(bl_row, ur_row + 1):
                gx0 = col * self.grid_w
                gx1 = (col + 1) * self.grid_w
                gy0 = row * self.grid_h
                gy1 = (row + 1) * self.grid_h
                _, y_dist = self._overlap_dist(
                    x0, x1, y0, y1, gx0, gx1, gy0, gy1
                )
                hpts.append(
                    (row, col, -y_dist * self.hrouting_alloc * self.inv_grid_h_routes)
                )
        return vpts, hpts

    def _apply_net(self, contrib, sign):
        vsegs, hsegs = contrib
        if vsegs:
            v_route = self.v_route
            if sign == 1.0:
                for row, c_lo, c_hi, val in vsegs:
                    v_route[row, c_lo:c_hi] += val
            elif sign == -1.0:
                for row, c_lo, c_hi, val in vsegs:
                    v_route[row, c_lo:c_hi] -= val
            else:
                for row, c_lo, c_hi, val in vsegs:
                    v_route[row, c_lo:c_hi] += sign * val
        if hsegs:
            h_route = self.h_route
            if sign == 1.0:
                for r_lo, r_hi, col, val in hsegs:
                    h_route[r_lo:r_hi, col] += val
            elif sign == -1.0:
                for r_lo, r_hi, col, val in hsegs:
                    h_route[r_lo:r_hi, col] -= val
            else:
                for r_lo, r_hi, col, val in hsegs:
                    h_route[r_lo:r_hi, col] += sign * val


    def _apply_macro(self, contrib, sign):
        vpts, hpts = contrib
        if vpts:
            v_macro = self.v_macro
            if sign == 1.0:
                for row, col, val in vpts:
                    v_macro[row, col] += val
            elif sign == -1.0:
                for row, col, val in vpts:
                    v_macro[row, col] -= val
            else:
                for row, col, val in vpts:
                    v_macro[row, col] += sign * val
        if hpts:
            h_macro = self.h_macro
            if sign == 1.0:
                for row, col, val in hpts:
                    h_macro[row, col] += val
            elif sign == -1.0:
                for row, col, val in hpts:
                    h_macro[row, col] -= val
            else:
                for row, col, val in hpts:
                    h_macro[row, col] += sign * val

    def _abu_cost(self):
        np.add(self.v_route, self.v_macro, out=self._v_total)
        np.add(self.h_route, self.h_macro, out=self._h_total)
        self._abu_total[: self.grid_size] = self._v_total.reshape(-1)
        self._abu_total[self.grid_size :] = self._h_total.reshape(-1)
        cnt = self.abu_top_count
        if cnt == 0:
            return float(self._abu_total.max()) if len(self._abu_total) else 0.0
        return float(np.partition(self._abu_total, -cnt)[-cnt:].mean())


def _vis_step_dir() -> Path:
    raw = os.environ.get("VIBECPP_VIS_STEP_DIR")
    if raw:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else _REPO_ROOT / path
    return _REPO_ROOT / "vis_step"


def _safe_file_stem(value: str) -> str:
    cleaned = [
        ch if ch.isalnum() or ch in {"_", "-"} else "_"
        for ch in str(value)
    ]
    stem = "".join(cleaned).strip("_")
    return stem or "benchmark"


def _top_grid_mask(grid: np.ndarray, top_frac: float) -> Tuple[np.ndarray, float, float]:
    grid = np.asarray(grid, dtype=np.float64)
    finite = np.isfinite(grid)
    positive = grid[finite & (grid > 1e-12)]
    if positive.size == 0:
        return np.zeros_like(grid, dtype=bool), 0.0, 0.0
    count = max(1, int(math.ceil(grid.size * float(top_frac))))
    count = min(count, positive.size)
    threshold = float(np.partition(positive, -count)[-count])
    return grid >= threshold, threshold, float(positive.max())


def _snapshot_cost_grids(
    benchmark: Benchmark,
    positions: np.ndarray,
    sizes_np: np.ndarray,
    net_weights_np: np.ndarray | None = None,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    density_eval = _IncrementalDensityCost(benchmark, positions, sizes_np)
    density_grid = density_eval.occupied / max(density_eval.grid_area, EPS)

    pin_owner_np, pin_offset_np, pin_net_np, port_pos_np = _build_pin_arrays_np(
        benchmark
    )
    num_nets = len(benchmark.net_pin_nodes)
    nets_pin_indices = _net_pin_index_table_cached(
        benchmark, pin_net_np, num_nets
    )
    congestion_eval = _IncrementalCongestionCost(
        benchmark,
        positions,
        sizes_np,
        pin_owner_np,
        pin_offset_np,
        nets_pin_indices,
        port_pos_np,
        net_weights_np=net_weights_np,
    )
    congestion_grid = np.maximum(
        congestion_eval.v_route + congestion_eval.v_macro,
        congestion_eval.h_route + congestion_eval.h_macro,
    )
    return density_grid, congestion_grid


def _draw_hot_grid_edges(ax, mask: np.ndarray, cw: float, ch: float, color: str):
    if mask is None or not mask.any():
        return
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    rows, cols = mask.shape
    grid_w = cw / max(1, cols)
    grid_h = ch / max(1, rows)
    patches = [
        Rectangle((c * grid_w, r * grid_h), grid_w, grid_h)
        for r, c in zip(*np.nonzero(mask))
    ]
    if patches:
        ax.add_collection(
            PatchCollection(
                patches,
                facecolor="none",
                edgecolor=color,
                linewidth=0.55,
                alpha=0.95,
                zorder=6,
            )
        )


def _dump_global_step_snapshot(
    benchmark: Benchmark,
    positions,
    sizes_np: np.ndarray,
    fixed_np: np.ndarray,
    iter_idx: int,
    step_idx: int,
    label: str,
    net_weights_np: np.ndarray | None = None,
) -> None:
    if not _env_flag("VIBECPP_DUMP_VIS_STEPS", False):
        return

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Patch

        pos_np = (
            positions.detach().cpu().numpy().astype(np.float64)
            if isinstance(positions, torch.Tensor)
            else np.asarray(positions, dtype=np.float64)
        )
        sizes_eval = np.asarray(sizes_np, dtype=np.float64)
        fixed_eval = np.asarray(fixed_np, dtype=bool)

        density_grid, congestion_grid = _snapshot_cost_grids(
            benchmark, pos_np, sizes_eval, net_weights_np=net_weights_np
        )
        density_mask, density_threshold, density_max = _top_grid_mask(
            density_grid, DENSITY_TOP_FRAC
        )
        congestion_mask, congestion_threshold, congestion_max = _top_grid_mask(
            congestion_grid, CONGESTION_TOP_FRAC
        )

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        n_hard = int(benchmark.num_hard_macros)
        n_total = int(benchmark.num_macros)
        aspect = ch / max(cw, EPS)
        fig_w = 10.5
        fig_h = min(12.0, max(6.0, fig_w * aspect))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
        ax.set_facecolor("#f8fafc")
        ax.set_aspect("equal")
        margin = max(cw, ch) * 0.025
        ax.set_xlim(-margin, cw + margin)
        ax.set_ylim(-margin, ch + margin)
        ax.add_patch(
            mpatches.Rectangle(
                (0, 0),
                cw,
                ch,
                facecolor="#ffffff",
                edgecolor="#0f172a",
                linewidth=1.0,
                zorder=1,
            )
        )

        if density_mask.any():
            dens_cmap = mcolors.LinearSegmentedColormap.from_list(
                "vibe_density_hot", ["#fed7aa", "#fb923c", "#ea580c"], N=64
            )
            dens = np.ma.array(density_grid, mask=~density_mask)
            ax.imshow(
                dens,
                origin="lower",
                extent=[0, cw, 0, ch],
                cmap=dens_cmap,
                alpha=0.46,
                interpolation="nearest",
                vmin=density_threshold,
                vmax=max(density_max, density_threshold * 1.01),
                zorder=2,
            )

        if congestion_mask.any():
            cong_cmap = mcolors.LinearSegmentedColormap.from_list(
                "vibe_congestion_hot", ["#bae6fd", "#818cf8", "#a855f7"], N=64
            )
            cong = np.ma.array(congestion_grid, mask=~congestion_mask)
            ax.imshow(
                cong,
                origin="lower",
                extent=[0, cw, 0, ch],
                cmap=cong_cmap,
                alpha=0.36,
                interpolation="nearest",
                vmin=congestion_threshold,
                vmax=max(congestion_max, congestion_threshold * 1.01),
                zorder=3,
            )

        _draw_hot_grid_edges(ax, density_mask, cw, ch, "#ea580c")
        _draw_hot_grid_edges(ax, congestion_mask, cw, ch, "#7c3aed")

        soft_patches = []
        hard_movable_patches = []
        hard_fixed_patches = []
        for i in range(n_total):
            w = float(sizes_eval[i, 0])
            h = float(sizes_eval[i, 1])
            if w <= 0.0 or h <= 0.0:
                continue
            cx = float(pos_np[i, 0])
            cy = float(pos_np[i, 1])
            rect = mpatches.Rectangle((cx - w / 2.0, cy - h / 2.0), w, h)
            if i >= n_hard:
                soft_patches.append(rect)
            elif fixed_eval[i]:
                hard_fixed_patches.append(rect)
            else:
                hard_movable_patches.append(rect)

        if soft_patches:
            ax.add_collection(
                PatchCollection(
                    soft_patches,
                    facecolor="#67e8f9",
                    edgecolor="#0891b2",
                    linewidth=0.18,
                    alpha=0.42,
                    zorder=4,
                )
            )
        if hard_movable_patches:
            ax.add_collection(
                PatchCollection(
                    hard_movable_patches,
                    facecolor="#2563eb",
                    edgecolor="#1e3a8a",
                    linewidth=0.35,
                    alpha=0.74,
                    zorder=5,
                )
            )
        if hard_fixed_patches:
            ax.add_collection(
                PatchCollection(
                    hard_fixed_patches,
                    facecolor="#64748b",
                    edgecolor="#334155",
                    linewidth=0.35,
                    alpha=0.86,
                    zorder=5,
                )
            )

        benchmark_name = str(getattr(benchmark, "name", "benchmark"))
        ax.set_title(
            f"{benchmark_name} iter {iter_idx} step {step_idx}: {label}",
            fontsize=11,
            color="#0f172a",
            pad=10,
        )
        ax.set_xlabel("x", fontsize=8, color="#334155")
        ax.set_ylabel("y", fontsize=8, color="#334155")
        ax.tick_params(axis="both", labelsize=7, colors="#64748b")
        ax.legend(
            handles=[
                Patch(facecolor="#2563eb", edgecolor="#1e3a8a", label="hard"),
                Patch(facecolor="#64748b", edgecolor="#334155", label="fixed"),
                Patch(facecolor="#67e8f9", edgecolor="#0891b2", label="soft"),
                Patch(facecolor="#fb923c", edgecolor="#ea580c", label="hot density"),
                Patch(facecolor="#818cf8", edgecolor="#7c3aed", label="hot congestion"),
            ],
            loc="upper right",
            fontsize=7,
            framealpha=0.92,
        )
        ax.text(
            0.01,
            0.01,
            f"density top {DENSITY_TOP_FRAC:.0%}: >= {density_threshold:.3f} "
            f"(max {density_max:.3f})\n"
            f"congestion top {CONGESTION_TOP_FRAC:.0%}: >= {congestion_threshold:.3f} "
            f"(max {congestion_max:.3f})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            color="#334155",
            bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.88},
            zorder=7,
        )

        out_dir = _vis_step_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_file_stem(benchmark_name)
        out_path = out_dir / f"{stem}_iter{int(iter_idx)}_step{int(step_idx)}.png"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        print(f"[vis-step] wrote {out_path}", flush=True)
    except Exception as exc:
        if not getattr(benchmark, "_vibecpp_vis_step_error_reported", False):
            setattr(benchmark, "_vibecpp_vis_step_error_reported", True)
            print(f"[vis-step] snapshot disabled after error: {exc}", flush=True)


def _wl_per_net(
    values: torch.Tensor, pin_net: torch.Tensor, num_nets: int, gamma: float
) -> torch.Tensor:
    scaled = values / gamma
    with torch.no_grad():
        max_per_net = torch.full(
            (num_nets,), -float("inf"), dtype=scaled.dtype, device=scaled.device
        )
        max_per_net = max_per_net.scatter_reduce(
            0, pin_net, scaled.detach(), reduce="amax", include_self=True
        )
        max_per_net = torch.where(
            torch.isfinite(max_per_net),
            max_per_net,
            torch.zeros_like(max_per_net),
        )
    shifted = scaled - max_per_net[pin_net]
    exp_term = torch.exp(shifted)
    sum_exp = torch.zeros(num_nets, dtype=scaled.dtype, device=scaled.device)
    sum_exp = sum_exp.scatter_add(0, pin_net, exp_term)
    sum_exp = torch.clamp(sum_exp, min=EPS)
    return gamma * (max_per_net + torch.log(sum_exp))

def _wirelength(
    all_pos: torch.Tensor,
    pin_owner: torch.Tensor,
    pin_offset: torch.Tensor,
    pin_net: torch.Tensor,
    num_nets: int,
    gamma: float,
    net_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if pin_owner.numel() == 0:
        return all_pos.new_zeros(())
    px = all_pos[pin_owner, 0] + pin_offset[:, 0]
    py = all_pos[pin_owner, 1] + pin_offset[:, 1]
    per_net = (
        _wl_per_net(px, pin_net, num_nets, gamma)
        + _wl_per_net(-px, pin_net, num_nets, gamma)
        + _wl_per_net(py, pin_net, num_nets, gamma)
        + _wl_per_net(-py, pin_net, num_nets, gamma)
    )
    if net_weights is not None and net_weights.numel() == num_nets:
        per_net = per_net * net_weights.to(dtype=per_net.dtype)
    return per_net.sum()

def _density_grid_gauss(
    pos: torch.Tensor,
    sizes: torch.Tensor,
    bin_x: torch.Tensor,
    bin_y: torch.Tensor,
    bw: float,
    bh: float,
) -> torch.Tensor:
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    area = sizes[:, 0] * sizes[:, 1]
    sigma_x = torch.clamp(half_w / 1.7320508, min=bw)
    sigma_y = torch.clamp(half_h / 1.7320508, min=bh)
    dx = bin_x[None, :] - pos[:, 0:1]
    dy = bin_y[None, :] - pos[:, 1:2]
    wx = torch.exp(-0.5 * (dx / sigma_x[:, None]) ** 2)
    wy = torch.exp(-0.5 * (dy / sigma_y[:, None]) ** 2)
    nx = wx.sum(dim=1, keepdim=True).clamp(min=EPS)
    ny = wy.sum(dim=1, keepdim=True).clamp(min=EPS)
    wx = wx / nx
    wy = wy / ny
    density_area = torch.einsum("mi,mj,m->ji", wx, wy, area)
    return density_area / (bw * bh)


def _density_grid_exact_rect_edges(
    pos: torch.Tensor,
    sizes: torch.Tensor,
    bx_l: torch.Tensor,
    bx_r: torch.Tensor,
    by_b: torch.Tensor,
    by_t: torch.Tensor,
    bin_area: float,
) -> torch.Tensor:
    xl = pos[:, 0:1] - sizes[:, 0:1] / 2
    xr = pos[:, 0:1] + sizes[:, 0:1] / 2
    yb = pos[:, 1:2] - sizes[:, 1:2] / 2
    yt = pos[:, 1:2] + sizes[:, 1:2] / 2

    ox = torch.relu(
        torch.minimum(xr, bx_r[None, :]) - torch.maximum(xl, bx_l[None, :])
    )
    oy = torch.relu(
        torch.minimum(yt, by_t[None, :]) - torch.maximum(yb, by_b[None, :])
    )
    density_area = torch.einsum("mx,my->yx", ox, oy)
    return density_area / bin_area

def _routing_congestion_penalty(
    all_pos: torch.Tensor,
    pin_owner: torch.Tensor,
    pin_offset: torch.Tensor,
    pin_net: torch.Tensor,
    num_nets: int,
    gamma: float,
    bin_x: torch.Tensor,
    bin_y: torch.Tensor,
    cw: float,
    ch: float,
    net_weights: torch.Tensor | None = None,
    top_frac: float = CONGESTION_TOP_FRAC,
) -> torch.Tensor:
    if pin_owner.numel() == 0 or num_nets == 0:
        return all_pos.new_zeros(())

    px = all_pos[pin_owner, 0] + pin_offset[:, 0]
    py = all_pos[pin_owner, 1] + pin_offset[:, 1]

    xmax = _wl_per_net(px, pin_net, num_nets, gamma)
    xmin = -_wl_per_net(-px, pin_net, num_nets, gamma)
    ymax = _wl_per_net(py, pin_net, num_nets, gamma)
    ymin = -_wl_per_net(-py, pin_net, num_nets, gamma)

    dx = (xmax - xmin).detach()
    dy = (ymax - ymin).detach()

    n_bins_x = bin_x.shape[0]
    n_bins_y = bin_y.shape[0]
    bw = cw / n_bins_x
    bh = ch / n_bins_y

    valid = (dx > bw * 0.1) | (dy > bh * 0.1)
    if not valid.any():
        return all_pos.new_zeros(())

    dx = dx[valid]
    dy = dy[valid]
    xmin = xmin[valid]
    xmax = xmax[valid]
    ymin = ymin[valid]
    ymax = ymax[valid]

    area = torch.clamp(dx * dy, min=bw * bh)
    weight = (dx + dy) / area
    if net_weights is not None and net_weights.numel() == num_nets:
        weight = weight * net_weights[valid].to(dtype=weight.dtype)

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    sigma_x = torch.clamp(dx / 3.464, min=bw)
    sigma_y = torch.clamp(dy / 3.464, min=bh)

    diff_x = bin_x[None, :] - cx[:, None]
    diff_y = bin_y[None, :] - cy[:, None]

    exp_x = torch.exp(-0.5 * (diff_x / sigma_x[:, None]) ** 2)
    exp_y = torch.exp(-0.5 * (diff_y / sigma_y[:, None]) ** 2)

    norm_x = sigma_x * 2.5066
    norm_y = sigma_y * 2.5066

    px_grid = exp_x / norm_x[:, None]
    py_grid = exp_y / norm_y[:, None]

    cong_grid = torch.matmul((weight[:, None] * py_grid).T, px_grid)
    cong_flat = cong_grid.reshape(-1)
    top_k = max(1, int(n_bins_x * n_bins_y * top_frac))
    hot_cong = torch.topk(cong_flat, top_k).values
    return (hot_cong * hot_cong).mean()


def _routing_congestion_penalty_abu(
    all_pos: torch.Tensor,
    macro_pos: torch.Tensor,
    macro_sizes: torch.Tensor,
    pin_owner: torch.Tensor,
    pin_offset: torch.Tensor,
    pin_net: torch.Tensor,
    num_nets: int,
    gamma: float,
    bin_x: torch.Tensor,
    bin_y: torch.Tensor,
    bx_l: torch.Tensor,
    bx_r: torch.Tensor,
    by_b: torch.Tensor,
    by_t: torch.Tensor,
    bw: float,
    bh: float,
    cw: float,
    ch: float,
    hroutes_per_micron: float,
    vroutes_per_micron: float,
    hrouting_alloc: float,
    vrouting_alloc: float,
    macro_blockage_weight: float = CONG_MACRO_BLOCKAGE_WEIGHT,
    net_weights: torch.Tensor | None = None,
    top_frac: float = CONGESTION_TOP_FRAC,
) -> torch.Tensor:
    """ABU-style congestion penalty that includes macro routing blockage.

    Mirrors the official ``get_congestion_cost`` formulation:

      v_total[r, c] = v_route_demand[r, c] + v_macro_blockage[r, c]
      h_total[r, c] = h_route_demand[r, c] + h_macro_blockage[r, c]
      cost = mean( top_{2*N*top_frac}( {v_total} U {h_total} )^2 )

    The macro blockage term is the key addition. Differentiable through both
    macro positions (via overlap areas) and through pin owners (RUDY routing).
    """
    n_bins_y = bin_y.shape[0]
    n_bins_x = bin_x.shape[0]
    grid_v_routes = max(bw * float(vroutes_per_micron), EPS)
    grid_h_routes = max(bh * float(hroutes_per_micron), EPS)

    # --- 1. Routing demand from net bounding boxes ---
    if pin_owner.numel() > 0 and num_nets > 0:
        px = all_pos[pin_owner, 0] + pin_offset[:, 0]
        py = all_pos[pin_owner, 1] + pin_offset[:, 1]
        xmax = _wl_per_net(px, pin_net, num_nets, gamma)
        xmin = -_wl_per_net(-px, pin_net, num_nets, gamma)
        ymax = _wl_per_net(py, pin_net, num_nets, gamma)
        ymin = -_wl_per_net(-py, pin_net, num_nets, gamma)

        dx_bbox = (xmax - xmin).detach()
        dy_bbox = (ymax - ymin).detach()

        valid = (dx_bbox > bw * 0.05) | (dy_bbox > bh * 0.05)
        if valid.any():
            xmin_v = xmin[valid]
            xmax_v = xmax[valid]
            ymin_v = ymin[valid]
            ymax_v = ymax[valid]
            if net_weights is not None and net_weights.numel() == num_nets:
                w_net = net_weights[valid].to(dtype=all_pos.dtype)
            else:
                w_net = torch.ones(
                    xmin_v.shape[0], dtype=all_pos.dtype, device=all_pos.device
                )

            # Soft x-overlap fraction with each column bin (smooth via exact rect)
            ox_net = F.relu(
                torch.minimum(xmax_v[:, None], bx_r[None, :])
                - torch.maximum(xmin_v[:, None], bx_l[None, :])
            ) / bw  # (n_valid, n_bins_x)
            oy_net = F.relu(
                torch.minimum(ymax_v[:, None], by_t[None, :])
                - torch.maximum(ymin_v[:, None], by_b[None, :])
            ) / bh  # (n_valid, n_bins_y)

            cols_span = ox_net.sum(dim=1).clamp(min=1.0)
            rows_span = oy_net.sum(dim=1).clamp(min=1.0)
            # V-demand per bin = weight / cols_span / grid_v_routes when bbox covers the bin
            wv = w_net / (cols_span * grid_v_routes)
            wh = w_net / (rows_span * grid_h_routes)
            # Aggregate to grids
            v_route = torch.einsum("nc,nr,n->rc", ox_net, oy_net, wv)
            h_route = torch.einsum("nc,nr,n->rc", ox_net, oy_net, wh)
        else:
            v_route = all_pos.new_zeros((n_bins_y, n_bins_x))
            h_route = all_pos.new_zeros((n_bins_y, n_bins_x))
    else:
        v_route = all_pos.new_zeros((n_bins_y, n_bins_x))
        h_route = all_pos.new_zeros((n_bins_y, n_bins_x))

    # --- 2. Macro blockage contribution (differentiable in macro_pos) ---
    if macro_blockage_weight > 0.0 and macro_pos.shape[0] > 0:
        xl = macro_pos[:, 0:1] - macro_sizes[:, 0:1] / 2
        xr = macro_pos[:, 0:1] + macro_sizes[:, 0:1] / 2
        yb = macro_pos[:, 1:2] - macro_sizes[:, 1:2] / 2
        yt = macro_pos[:, 1:2] + macro_sizes[:, 1:2] / 2
        ox_m = F.relu(
            torch.minimum(xr, bx_r[None, :])
            - torch.maximum(xl, bx_l[None, :])
        )  # absolute width overlap (n_macros, n_bins_x)
        oy_m = F.relu(
            torch.minimum(yt, by_t[None, :])
            - torch.maximum(yb, by_b[None, :])
        )  # absolute height overlap (n_macros, n_bins_y)

        oy_frac = (oy_m / bh).clamp(max=1.0)
        ox_frac = (ox_m / bw).clamp(max=1.0)
        # v_macro[r,c] = sum over macros of x_overlap * (y-touch indicator) * vrouting_alloc / grid_v_routes
        v_macro = (
            torch.einsum("mc,mr->rc", ox_m, oy_frac)
            * (float(vrouting_alloc) / grid_v_routes)
        )
        h_macro = (
            torch.einsum("mc,mr->rc", ox_frac, oy_m)
            * (float(hrouting_alloc) / grid_h_routes)
        )
        v_total = v_route + macro_blockage_weight * v_macro
        h_total = h_route + macro_blockage_weight * h_macro
    else:
        v_total = v_route
        h_total = h_route

    combined = torch.cat([v_total.reshape(-1), h_total.reshape(-1)])
    top_k = max(1, int(2 * n_bins_x * n_bins_y * float(top_frac)))
    hot = torch.topk(combined, top_k).values
    return (hot * hot).mean()


PERTURB_INTERVAL_START = 50
PERTURB_INTERVAL_END = 50
PERTURB_SCALE_START = 2.0
PERTURB_SCALE_END = 0.35
PERTURB_DECAY_POWER = 1.25
PERTURB_MIN_RECOVERY_ITERS = 50

ADAM_EARLY_STOP_WINDOW = 10
ADAM_EARLY_STOP_THRESHOLD = 0.01  # 1% spread of last N prev_move values
ADAM_EARLY_STOP_TOTAL_MOVE_FRAC = 1e-4


def _adam_converged(move_history: deque, max_avg_move: float | None = None) -> bool:
    """Return True when the spread of the last N prev_move values is small.

    Specifically: (max - min) / mean < ADAM_EARLY_STOP_THRESHOLD.
    Requires the window to be full.
    """
    if len(move_history) < ADAM_EARLY_STOP_WINDOW:
        return False
    hi = max(move_history)
    lo = min(move_history)
    avg = sum(move_history) / len(move_history)
    if avg <= EPS:
        return False
    if max_avg_move is not None and avg > max_avg_move:
        return False
    return (hi - lo) / avg < ADAM_EARLY_STOP_THRESHOLD


def _perturb_progress(iter_no: int, iters: int) -> float:
    return min(1.0, max(0.0, iter_no / max(1, iters)))


def _perturb_interval(progress: float) -> int:
    return max(
        1,
        int(
            round(
                PERTURB_INTERVAL_START
                + (PERTURB_INTERVAL_END - PERTURB_INTERVAL_START)
                * (progress ** PERTURB_DECAY_POWER)
            )
        ),
    )


def _perturb_scale(progress: float) -> float:
    remaining = (1.0 - progress) ** PERTURB_DECAY_POWER
    return PERTURB_SCALE_END + (PERTURB_SCALE_START - PERTURB_SCALE_END) * remaining


def _adam_phase(
    benchmark: Benchmark,
    positions_np: np.ndarray,
    movable_mask_np: np.ndarray,
    sizes_np: np.ndarray,
    cw: float,
    ch: float,
    n_total: int,
    device: torch.device,
    iters: int,
    lr_frac: float,
    lambda_start: float,
    lambda_end: float,
    density_target_slack: float = 1.0,
    cong_weight: float = 0.5,
    net_weights_np: np.ndarray | None = None,
    label: str = "adam_phase",
    perturbation: bool = True,
    macro_blockage_weight: float = CONG_MACRO_BLOCKAGE_WEIGHT,
) -> np.ndarray:
    print(
        f"[adam] {label} starting | iters={iters} "
        f"perturbation={'on' if perturbation else 'off'}",
        flush=True,
    )
    canvas = max(cw, ch)

    pin_owner_np, pin_offset_np, pin_net_np, port_pos_np = _build_pin_arrays_np(
        benchmark
    )
    pin_owner = torch.tensor(pin_owner_np, dtype=torch.long, device=device)
    pin_offset = torch.tensor(pin_offset_np, dtype=torch.float32, device=device)
    pin_net = torch.tensor(pin_net_np, dtype=torch.long, device=device)
    port_pos = torch.tensor(port_pos_np, dtype=torch.float32, device=device)
    num_nets = len(benchmark.net_pin_nodes)
    n_ports = port_pos.shape[0]
    net_weights = None
    if net_weights_np is not None and len(net_weights_np) == num_nets:
        net_weights = torch.tensor(
            net_weights_np, dtype=torch.float32, device=device
        )

    sizes_t = torch.tensor(sizes_np, dtype=torch.float32, device=device)
    movable_t = torch.tensor(movable_mask_np, dtype=torch.bool, device=device)

    pos_var = (
        torch.tensor(positions_np, dtype=torch.float32, device=device)
        .clone()
        .requires_grad_(True)
    )
    init_pos_t = (
        torch.tensor(positions_np, dtype=torch.float32, device=device)
        .clone()
        .detach()
    )

    n_bins_x = max(8, int(benchmark.grid_cols))
    n_bins_y = max(8, int(benchmark.grid_rows))
    bw = cw / n_bins_x
    bh = ch / n_bins_y
    bin_x = (
        torch.arange(n_bins_x, dtype=torch.float32, device=device) * bw + bw / 2
    )
    bin_y = (
        torch.arange(n_bins_y, dtype=torch.float32, device=device) * bh + bh / 2
    )
    bx_l = torch.arange(n_bins_x, dtype=torch.float32, device=device) * bw
    bx_r = bx_l + bw
    by_b = torch.arange(n_bins_y, dtype=torch.float32, device=device) * bh
    by_t = by_b + bh

    hroutes_pm = float(getattr(benchmark, "hroutes_per_micron", 0.0))
    vroutes_pm = float(getattr(benchmark, "vroutes_per_micron", 0.0))
    use_abu_cong = hroutes_pm > 0.0 and vroutes_pm > 0.0

    total_area = float((sizes_np[:, 0] * sizes_np[:, 1]).sum())
    target = (total_area / (cw * ch)) * density_target_slack

    half_w = sizes_t[:, 0] / 2
    half_h = sizes_t[:, 1] / 2

    opt = torch.optim.Adam([pos_var], lr=canvas * lr_frac)
    gamma_start = canvas * 0.05
    gamma_end = canvas * 0.005

    prev_total_move = 0.0
    move_history: deque = deque(maxlen=ADAM_EARLY_STOP_WINDOW)
    movable_count = max(1, int(np.count_nonzero(movable_mask_np)))
    early_stop_move_limit = (
        canvas * movable_count * ADAM_EARLY_STOP_TOTAL_MOVE_FRAC
    )
    ran_iters = 0
    next_perturb_iter = PERTURB_INTERVAL_START
    for it in range(iters):
        ran_iters = it + 1
        frac = min(1.0, it / max(1, iters - 1))
        gamma = gamma_start * (gamma_end / gamma_start) ** frac
        lam = lambda_start * (lambda_end / lambda_start) ** frac

        with torch.no_grad():
            pre_opt_pos = pos_var.data.clone()

        opt.zero_grad(set_to_none=True)
        all_pos = (
            torch.cat([pos_var, port_pos], dim=0) if n_ports > 0 else pos_var
        )
        wl = _wirelength(
            all_pos, pin_owner, pin_offset, pin_net, num_nets, gamma,
            net_weights=net_weights,
        )
        density = _density_grid_gauss(pos_var, sizes_t, bin_x, bin_y, bw, bh)
        density_pen = (F.relu(density - target) ** 2).sum()

        if use_abu_cong and not USE_LEGACY_CONG_PENALTY:
            cong_pen = _routing_congestion_penalty_abu(
                all_pos, pos_var, sizes_t,
                pin_owner, pin_offset, pin_net, num_nets, gamma,
                bin_x, bin_y, bx_l, bx_r, by_b, by_t,
                bw, bh, cw, ch,
                hroutes_pm, vroutes_pm,
                float(HROUTING_ALLOC), float(VROUTING_ALLOC),
                macro_blockage_weight=float(macro_blockage_weight),
                net_weights=net_weights,
            )
        else:
            cong_pen = _routing_congestion_penalty(
                all_pos, pin_owner, pin_offset, pin_net, num_nets, gamma,
                bin_x, bin_y, cw, ch, net_weights=net_weights,
            )

        loss = wl + lam * (density_pen + cong_weight * cong_pen)
        loss.backward()
        with torch.no_grad():
            if pos_var.grad is not None:
                pos_var.grad[~movable_t] = 0
        opt.step()
        with torch.no_grad():
            pos_var.data[:, 0].clamp_(half_w + GAP, cw - half_w - GAP)
            pos_var.data[:, 1].clamp_(half_h + GAP, ch - half_h - GAP)
            pos_var.data[~movable_t] = init_pos_t[~movable_t]

        with torch.no_grad():
            diffs = pos_var.data - pre_opt_pos
            per_macro = torch.norm(diffs, dim=1)
            per_macro[~movable_t] = 0.0
            prev_total_move = float(per_macro.sum())
        move_history.append(prev_total_move)

        if _adam_converged(move_history, early_stop_move_limit):
            print(
                f"[adam] {label} early stop @ iter={ran_iters}/{iters} | "
                f"loss={float(loss):.4f} WL={float(wl):.4f} "
                f"density={float(density_pen):.4f} cong={float(cong_pen):.4f}",
                flush=True,
            )
            break

        if perturbation:
            iter_no = it + 1
            can_recover = it + 1 + PERTURB_MIN_RECOVERY_ITERS < iters
            if (
                can_recover
                and iter_no >= next_perturb_iter
                and prev_total_move > 0.0
            ):
                perturb_progress = _perturb_progress(iter_no, iters)
                perturb_scale = _perturb_scale(perturb_progress)
                next_perturb_iter = iter_no + _perturb_interval(perturb_progress)
                with torch.no_grad():
                    target_total = perturb_scale * prev_total_move
                    rnd = torch.randn_like(pos_var.data)
                    rnd[~movable_t] = 0.0
                    cur_total = float(torch.norm(rnd, dim=1).sum())
                    if cur_total > EPS:
                        scale = target_total / cur_total
                        pos_var.data.add_(rnd, alpha=scale)
                        pos_var.data[:, 0].clamp_(
                            half_w + GAP, cw - half_w - GAP
                        )
                        pos_var.data[:, 1].clamp_(
                            half_h + GAP, ch - half_h - GAP
                        )
                        pos_var.data[~movable_t] = init_pos_t[~movable_t]
                        # perturbation invalidates the convergence history
                        move_history.clear()

    print(
        f"[adam] {label} done | ran {ran_iters}/{iters} iters | "
        f"loss={float(loss):.4f} WL={float(wl):.4f} "
        f"density={float(density_pen):.4f} cong={float(cong_pen):.4f}",
        flush=True,
    )
    return pos_var.detach().cpu().numpy().astype(np.float64)

def _adam_exact_density_phase(
    benchmark: Benchmark,
    positions_np: np.ndarray,
    movable_mask_np: np.ndarray,
    sizes_np: np.ndarray,
    cw: float,
    ch: float,
    n_total: int,
    device: torch.device,
    iters: int,
    lr_frac: float,
    lambda_start: float,
    lambda_end: float,
    density_target: float,
    overflow_weight: float,
    cong_weight: float = 0.5,
    net_weights_np: np.ndarray | None = None,
    top_frac: float = DENSITY_TOP_FRAC,
    label: str = "adam_exact_density_phase",
    perturbation: bool = True,
    macro_blockage_weight: float = CONG_MACRO_BLOCKAGE_WEIGHT,
) -> np.ndarray:
    print(
        f"[adam] {label} starting | iters={iters} "
        f"perturbation={'on' if perturbation else 'off'}",
        flush=True,
    )
    canvas = max(cw, ch)

    pin_owner_np, pin_offset_np, pin_net_np, port_pos_np = _build_pin_arrays_np(
        benchmark
    )
    pin_owner = torch.tensor(pin_owner_np, dtype=torch.long, device=device)
    pin_offset = torch.tensor(pin_offset_np, dtype=torch.float32, device=device)
    pin_net = torch.tensor(pin_net_np, dtype=torch.long, device=device)
    port_pos = torch.tensor(port_pos_np, dtype=torch.float32, device=device)
    num_nets = len(benchmark.net_pin_nodes)
    n_ports = port_pos.shape[0]
    net_weights = None
    if net_weights_np is not None and len(net_weights_np) == num_nets:
        net_weights = torch.tensor(
            net_weights_np, dtype=torch.float32, device=device
        )

    sizes_t = torch.tensor(sizes_np, dtype=torch.float32, device=device)
    movable_t = torch.tensor(movable_mask_np, dtype=torch.bool, device=device)
    pos_var = (
        torch.tensor(positions_np, dtype=torch.float32, device=device)
        .clone()
        .requires_grad_(True)
    )
    init_pos_t = pos_var.detach().clone()

    n_bins_x = max(8, int(benchmark.grid_cols))
    n_bins_y = max(8, int(benchmark.grid_rows))
    top_k = max(1, int(n_bins_x * n_bins_y * top_frac))
    bw = cw / n_bins_x
    bh = ch / n_bins_y
    bx_l = torch.arange(n_bins_x, dtype=torch.float32, device=device) * bw
    bx_r = bx_l + bw
    by_b = torch.arange(n_bins_y, dtype=torch.float32, device=device) * bh
    by_t = by_b + bh
    bin_area = bw * bh
    bin_x = torch.linspace(0, cw, n_bins_x, device=device) + cw / (2 * n_bins_x)
    bin_y = torch.linspace(0, ch, n_bins_y, device=device) + ch / (2 * n_bins_y)

    hroutes_pm = float(getattr(benchmark, "hroutes_per_micron", 0.0))
    vroutes_pm = float(getattr(benchmark, "vroutes_per_micron", 0.0))
    use_abu_cong = hroutes_pm > 0.0 and vroutes_pm > 0.0

    half_w = sizes_t[:, 0] / 2
    half_h = sizes_t[:, 1] / 2
    opt = torch.optim.Adam([pos_var], lr=canvas * lr_frac)
    gamma_start = canvas * 0.02
    gamma_end = canvas * 0.002

    prev_total_move = 0.0
    move_history: deque = deque(maxlen=ADAM_EARLY_STOP_WINDOW)
    movable_count = max(1, int(np.count_nonzero(movable_mask_np)))
    early_stop_move_limit = (
        canvas * movable_count * ADAM_EARLY_STOP_TOTAL_MOVE_FRAC
    )
    ran_iters = 0
    next_perturb_iter = PERTURB_INTERVAL_START
    for it in range(iters):
        ran_iters = it + 1
        frac = min(1.0, it / max(1, iters - 1))
        gamma = gamma_start * (gamma_end / gamma_start) ** frac
        lam = lambda_start * (lambda_end / lambda_start) ** frac

        with torch.no_grad():
            pre_opt_pos = pos_var.data.clone()

        opt.zero_grad(set_to_none=True)
        all_pos = (
            torch.cat([pos_var, port_pos], dim=0) if n_ports > 0 else pos_var
        )
        wl = _wirelength(
            all_pos, pin_owner, pin_offset, pin_net, num_nets, gamma,
            net_weights=net_weights,
        )
        density = _density_grid_exact_rect_edges(
            pos_var, sizes_t, bx_l, bx_r, by_b, by_t, bin_area
        )
        density_flat = density.reshape(-1)
        hot_density = torch.topk(density_flat, top_k).values
        density_pen = (hot_density * hot_density).mean()
        density_pen = density_pen + overflow_weight * (
            F.relu(density_flat - density_target) ** 2
        ).mean()

        if use_abu_cong and not USE_LEGACY_CONG_PENALTY:
            cong_pen = _routing_congestion_penalty_abu(
                all_pos, pos_var, sizes_t,
                pin_owner, pin_offset, pin_net, num_nets, gamma,
                bin_x, bin_y, bx_l, bx_r, by_b, by_t,
                bw, bh, cw, ch,
                hroutes_pm, vroutes_pm,
                float(HROUTING_ALLOC), float(VROUTING_ALLOC),
                macro_blockage_weight=float(macro_blockage_weight),
                net_weights=net_weights,
            )
        else:
            cong_pen = _routing_congestion_penalty(
                all_pos, pin_owner, pin_offset, pin_net, num_nets, gamma,
                bin_x, bin_y, cw, ch, net_weights=net_weights,
            )

        loss = wl + lam * (density_pen + cong_weight * cong_pen)
        loss.backward()
        with torch.no_grad():
            if pos_var.grad is not None:
                pos_var.grad[~movable_t] = 0
        opt.step()
        with torch.no_grad():
            pos_var.data[:, 0].clamp_(half_w + GAP, cw - half_w - GAP)
            pos_var.data[:, 1].clamp_(half_h + GAP, ch - half_h - GAP)
            pos_var.data[~movable_t] = init_pos_t[~movable_t]

        with torch.no_grad():
            diffs = pos_var.data - pre_opt_pos
            per_macro = torch.norm(diffs, dim=1)
            per_macro[~movable_t] = 0.0
            prev_total_move = float(per_macro.sum())
        move_history.append(prev_total_move)

        if _adam_converged(move_history, early_stop_move_limit):
            print(
                f"[adam] {label} early stop @ iter={ran_iters}/{iters} | "
                f"loss={float(loss):.4f} WL={float(wl):.4f} "
                f"density={float(density_pen):.4f} cong={float(cong_pen):.4f}",
                flush=True,
            )
            break

        if perturbation:
            iter_no = it + 1
            can_recover = it + 1 + PERTURB_MIN_RECOVERY_ITERS < iters
            if (
                can_recover
                and iter_no >= next_perturb_iter
                and prev_total_move > 0.0
            ):
                perturb_progress = _perturb_progress(iter_no, iters)
                perturb_scale = _perturb_scale(perturb_progress)
                next_perturb_iter = iter_no + _perturb_interval(perturb_progress)
                with torch.no_grad():
                    target_total = perturb_scale * prev_total_move
                    rnd = torch.randn_like(pos_var.data)
                    rnd[~movable_t] = 0.0
                    cur_total = float(torch.norm(rnd, dim=1).sum())
                    if cur_total > EPS:
                        scale = target_total / cur_total
                        pos_var.data.add_(rnd, alpha=scale)
                        pos_var.data[:, 0].clamp_(
                            half_w + GAP, cw - half_w - GAP
                        )
                        pos_var.data[:, 1].clamp_(
                            half_h + GAP, ch - half_h - GAP
                        )
                        pos_var.data[~movable_t] = init_pos_t[~movable_t]
                        # perturbation invalidates the convergence history
                        move_history.clear()

    print(
        f"[adam] {label} done | ran {ran_iters}/{iters} iters | "
        f"loss={float(loss):.4f} WL={float(wl):.4f} "
        f"density={float(density_pen):.4f} cong={float(cong_pen):.4f}",
        flush=True,
    )
    return pos_var.detach().cpu().numpy().astype(np.float64)

def _overlaps_any(x, y, w, h, placed, gap):
    half_w = w / 2
    half_h = h / 2
    for (xl, xr, yb, yt) in placed:
        if (
            x + half_w > xl - gap
            and x - half_w < xr + gap
            and y + half_h > yb - gap
            and y - half_h < yt + gap
        ):
            return True
    return False

def _try_place_on_boundary(bound, cx, cy, w, h, cw, ch, placed, gap):
    half_w = w / 2
    half_h = h / 2

    if bound == 0:                              
        fixed_coord = half_w + gap
        target = cy
        coord_min = half_h + gap
        coord_max = ch - half_h - gap
        slide_axis = "y"
    elif bound == 1:         
        fixed_coord = cw - half_w - gap
        target = cy
        coord_min = half_h + gap
        coord_max = ch - half_h - gap
        slide_axis = "y"
    elif bound == 2:                             
        fixed_coord = half_h + gap
        target = cx
        coord_min = half_w + gap
        coord_max = cw - half_w - gap
        slide_axis = "x"
    else:                  
        fixed_coord = ch - half_h - gap
        target = cx
        coord_min = half_w + gap
        coord_max = cw - half_w - gap
        slide_axis = "x"

    target = max(coord_min, min(coord_max, target))
    step = max(min(w, h) * 0.10, 0.05)

    def make_pos(slide_val):
        if slide_axis == "y":
            return (fixed_coord, slide_val)
        return (slide_val, fixed_coord)

    for k in range(0, 5000):
        signs = (0,) if k == 0 else (1, -1)
        for sign in signs:
            slide = target + sign * k * step
            if slide < coord_min or slide > coord_max:
                continue
            x, y = make_pos(slide)
            if not _overlaps_any(x, y, w, h, placed, gap):
                return (x, y)
    return None

def _legalize_to_boundary(
    positions: np.ndarray,
    large_indices: np.ndarray,
    sizes: np.ndarray,
    cw: float,
    ch: float,
    gap: float = GAP,
) -> np.ndarray:
    sorted_idx = sorted(
        list(large_indices), key=lambda i: -float(sizes[i, 0] * sizes[i, 1])
    )
    placed: List[Tuple[float, float, float, float]] = []
    new_positions = positions.copy()

    for idx in sorted_idx:
        cx = float(positions[idx, 0])
        cy = float(positions[idx, 1])
        w = float(sizes[idx, 0])
        h = float(sizes[idx, 1])

        d_left = cx - w / 2
        d_right = cw - cx - w / 2
        d_bot = cy - h / 2
        d_top = ch - cy - h / 2
        boundary_order = sorted(
            range(4), key=lambda b: [d_left, d_right, d_bot, d_top][b]
        )

        placed_this = False
        for bound in boundary_order:
            result = _try_place_on_boundary(
                bound, cx, cy, w, h, cw, ch, placed, gap
            )
            if result is not None:
                x, y = result
                new_positions[idx] = (x, y)
                placed.append((x - w / 2, x + w / 2, y - h / 2, y + h / 2))
                placed_this = True
                break

        if not placed_this:

            new_positions[idx] = (cw / 2, ch / 2)

    return new_positions

def _spiral_legalize(
    pos, sizes, fixed, cw, ch, gap=GAP, ring_cap=LEGALIZE_RING,
):
    n = pos.shape[0]
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2 + gap
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2 + gap

    placed = np.zeros(n, dtype=bool)
    legal = pos.copy()
    placed_idx = np.where(fixed)[0].tolist()
    placed_arr = np.asarray(placed_idx, dtype=np.int64)
    for i in placed_idx:
        placed[i] = True

    movable_idx = np.where(~fixed)[0]
    order = sorted(
        movable_idx.tolist(), key=lambda i: -float(sizes[i, 0] * sizes[i, 1])
    )

    def overlaps(idx, x, y):
        if placed_arr.size == 0:
            return False
        dx = np.abs(x - legal[placed_arr, 0])
        dy = np.abs(y - legal[placed_arr, 1])
        bad = (dx < sep_x[idx, placed_arr]) & (dy < sep_y[idx, placed_arr])
        return bool(bad.any())

    for idx in order:
        x_min = float(half_w[idx] + gap)
        x_max = float(cw - half_w[idx] - gap)
        y_min = float(half_h[idx] + gap)
        y_max = float(ch - half_h[idx] - gap)
        x0 = max(x_min, min(x_max, float(pos[idx, 0])))
        y0 = max(y_min, min(y_max, float(pos[idx, 1])))
        if not overlaps(idx, x0, y0):
            legal[idx, 0] = x0
            legal[idx, 1] = y0
            placed[idx] = True
            placed_idx.append(idx)
            placed_arr = np.asarray(placed_idx, dtype=np.int64)
            continue

        step = max(min(sizes[idx, 0], sizes[idx, 1]) * 0.25, 0.05)
        best = (x0, y0)
        best_d = float("inf")
        found = False
        for r in range(1, ring_cap + 1):
            for dxm in range(-r, r + 1):
                dym_values = range(-r, r + 1) if dxm in (-r, r) else (-r, r)
                for dym in dym_values:
                    cx = max(x_min, min(x_max, float(x0 + dxm * step)))
                    cy = max(y_min, min(y_max, float(y0 + dym * step)))
                    if overlaps(idx, cx, cy):
                        continue
                    d = (cx - x0) ** 2 + (cy - y0) ** 2
                    if d < best_d:
                        best_d = d
                        best = (cx, cy)
                        found = True
            if found:
                break
        legal[idx, 0] = best[0]
        legal[idx, 1] = best[1]
        placed[idx] = True
        placed_idx.append(idx)
        placed_arr = np.asarray(placed_idx, dtype=np.int64)
    return legal

def _push_legalize(
    pos, sizes, fixed, cw, ch, gap=GAP, max_iters=80
):
    pos = pos.copy()
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2 + gap
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2 + gap

    fixed_idx = np.where(fixed)[0]
    init_fixed = pos[fixed_idx].copy()

    for _ in range(max_iters):
        dx = pos[:, 0:1] - pos[:, 0:1].T
        dy = pos[:, 1:2] - pos[:, 1:2].T
        adx = np.abs(dx)
        ady = np.abs(dy)
        ovr_x = np.maximum(0.0, sep_x - adx)
        ovr_y = np.maximum(0.0, sep_y - ady)
        overlapping = (ovr_x > 1e-12) & (ovr_y > 1e-12)
        np.fill_diagonal(overlapping, False)
        if not overlapping.any():
            break

        push_x_mask = overlapping & (ovr_x <= ovr_y)
        push_y_mask = overlapping & (ovr_y < ovr_x)
        sign_x = np.where(dx >= 0, 1.0, -1.0)
        sign_y = np.where(dy >= 0, 1.0, -1.0)
        push_x = np.where(push_x_mask, ovr_x * 0.5 * sign_x, 0.0)
        push_y = np.where(push_y_mask, ovr_y * 0.5 * sign_y, 0.0)
        force_x = push_x.sum(axis=1)
        force_y = push_y.sum(axis=1)
        pos[:, 0] += force_x
        pos[:, 1] += force_y
        pos[:, 0] = np.clip(pos[:, 0], half_w + gap, cw - half_w - gap)
        pos[:, 1] = np.clip(pos[:, 1], half_h + gap, ch - half_h - gap)
        pos[fixed_idx] = init_fixed
    return pos

def _robust_legalize(pos, sizes, fixed, cw, ch, gap=GAP):
    pos = _spiral_legalize(pos, sizes, fixed, cw, ch, gap=gap)
    n = pos.shape[0]
    if n > 1:
        dx = np.abs(pos[:, 0:1] - pos[:, 0:1].T)
        dy = np.abs(pos[:, 1:2] - pos[:, 1:2].T)
        sep_x_strict = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2 - 1e-6
        sep_y_strict = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2 - 1e-6
        bad = (dx < sep_x_strict) & (dy < sep_y_strict)
        np.fill_diagonal(bad, False)
        if bad.any():
            pos = _push_legalize(
                pos, sizes, fixed, cw, ch, gap=gap, max_iters=80,
            )
            pos = _spiral_legalize(
                pos, sizes, fixed, cw, ch, gap=gap, ring_cap=20,
            )
    return pos

def _has_hard_overlaps(pos, sizes, gap=0.0, use_float32=True):
    check_pos = pos.astype(np.float32) if use_float32 else pos
    check_sizes = sizes.astype(np.float32) if use_float32 else sizes
    if check_pos.shape[0] <= 1:
        return False
    dx = np.abs(check_pos[:, 0:1] - check_pos[:, 0:1].T)
    dy = np.abs(check_pos[:, 1:2] - check_pos[:, 1:2].T)
    sep_x = (check_sizes[:, 0:1] + check_sizes[:, 0:1].T) / 2 + gap
    sep_y = (check_sizes[:, 1:2] + check_sizes[:, 1:2].T) / 2 + gap
    bad = (dx < sep_x) & (dy < sep_y)
    np.fill_diagonal(bad, False)
    return bool(bad.any())


def _legalize_large_then_small(
    pos, sizes, fixed, large_mask, cw, ch, gap=GAP
):
    legal = pos.copy()
    large_mask = np.asarray(large_mask, dtype=bool)
    fixed = np.asarray(fixed, dtype=bool)

    large_idx = np.where(large_mask)[0]
    if len(large_idx) > 0:
        sub_pos = legal[large_idx].copy()
        sub_sizes = sizes[large_idx]
        sub_fixed = fixed[large_idx]
        legal[large_idx] = _robust_legalize(
            sub_pos, sub_sizes, sub_fixed, cw, ch, gap=gap,
        )

    fixed_for_small = fixed.copy()
    fixed_for_small[large_mask] = True
    legal = _robust_legalize(
        legal, sizes, fixed_for_small, cw, ch, gap=gap,
    )

    for _ in range(3):
        if not _has_hard_overlaps(legal, sizes, gap=0.0, use_float32=True):
            break
        legal = _push_legalize(
            legal, sizes, fixed_for_small, cw, ch,
            gap=gap, max_iters=300,
        )
        if not _has_hard_overlaps(legal, sizes, gap=0.0, use_float32=True):
            break
        legal = _spiral_legalize(
            legal, sizes, fixed_for_small, cw, ch,
            gap=gap, ring_cap=160,
        )

    if _has_hard_overlaps(legal, sizes, gap=0.0, use_float32=True):
        for _ in range(2):
            legal = _push_legalize(
                legal, sizes, fixed, cw, ch,
                gap=gap, max_iters=500,
            )
            if not _has_hard_overlaps(legal, sizes, gap=0.0, use_float32=True):
                break
            legal = _spiral_legalize(
                legal, sizes, fixed, cw, ch, gap=gap, ring_cap=200,
            )
            if not _has_hard_overlaps(legal, sizes, gap=0.0, use_float32=True):
                break

    return legal

PROXY_WL_WEIGHT = 1.0
PROXY_DENSITY_WEIGHT = 0.5
PROXY_CONGESTION_WEIGHT = 0.5


def _proxy_overall(wl_norm_cost, density_cost, congestion_cost):
    """Mimic official PlacementCost proxy formula:
    proxy = 1.0 * wirelength + 0.5 * density + 0.5 * congestion
    where wirelength is normalized as HPWL / ((cw+ch) * net_cnt).
    """
    return (
        PROXY_WL_WEIGHT * wl_norm_cost
        + PROXY_DENSITY_WEIGHT * density_cost
        + PROXY_CONGESTION_WEIGHT * congestion_cost
    )


def _compute_proxy_costs(
    benchmark,
    pos,
    sizes_np,
    n_total,
    net_weights_np: np.ndarray | None = None,
    wl_net_count: float | None = None,
):
    """Fast local proxy estimate for progress logging.

    This uses Benchmark's filtered netlist and lightweight density/congestion
    reimplementations, so it is not a bit-exact PlacementCost score.
    """
    pin_owner_np, pin_offset_np, pin_net_np, port_pos_np = (
        _build_pin_arrays_np(benchmark)
    )
    pos_eval = np.asarray(pos, dtype=np.float32)
    pin_offset_eval = pin_offset_np.astype(np.float32, copy=False)
    port_pos_eval = port_pos_np.astype(np.float32, copy=False)
    num_nets = len(benchmark.net_pin_nodes)
    nets_pin_indices = _net_pin_index_table_cached(
        benchmark, pin_net_np, num_nets
    )

    hpwl = _all_net_hpwls_vec(
        pin_owner_np, pin_offset_eval, pin_net_np, pos_eval, port_pos_eval,
        n_total, num_nets,
    )
    if net_weights_np is not None and len(net_weights_np) == num_nets:
        hpwl = hpwl * np.asarray(net_weights_np, dtype=np.float64)
    wl_total = float(hpwl.sum())

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    norm_net_count = (
        float(wl_net_count)
        if wl_net_count is not None and wl_net_count > 0.0
        else float(max(num_nets, 1))
    )
    wl_denom = max((cw + ch) * norm_net_count, EPS)
    wl_norm_cost = wl_total / wl_denom

    den = _IncrementalDensityCost(benchmark, pos_eval, sizes_np)
    cong = _IncrementalCongestionCost(
        benchmark, pos_eval, sizes_np, pin_owner_np, pin_offset_eval,
        nets_pin_indices, port_pos_eval, net_weights_np=net_weights_np,
    )
    density_cost = float(den.current_cost)
    congestion_cost = float(cong.current_cost)
    proxy = _proxy_overall(wl_norm_cost, density_cost, congestion_cost)

    return wl_total, wl_norm_cost, density_cost, congestion_cost, proxy


def _benchmark_dir_candidates(benchmark: Benchmark):
    name = str(getattr(benchmark, "name", ""))
    if not name:
        return
    candidates = [
        (
            _REPO_ROOT
            / "external"
            / "MacroPlacement"
            / "Testcases"
            / "ICCAD04"
            / name
        ),
        (
            _REPO_ROOT
            / "external"
            / "MacroPlacement"
            / "Flows"
            / "NanGate45"
            / name
            / "netlist"
            / "output_CT_Grouping"
        ),
        (
            _REPO_ROOT
            / "external"
            / "MacroPlacement"
            / "Flows"
            / "NanGate45"
            / name
            / "netlist"
            / "output_CodeElement"
        ),
    ]
    for path in candidates:
        if (path / "netlist.pb.txt").exists():
            yield path


def _load_official_proxy_plc(benchmark: Benchmark):
    for bench_dir in _benchmark_dir_candidates(benchmark):
        try:
            from macro_place.loader import load_benchmark_from_dir

            # PlacementCost is chatty when parsing; keep proxy logging readable.
            with contextlib.redirect_stdout(io.StringIO()):
                _, plc = load_benchmark_from_dir(str(bench_dir))
            return plc
        except Exception as exc:
            print(
                f"[proxy-cost] {benchmark.name}: official PLC load failed "
                f"from {bench_dir}: {exc}",
                flush=True,
            )
    return None


def _official_proxy_params_from_plc(benchmark: Benchmark, plc, num_nets: int):
    if plc is None:
        return None, None
    try:
        weights = []
        for driver in plc.nets.keys():
            mod_idx = plc.mod_name_to_indices[driver]
            weights.append(float(plc.modules_w_pins[mod_idx].get_weight()))
        if len(weights) != num_nets:
            print(
                f"[proxy-cost] {benchmark.name}: official net weight count "
                f"mismatch ({len(weights)} != {num_nets}); "
                "using unweighted local proxy",
                flush=True,
            )
            return None, None
        net_count = float(getattr(plc, "net_cnt", 0.0) or 0.0)
        if net_count <= 0.0:
            net_count = float(sum(weights))
        return np.asarray(weights, dtype=np.float64), net_count
    except Exception as exc:
        print(
            f"[proxy-cost] {benchmark.name}: official proxy params failed: {exc}",
            flush=True,
        )
        return None, None


def _compute_official_proxy_costs(benchmark, pos, official_plc):
    if official_plc is None:
        return None
    try:
        from macro_place.objective import compute_proxy_cost

        placement_t = torch.tensor(pos, dtype=torch.float32)
        return compute_proxy_cost(placement_t, benchmark, official_plc)
    except Exception as exc:
        print(
            f"[proxy-cost] {benchmark.name}: official proxy check failed: {exc}",
            flush=True,
        )
        return None


def _proxy_components_for_acceptance(
    benchmark,
    pos,
    sizes_np,
    n_total,
    official_plc=None,
    net_weights_np: np.ndarray | None = None,
    wl_net_count: float | None = None,
):
    official = _compute_official_proxy_costs(benchmark, pos, official_plc)
    if official is not None:
        return {
            "proxy": float(official["proxy_cost"]),
            "wirelength": float(official["wirelength_cost"]),
            "density": float(official["density_cost"]),
            "congestion": float(official["congestion_cost"]),
            "source": "official",
        }

    _wl_total, wl_norm, density, congestion, proxy = _compute_proxy_costs(
        benchmark,
        pos,
        sizes_np,
        n_total,
        net_weights_np=net_weights_np,
        wl_net_count=wl_net_count,
    )
    return {
        "proxy": float(proxy),
        "wirelength": float(wl_norm),
        "density": float(density),
        "congestion": float(congestion),
        "source": "estimate",
    }


class VibePlacer:

    def __init__(
        self,
        seed: int = 7,
        num_threads=None,
        device=None,
        deterministic: bool = True,
    ):
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        if num_threads is None:
            num_threads = max(1, os.cpu_count() or 4)
        self.num_threads = int(num_threads)

        if device is not None:
            self.device = torch.device(device)
        elif self.deterministic:
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    @staticmethod
    def _finalize_positions(positions, half_w, half_h, fixed_np, original_pos, cw, ch):
        out = positions.copy()
        out[:, 0] = np.clip(out[:, 0], half_w + 1e-6, cw - half_w - 1e-6)
        out[:, 1] = np.clip(out[:, 1], half_h + 1e-6, ch - half_h - 1e-6)
        for fixed_i in np.where(fixed_np)[0]:
            out[fixed_i] = original_pos[fixed_i]
        return out

    def _step1_center_init(self, positions, fixed_np, half_w, half_h, n_total, cw, ch):
        rng_init = np.random.RandomState(self.seed)
        movable_all = ~fixed_np
        jitter = rng_init.uniform(-1e-3, 1e-3, size=(n_total, 2))
        positions[movable_all, 0] = cw / 2 + jitter[movable_all, 0]
        positions[movable_all, 1] = ch / 2 + jitter[movable_all, 1]
        positions[:, 0] = np.clip(
            positions[:, 0], half_w + GAP, cw - half_w - GAP
        )
        positions[:, 1] = np.clip(
            positions[:, 1], half_h + GAP, ch - half_h - GAP
        )
        return positions

    def _step3_adam_large(
        self, benchmark, positions, large_mask, fixed_np,
        sizes_np, cw, ch, n_total, official_net_weights_np=None,
        knobs=None,
    ):
        knobs = knobs or _baseline_knobs()
        movable_step3 = large_mask & ~fixed_np
        # Disable macro-blockage cong gradient: at this point most macros are
        # still center-bunched, so blockage signal is mostly self-noise and
        # tends to create phase-separated layouts on dense designs.
        early_blockage = (
            0.0 if CONG_MACRO_BLOCKAGE_EARLY_DISABLE
            else CONG_MACRO_BLOCKAGE_WEIGHT
        )
        return _adam_phase(
            benchmark, positions, movable_step3, sizes_np, cw, ch, n_total,
            device=self.device,
            iters=knobs["step3_iters"],
            lr_frac=knobs["step3_lr_frac"],
            lambda_start=knobs["step3_lambda_start"],
            lambda_end=knobs["step3_lambda_end"],
            net_weights_np=official_net_weights_np,
            label="step3_adam_large",
            perturbation=True,
            macro_blockage_weight=early_blockage,
        )

    def _step6_adam_small_and_exact(
        self, benchmark, positions, large_mask, fixed_np,
        sizes_np, cw, ch, n_total, official_net_weights_np=None,
        knobs=None,
    ):
        knobs = knobs or _baseline_knobs()
        movable_step6 = (~large_mask) & (~fixed_np)
        # Step6 small: still early in optimization; disable macro blockage in
        # the cong penalty so density can drive uniform spread without being
        # distorted by spurious blockage attraction/repulsion patterns.
        early_blockage = (
            0.0 if CONG_MACRO_BLOCKAGE_EARLY_DISABLE
            else CONG_MACRO_BLOCKAGE_WEIGHT
        )
        positions = _adam_phase(
            benchmark, positions, movable_step6, sizes_np, cw, ch, n_total,
            device=self.device,
            iters=knobs["step6_iters"],
            lr_frac=knobs["step6_lr_frac"],
            lambda_start=knobs["step6_lambda_start"],
            lambda_end=knobs["step6_lambda_end"],
            net_weights_np=official_net_weights_np,
            label="step6_adam_small",
            macro_blockage_weight=early_blockage,
        )

        # Step6 exact density: macros are spread by now; full blockage weight.
        positions = _adam_exact_density_phase(
            benchmark, positions, movable_step6, sizes_np,
            cw, ch, n_total, device=self.device,
            iters=knobs["exact_density_iters"],
            lr_frac=knobs["exact_density_lr_frac"],
            lambda_start=knobs["exact_density_lambda_start"],
            lambda_end=knobs["exact_density_lambda_end"],
            density_target=knobs["exact_density_target"],
            overflow_weight=knobs["exact_density_overflow_weight"],
            net_weights_np=official_net_weights_np,
            label="step6_adam_exact_density",
        )
        return positions


    def _global_analytical_place(
        self, benchmark, positions, sizes_np, fixed_np, large_mask,
        n_hard, n_total, cw, ch,
        official_net_weights_np=None, official_wl_net_count=None,
        knobs=None,
    ):
        knobs = knobs or _baseline_knobs()
        if knobs["global_place_iters"] <= 0:
            return positions

        movable_all = ~large_mask & ~fixed_np
        out = _adam_phase(
            benchmark, positions, movable_all, sizes_np,
            cw, ch, n_total, device=self.device,
            iters=knobs["global_place_iters"],
            lr_frac=knobs["global_place_lr_frac"],
            lambda_start=knobs["global_place_lambda_start"],
            lambda_end=knobs["global_place_lambda_end"],
            density_target_slack=knobs["global_place_density_target_slack"],
            cong_weight=knobs["global_place_cong_weight"],
            net_weights_np=official_net_weights_np,
            label="global_analytical_place",
        )
        out[:n_hard] = _legalize_large_then_small(
            out[:n_hard], sizes_np[:n_hard], fixed_np[:n_hard],
            large_mask[:n_hard], cw, ch, gap=GAP,
        )
        before = _compute_proxy_costs(
            benchmark, positions, sizes_np, n_total,
            net_weights_np=official_net_weights_np,
            wl_net_count=official_wl_net_count,
        )[-1]
        after = _compute_proxy_costs(
            benchmark, out, sizes_np, n_total,
            net_weights_np=official_net_weights_np,
            wl_net_count=official_wl_net_count,
        )[-1]
        if after > before + GLOBAL_PLACE_ACCEPT_TOL:
            print(
                f"[adam] global_analytical_place rejected | "
                f"proxy {before:.6f} -> {after:.6f}",
                flush=True,
            )
            return positions
        print(
            f"[adam] global_analytical_place accepted | "
            f"proxy {before:.6f} -> {after:.6f}",
            flush=True,
        )
        return out

if __name__ not in sys.modules:
    import types
    mod = types.ModuleType(__name__)
    mod.__dict__.update(globals())
    sys.modules[__name__] = mod
