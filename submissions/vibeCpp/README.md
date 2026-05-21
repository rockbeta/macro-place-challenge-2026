# vibeCpp

This directory contains a C++ implementation of the Vibe placer flow with a
small Python adapter so it can still be loaded by the repository's existing
`uv run evaluate ...` harness.

## Files

- `placer.cpp` - full standalone C++17 SA placer binary.
- `placerFast.cpp` - faster standalone C++17 SA placer binary for seed screening.
- `placer.py` - challenge API adapter that serializes a `Benchmark`, builds and
  runs the C++ binaries, then returns a `torch.Tensor`.
- `Makefile` - optional manual build target.

## Usage

```bash
uv run evaluate submissions/vibeCpp/placer.py -b ibm01
```

The adapter builds `build/vibe_placer_full` and `build/vibe_placer_fast`
automatically on first use. To build by hand:

```bash
make -C submissions/vibeCpp
```

## Threading

The C++ placer uses `std::thread`/`std::async` for parallel all-macro SA shards.
By default the main placer uses up to 8 hardware threads, and the fast placer uses up to 2. Override that with:

```bash
VIBECPP_THREADS=4 uv run evaluate submissions/vibeCpp/placer.py -b ibm04
```

For reproducible single-thread behavior:

```bash
VIBECPP_THREADS=1 uv run evaluate submissions/vibeCpp/placer.py -b ibm04
```

## SA Iteration Controls

The C++ flow has a normal all-macro SA pass, a smaller congestion polish, a
hot-grid congestion SA polish, an area-neighborhood swap SA pass, and a final
all-macro fix SA pass. The hot-grid pass selects only movable macros on nets
that contribute to the top congestion-cost grid cells, then evaluates move
distances at `0.4, 0.7, 1.0, 1.3, 1.7, 2.0x`. The swap pass sorts movable
hard/soft macros by area, samples six similar-area macros, tries all pair swaps,
keeps the lowest proxy-improving legal swap, and stops early if the cumulative
success rate falls below `0.01` at a 10000-iteration report point.

Useful overrides:

```bash
VIBECPP_SA_FULL_ITERS=600000 \
VIBECPP_CONGESTION_POLISH_ITERS=50000 \
VIBECPP_HOT_CONGESTION_SA_ITERS=600000 \
VIBECPP_SWAP_SA_ITERS=200000 \
VIBECPP_FINAL_FIX_SA_ITERS=600000 \
uv run evaluate submissions/vibeCpp/placer.py -b ibm04
```

## Analytical Warm Start

By default, `placer.py` runs the analytical frontend from
`submissions/vibe/vplacer.py` through `step8_global_analytical_place`, then
serializes that placement into the C++ legalizer/SA flow. The adapter runs eight
different analytical warm-start seeds **in parallel worker processes**, sends each
through `placerFast.cpp`, keeps the top 2 lowest proxy-cost candidates, and then
sends those candidates through `placer.cpp` for the final SA push.

Change the seed sweep settings with:

```bash
VIBECPP_SEED_SWEEP_COUNT=8 \
VIBECPP_SEED_SWEEP_STRIDE=1009 \
VIBECPP_FINAL_TOPK=2 \
VIBECPP_KNOB_PERTURB=1 \
uv run evaluate submissions/vibeCpp/placer.py -b ibm04
```

Each analytical seed pass can write step snapshots to `vis_step/`, with
filenames such as `ibm01_iter1_step1.png`. The snapshots show hard, fixed, and
soft macros, plus the hottest density and congestion grid cells. They are
disabled by default for speed. Enable or redirect these images with:

```bash
VIBECPP_DUMP_VIS_STEPS=1 uv run evaluate submissions/vibeCpp/placer.py -b ibm04
VIBECPP_VIS_STEP_DIR=/tmp/vibe_steps uv run evaluate submissions/vibeCpp/placer.py -b ibm04
```

Disable the Python analytical warm start and use the input `.plc` placement
directly with:

```bash
VIBECPP_ANALYTICAL_WARMSTART=0 uv run evaluate submissions/vibeCpp/placer.py -b ibm04
```

Limit PyTorch CPU threads used by the analytical warm start in each worker with:

```bash
VIBECPP_ANALYTICAL_THREADS=4 uv run evaluate submissions/vibeCpp/placer.py -b ibm04
```
