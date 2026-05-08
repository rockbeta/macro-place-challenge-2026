from __future__ import annotations

import math
import os
import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from macro_place.benchmark import Benchmark

GAP = 0.0005
EPS = 1e-12
LEGALIZE_RING = 80

LARGE_MACRO_AREA_FRAC = 0.005

STEP3_ITERS = 200
STEP3_LR_FRAC = 0.01
STEP3_LAMBDA_START = 1.0
STEP3_LAMBDA_END = 1000.0

STEP6_ITERS = 360
STEP6_LR_FRAC = 0.01
STEP6_LAMBDA_START = 0.01
STEP6_LAMBDA_END = 200.0

EXACT_DENSITY_ITERS = 800
EXACT_DENSITY_LR_FRAC = 0.0004
EXACT_DENSITY_LAMBDA_START = 100_000.0
EXACT_DENSITY_LAMBDA_END = 10_000_000.0
EXACT_DENSITY_TARGET = 0.65
EXACT_DENSITY_OVERFLOW_WEIGHT = 8.0

GLOBAL_PLACE_ITERS = 600
GLOBAL_PLACE_LR_FRAC = 0.0015
GLOBAL_PLACE_LAMBDA_START = 20.0
GLOBAL_PLACE_LAMBDA_END = 3000.0
GLOBAL_PLACE_DENSITY_TARGET_SLACK = 0.85
GLOBAL_PLACE_CONG_WEIGHT = 0.35

ALL_MACRO_SA_MAX_ITERS = 600_000
ALL_MACRO_SA_CONG_WEIGHT = 0.8
ALL_MACRO_SA_DENSITY_WEIGHT = 0.5

HROUTING_ALLOC = 30.304
VROUTING_ALLOC = 71.304
SMOOTH_RANGE = 2
CONGESTION_TOP_FRAC = 0.05
DENSITY_TOP_FRAC = 0.10

def _build_pin_arrays_np(
    benchmark: Benchmark,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros((0, 2), dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            port_pos_np,
        )
    return (
        np.asarray(owners, dtype=np.int64),
        np.asarray(offsets, dtype=np.float64),
        np.asarray(nets, dtype=np.int64),
        port_pos_np,
    )

def _build_macro_to_nets(
    pin_owner_np: np.ndarray, pin_net_np: np.ndarray, n_total: int
) -> List[np.ndarray]:
    macro_to_nets: List[List[int]] = [[] for _ in range(n_total)]
    seen: List[set] = [set() for _ in range(n_total)]
    for owner, net in zip(pin_owner_np.tolist(), pin_net_np.tolist()):
        if 0 <= owner < n_total and net not in seen[owner]:
            seen[owner].add(net)
            macro_to_nets[owner].append(net)
    return [np.asarray(x, dtype=np.int64) for x in macro_to_nets]

def _net_pin_index_table(
    pin_net_np: np.ndarray, num_nets: int
) -> List[np.ndarray]:
    if num_nets == 0:
        return []
    order = np.argsort(pin_net_np, kind="stable")
    sorted_nets = pin_net_np[order]
    bounds = np.searchsorted(sorted_nets, np.arange(num_nets + 1))
    return [order[bounds[ni] : bounds[ni + 1]] for ni in range(num_nets)]

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

def _net_hpwl_world(
    pin_owner_np: np.ndarray,
    pin_offset_np: np.ndarray,
    idx_arr: np.ndarray,
    pos: np.ndarray,
    port_pos_np: np.ndarray,
    n_total: int,
) -> float:
    if len(idx_arr) < 2:
        return 0.0
    owners = pin_owner_np[idx_arr]
    offs = pin_offset_np[idx_arr]
    mac_mask = owners < n_total
    xs = np.empty(len(idx_arr), dtype=np.float64)
    ys = np.empty(len(idx_arr), dtype=np.float64)
    if mac_mask.any():
        mo = owners[mac_mask]
        xs[mac_mask] = pos[mo, 0] + offs[mac_mask, 0]
        ys[mac_mask] = pos[mo, 1] + offs[mac_mask, 1]
    if (~mac_mask).any():
        po = owners[~mac_mask] - n_total
        xs[~mac_mask] = port_pos_np[po, 0]
        ys[~mac_mask] = port_pos_np[po, 1]
    return float((xs.max() - xs.min()) + (ys.max() - ys.min()))

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

    def _macro_cell_range(self, cx, cy, w, h):
        if w <= 0.0 or h <= 0.0:
            return None
        lx = float(cx - w / 2)
        ux = float(cx + w / 2)
        ly = float(cy - h / 2)
        uy = float(cy + h / 2)
        if ux <= 0.0 or uy <= 0.0 or lx >= self.cw or ly >= self.ch:
            return None
        c0 = max(0, min(self.grid_col - 1, int(math.floor(lx / self.grid_w))))
        c1 = max(0, min(self.grid_col - 1, int(math.floor(ux / self.grid_w))))
        r0 = max(0, min(self.grid_row - 1, int(math.floor(ly / self.grid_h))))
        r1 = max(0, min(self.grid_row - 1, int(math.floor(uy / self.grid_h))))
        return (r0, r1, c0, c1)

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

    def begin_update(self, indices, new_pos):
        self.saved_occupied = self.occupied.copy()
        self.saved_region = None
        for idx in indices:
            self._add_macro(idx, self.pos[idx, 0], self.pos[idx, 1], -1.0)
        for idx, np_pos in zip(indices, new_pos):
            self._add_macro(idx, np_pos[0], np_pos[1], 1.0)
        new_cost = self._calc_cost()
        return new_cost, {"old_cost": self.current_cost}

    def begin_single_update(self, idx, nx, ny):
        self.saved_occupied = None
        self.saved_region = None
        w = float(self.sizes[idx, 0])
        h = float(self.sizes[idx, 1])
        old_rng = self._macro_cell_range(self.pos[idx, 0], self.pos[idx, 1], w, h)
        new_rng = self._macro_cell_range(nx, ny, w, h)
        if old_rng is not None or new_rng is not None:
            ranges = [rng for rng in (old_rng, new_rng) if rng is not None]
            r_lo = min(rng[0] for rng in ranges)
            r_hi = max(rng[1] for rng in ranges)
            c_lo = min(rng[2] for rng in ranges)
            c_hi = max(rng[3] for rng in ranges)
            saved = self.occupied[r_lo : r_hi + 1, c_lo : c_hi + 1].copy()
            self.saved_region = (r_lo, r_hi, c_lo, c_hi, saved)
        self._add_macro(idx, self.pos[idx, 0], self.pos[idx, 1], -1.0)
        self._add_macro(idx, nx, ny, 1.0)
        new_cost = self._calc_cost()
        return new_cost, {"old_cost": self.current_cost, "new_cost": new_cost}

    def accept(self, token, indices, new_pos):
        self.current_cost = (
            token["new_cost"] if "new_cost" in token else self._calc_cost()
        )
        for idx, np_pos in zip(indices, new_pos):
            self.pos[idx, 0] = np_pos[0]
            self.pos[idx, 1] = np_pos[1]
        self.saved_occupied = None
        self.saved_region = None

    def accept_single(self, token, idx, nx, ny):
        self.current_cost = (
            token["new_cost"] if "new_cost" in token else self._calc_cost()
        )
        self.pos[idx, 0] = nx
        self.pos[idx, 1] = ny
        self.saved_occupied = None
        self.saved_region = None

    def reject(self, token):
        if self.saved_region is not None:
            r_lo, r_hi, c_lo, c_hi, saved = self.saved_region
            self.occupied[r_lo : r_hi + 1, c_lo : c_hi + 1] = saved
        elif self.saved_occupied is not None:
            self.occupied[:] = self.saved_occupied
        self.saved_occupied = None
        self.saved_region = None

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

        self.net_contribs = []
        self.net_route_keys = []
        for net_i in range(len(self.nets_pin_indices)):
            contrib, route_key = self._net_contrib(net_i, return_key=True)
            self.net_contribs.append(contrib)
            self.net_route_keys.append(route_key)
            self._apply(contrib, 1.0)

        self.macro_contribs: List = [None] * self.n_hard
        for macro_i in range(self.n_hard):
            contrib = self._macro_contrib(macro_i)
            self.macro_contribs[macro_i] = contrib
            self._apply(contrib, 1.0)

        self.current_cost = self._abu_cost()

    def _grid_cell(self, x, y):
        row = int(math.floor(y / self.grid_h))
        col = int(math.floor(x / self.grid_w))
        row = max(0, min(self.grid_row - 1, row))
        col = max(0, min(self.grid_col - 1, col))
        return row, col

    def _pin_xy(self, pin_idx):
        owner = int(self.pin_owner_np[pin_idx])
        off_x = float(self.pin_offset_np[pin_idx, 0])
        off_y = float(self.pin_offset_np[pin_idx, 1])
        if owner < self.n_total:
            return (
                float(self.pos[owner, 0] + off_x),
                float(self.pos[owner, 1] + off_y),
            )
        port_idx = owner - self.n_total
        return (
            float(self.port_pos_np[port_idx, 0]),
            float(self.port_pos_np[port_idx, 1]),
        )

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
        lp = col - self.smooth_range
        if lp < 0: lp = 0
        rp = col + self.smooth_range
        if rp >= self.grid_col: rp = self.grid_col - 1
        val = weight * self.inv_grid_v_routes / (rp - lp + 1)
        for ptr in range(lp, rp + 1):
            contrib.append((0, row, ptr, val))

    def _add_h_route(self, contrib, row, col, weight):
        if row < 0 or row >= self.grid_row or col < 0 or col >= self.grid_col:
            return
        lp = row - self.smooth_range
        if lp < 0: lp = 0
        up = row + self.smooth_range
        if up >= self.grid_row: up = self.grid_row - 1
        val = weight * self.inv_grid_h_routes / (up - lp + 1)
        for ptr in range(lp, up + 1):
            contrib.append((1, ptr, col, val))

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
        inv_grid_w = 1.0 / self.grid_w
        inv_grid_h = 1.0 / self.grid_h
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
                px = float(pos[owner, 0] + pin_offset_np[pin_idx, 0])
                py = float(pos[owner, 1] + pin_offset_np[pin_idx, 1])
            else:
                port_idx = owner - n_total
                px = float(port_pos_np[port_idx, 0])
                py = float(port_pos_np[port_idx, 1])

            row = int(py * inv_grid_h)
            col = int(px * inv_grid_w)

            if row < 0: row = 0
            elif row > grid_row_minus_1: row = grid_row_minus_1
            if col < 0: col = 0
            elif col > grid_col_minus_1: col = grid_col_minus_1

            gcell = (row, col)
            node_gcells.add(gcell)
            if i == 0:
                source_gcell = gcell

        return source_gcell, node_gcells

    def _net_route_key(self, net_i):
        idx_arr = self.nets_pin_indices[net_i]
        if len(idx_arr) < 2:
            return None
        source_gcell, node_gcells = self._get_net_gcells(idx_arr)
        return source_gcell, tuple(sorted(node_gcells))

    def _net_contrib(self, net_i, return_key=False):
        idx_arr = self.nets_pin_indices[net_i]
        contrib: List[Tuple[int, int, int, float]] = []
        if len(idx_arr) < 2:
            route_key = None
            return (contrib, route_key) if return_key else contrib
        source_gcell, node_gcells = self._get_net_gcells(idx_arr)
        route_key = source_gcell, tuple(sorted(node_gcells))
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
        return (contrib, route_key) if return_key else contrib

    def _macro_contrib(self, macro_i):
        contrib: List[Tuple[int, int, int, float]] = []
        x = float(self.pos[macro_i, 0])
        y = float(self.pos[macro_i, 1])
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
                contrib.append(
                    (
                        2,
                        row,
                        col,
                        x_dist * self.vrouting_alloc * self.inv_grid_v_routes,
                    )
                )
                contrib.append(
                    (
                        3,
                        row,
                        col,
                        y_dist * self.hrouting_alloc * self.inv_grid_h_routes,
                    )
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
                contrib.append(
                    (
                        2,
                        row,
                        col,
                        -x_dist * self.vrouting_alloc * self.inv_grid_v_routes,
                    )
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
                contrib.append(
                    (
                        3,
                        row,
                        col,
                        -y_dist * self.hrouting_alloc * self.inv_grid_h_routes,
                    )
                )
        return contrib

    def _apply(self, contrib, sign):
        grids = self.grids
        for grid_id, row, col, val in contrib:
            grids[grid_id][row, col] += sign * val

    def _abu_cost(self):
        np.add(self.v_route, self.v_macro, out=self._v_total)
        np.add(self.h_route, self.h_macro, out=self._h_total)
        self._abu_total[: self.grid_size] = self._v_total.reshape(-1)
        self._abu_total[self.grid_size :] = self._h_total.reshape(-1)
        cnt = self.abu_top_count
        if cnt == 0:
            return float(self._abu_total.max()) if len(self._abu_total) else 0.0
        return float(np.partition(self._abu_total, -cnt)[-cnt:].mean())

    def begin_update(
        self, moved_macros, affected_nets, skip_unchanged_routes=False
    ):
        affected_nets = sorted({int(ni) for ni in affected_nets})
        moved_macros = sorted(
            {int(mi) for mi in moved_macros if int(mi) < self.n_hard}
        )
        token = {
            "old_cost": self.current_cost,
            "nets": [],
            "macros": [],
        }
        for ni in affected_nets:
            old_key = self.net_route_keys[ni]
            if skip_unchanged_routes:
                new_key = self._net_route_key(ni)
                if new_key == old_key:
                    continue
                new = self._net_contrib(ni)
            else:
                new, new_key = self._net_contrib(ni, return_key=True)
            old = self.net_contribs[ni]
            self._apply(old, -1.0)
            self._apply(new, 1.0)
            token["nets"].append((ni, old, new, old_key, new_key))
        for mi in moved_macros:
            old = self.macro_contribs[mi]
            new = self._macro_contrib(mi)
            self._apply(old, -1.0)
            self._apply(new, 1.0)
            token["macros"].append((mi, old, new))
        token["new_cost"] = self._abu_cost()
        return token["new_cost"], token

    def begin_single_update(
        self, moved_macro, affected_nets, skip_unchanged_routes=False
    ):
        token = {
            "old_cost": self.current_cost,
            "nets": [],
            "macros": [],
        }
        for ni in affected_nets:
            old_key = self.net_route_keys[ni]
            if skip_unchanged_routes:
                new_key = self._net_route_key(ni)
                if new_key == old_key:
                    continue
                new = self._net_contrib(ni)
            else:
                new, new_key = self._net_contrib(ni, return_key=True)
            old = self.net_contribs[ni]
            self._apply(old, -1.0)
            self._apply(new, 1.0)
            token["nets"].append((ni, old, new, old_key, new_key))
        if moved_macro < self.n_hard:
            old = self.macro_contribs[moved_macro]
            new = self._macro_contrib(moved_macro)
            self._apply(old, -1.0)
            self._apply(new, 1.0)
            token["macros"].append((moved_macro, old, new))
        token["new_cost"] = self._abu_cost()
        return token["new_cost"], token

    def accept(self, token):
        for ni, _old, new, _old_key, new_key in token["nets"]:
            self.net_contribs[ni] = new
            self.net_route_keys[ni] = new_key
        for mi, _old, new in token["macros"]:
            self.macro_contribs[mi] = new
        self.current_cost = token["new_cost"]

    def reject(self, token):
        for _ni, old, new, _old_key, _new_key in token["nets"]:
            self._apply(new, -1.0)
            self._apply(old, 1.0)
        for _mi, old, new in token["macros"]:
            self._apply(new, -1.0)
            self._apply(old, 1.0)
        self.current_cost = token["old_cost"]

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
) -> torch.Tensor:
    if pin_owner.numel() == 0:
        return all_pos.new_zeros(())
    px = all_pos[pin_owner, 0] + pin_offset[:, 0]
    py = all_pos[pin_owner, 1] + pin_offset[:, 1]
    return (
        _wl_per_net(px, pin_net, num_nets, gamma)
        + _wl_per_net(-px, pin_net, num_nets, gamma)
        + _wl_per_net(py, pin_net, num_nets, gamma)
        + _wl_per_net(-py, pin_net, num_nets, gamma)
    ).sum()

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

def _density_grid_exact_rect(
    pos: torch.Tensor,
    sizes: torch.Tensor,
    n_bins_x: int,
    n_bins_y: int,
    cw: float,
    ch: float,
) -> torch.Tensor:
    bw = cw / n_bins_x
    bh = ch / n_bins_y

    xl = pos[:, 0:1] - sizes[:, 0:1] / 2
    xr = pos[:, 0:1] + sizes[:, 0:1] / 2
    yb = pos[:, 1:2] - sizes[:, 1:2] / 2
    yt = pos[:, 1:2] + sizes[:, 1:2] / 2

    bx_l = torch.arange(n_bins_x, dtype=pos.dtype, device=pos.device) * bw
    bx_r = bx_l + bw
    by_b = torch.arange(n_bins_y, dtype=pos.dtype, device=pos.device) * bh
    by_t = by_b + bh

    ox = torch.relu(
        torch.minimum(xr, bx_r[None, :]) - torch.maximum(xl, bx_l[None, :])
    )
    oy = torch.relu(
        torch.minimum(yt, by_t[None, :]) - torch.maximum(yb, by_b[None, :])
    )
    density_area = torch.einsum("mx,my->yx", ox, oy)
    return density_area / (bw * bh)

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
) -> np.ndarray:
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

    total_area = float((sizes_np[:, 0] * sizes_np[:, 1]).sum())
    target = (total_area / (cw * ch)) * density_target_slack

    half_w = sizes_t[:, 0] / 2
    half_h = sizes_t[:, 1] / 2

    opt = torch.optim.Adam([pos_var], lr=canvas * lr_frac)
    gamma_start = canvas * 0.05
    gamma_end = canvas * 0.005

    for it in range(iters):
        frac = min(1.0, it / max(1, iters - 1))
        gamma = gamma_start * (gamma_end / gamma_start) ** frac
        lam = lambda_start * (lambda_end / lambda_start) ** frac

        opt.zero_grad(set_to_none=True)
        all_pos = (
            torch.cat([pos_var, port_pos], dim=0) if n_ports > 0 else pos_var
        )
        wl = _wirelength(all_pos, pin_owner, pin_offset, pin_net, num_nets, gamma)
        density = _density_grid_gauss(pos_var, sizes_t, bin_x, bin_y, bw, bh)
        density_pen = (F.relu(density - target) ** 2).sum()

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
) -> np.ndarray:
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

    half_w = sizes_t[:, 0] / 2
    half_h = sizes_t[:, 1] / 2
    opt = torch.optim.Adam([pos_var], lr=canvas * lr_frac)
    gamma_start = canvas * 0.02
    gamma_end = canvas * 0.002

    for it in range(iters):
        frac = min(1.0, it / max(1, iters - 1))
        gamma = gamma_start * (gamma_end / gamma_start) ** frac
        lam = lambda_start * (lambda_end / lambda_start) ** frac

        opt.zero_grad(set_to_none=True)
        all_pos = (
            torch.cat([pos_var, port_pos], dim=0) if n_ports > 0 else pos_var
        )
        wl = _wirelength(all_pos, pin_owner, pin_offset, pin_net, num_nets, gamma)
        density = _density_grid_exact_rect(
            pos_var, sizes_t, n_bins_x, n_bins_y, cw, ch
        )
        density_flat = density.reshape(-1)
        hot_density = torch.topk(density_flat, top_k).values
        density_pen = (hot_density * hot_density).mean()
        density_pen = density_pen + overflow_weight * (
            F.relu(density_flat - density_target) ** 2
        ).mean()

        bin_x = torch.linspace(0, cw, n_bins_x, device=device) + cw / (2 * n_bins_x)
        bin_y = torch.linspace(0, ch, n_bins_y, device=device) + ch / (2 * n_bins_y)
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
    for i in np.where(fixed)[0]:
        placed[i] = True

    movable_idx = np.where(~fixed)[0]
    order = sorted(
        movable_idx.tolist(), key=lambda i: -float(sizes[i, 0] * sizes[i, 1])
    )

    def overlaps(idx, x, y):
        if not placed.any():
            return False
        dx = np.abs(x - legal[:, 0])
        dy = np.abs(y - legal[:, 1])
        bad = (dx < sep_x[idx]) & (dy < sep_y[idx]) & placed
        bad[idx] = False
        return bool(bad.any())

    for idx in order:
        x0 = float(np.clip(pos[idx, 0], half_w[idx] + gap, cw - half_w[idx] - gap))
        y0 = float(np.clip(pos[idx, 1], half_h[idx] + gap, ch - half_h[idx] - gap))
        if not overlaps(idx, x0, y0):
            legal[idx, 0] = x0
            legal[idx, 1] = y0
            placed[idx] = True
            continue

        step = max(min(sizes[idx, 0], sizes[idx, 1]) * 0.25, 0.05)
        best = (x0, y0)
        best_d = float("inf")
        found = False
        for r in range(1, ring_cap + 1):
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx = float(
                        np.clip(
                            x0 + dxm * step,
                            half_w[idx] + gap,
                            cw - half_w[idx] - gap,
                        )
                    )
                    cy = float(
                        np.clip(
                            y0 + dym * step,
                            half_h[idx] + gap,
                            ch - half_h[idx] - gap,
                        )
                    )
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

def _sa_all_macro_incremental(
    pos,
    n_hard,
    sizes_np,
    fixed_np,
    cw,
    ch,
    pin_owner_np,
    pin_offset_np,
    pin_net_np,
    macro_to_nets,
    nets_pin_indices,
    port_pos_np,
    benchmark: Benchmark,
    seed,
    congestion_weight: float = ALL_MACRO_SA_CONG_WEIGHT,
    density_weight: float = ALL_MACRO_SA_DENSITY_WEIGHT,
    hard_move_prob: float = 0.35,
    hard_sigma_start: float = 0.03,
    soft_sigma_start: float = 0.08,
    max_iters: int | None = None,
):
    rng = random.Random(seed)
    pos = pos.copy()
    n_total = pos.shape[0]
    half_w = sizes_np[:, 0] / 2
    half_h = sizes_np[:, 1] / 2
    movable_hard = [i for i in range(n_hard) if not fixed_np[i]]
    movable_soft = [i for i in range(n_hard, n_total) if not fixed_np[i]]
    if not movable_hard and not movable_soft:
        return pos

    num_nets = len(nets_pin_indices)
    wl_norm = max((cw + ch) * max(num_nets, 1), EPS)

    cur_hpwl = _all_net_hpwls_vec(
        pin_owner_np, pin_offset_np, pin_net_np, pos, port_pos_np,
        n_total, num_nets,
    )
    den_eval = _IncrementalDensityCost(benchmark, pos, sizes_np)
    cong_eval = _IncrementalCongestionCost(
        benchmark, pos, sizes_np, pin_owner_np, pin_offset_np,
        nets_pin_indices, port_pos_np,
    )
    macro_to_nets_cong = [
        tuple(sorted({int(net_i) for net_i in nets}))
        for nets in macro_to_nets
    ]

    cur_cost = float(cur_hpwl.sum()) + (
        congestion_weight * cong_eval.current_cost
        + density_weight * den_eval.current_cost
    ) * wl_norm
    best_cost = cur_cost
    best_pos = pos.copy()

    sep_x = (sizes_np[:n_hard, 0:1] + sizes_np[:n_hard, 0:1].T) / 2 + 1e-6
    sep_y = (sizes_np[:n_hard, 1:2] + sizes_np[:n_hard, 1:2].T) / 2 + 1e-6
    canvas = max(cw, ch)

    GRAD_STOP_THRESHOLD = 50.0
    GRAD_CONSEC_REQUIRED = 10
    grad_ref = GRAD_STOP_THRESHOLD * 10.0
    temp_start = canvas * 0.1
    temp_end = canvas * 0.00005
    temp_ratio = temp_end / temp_start

    iter_i = 0
    recent_costs = deque(maxlen=1000)
    consec_below = 0
    last_grad_abs = float("inf")
    frac = 0.0
    temp = temp_start * temp_ratio ** frac
    hard_sigma = canvas * (
        hard_sigma_start * (1.0 - frac) + 0.001 * frac
    )
    soft_sigma = canvas * (
        soft_sigma_start * (1.0 - frac) + 0.002 * frac
    )

    while True:
        if max_iters is not None and iter_i >= max_iters:
            break

        if movable_hard and (
            not movable_soft or rng.random() < hard_move_prob
        ):
            idx = int(rng.choice(movable_hard))
            sigma = hard_sigma
        else:
            idx = int(rng.choice(movable_soft))
            sigma = soft_sigma

        old_x = float(pos[idx, 0])
        old_y = float(pos[idx, 1])
        nx = max(
            half_w[idx] + 1e-6,
            min(cw - half_w[idx] - 1e-6, old_x + rng.gauss(0.0, sigma)),
        )
        ny = max(
            half_h[idx] + 1e-6,
            min(ch - half_h[idx] - 1e-6, old_y + rng.gauss(0.0, sigma)),
        )

        if idx < n_hard:
            pos[idx, 0] = nx
            pos[idx, 1] = ny
            dx = np.abs(nx - pos[:n_hard, 0])
            dy = np.abs(ny - pos[:n_hard, 1])
            bad = (dx < sep_x[idx]) & (dy < sep_y[idx])
            bad[idx] = False
            if bad.any():
                pos[idx, 0] = old_x
                pos[idx, 1] = old_y
                iter_i += 1
                continue
        else:
            pos[idx, 0] = nx
            pos[idx, 1] = ny

        new_den, den_token = den_eval.begin_single_update(idx, nx, ny)
        affected_nets = macro_to_nets[idx]
        old_wl = 0.0
        new_wl = 0.0
        updates = []
        for net_i in affected_nets:
            net_i = int(net_i)
            hpwl = _net_hpwl_world(
                pin_owner_np, pin_offset_np, nets_pin_indices[net_i],
                pos, port_pos_np, n_total,
            )
            old_wl += cur_hpwl[net_i]
            new_wl += hpwl
            updates.append((net_i, hpwl))

        delta = new_wl - old_wl
        skip_unchanged_routes = sigma < (
            0.35 * min(cong_eval.grid_w, cong_eval.grid_h)
        )
        new_cong, cong_token = cong_eval.begin_single_update(
            idx, macro_to_nets_cong[idx],
            skip_unchanged_routes=skip_unchanged_routes,
        )
        delta += (
            congestion_weight * (new_cong - cong_token["old_cost"])
            + density_weight * (new_den - den_token["old_cost"])
        ) * wl_norm

        if delta < 0.0 or rng.random() < math.exp(-delta / max(temp, EPS)):
            cur_cost += delta
            for net_i, hpwl in updates:
                cur_hpwl[net_i] = hpwl
            den_eval.accept_single(den_token, idx, nx, ny)
            cong_eval.accept(cong_token)
            if cur_cost < best_cost - 1e-12:
                best_cost = cur_cost
                best_pos = pos.copy()
        else:
            den_eval.reject(den_token)
            cong_eval.reject(cong_token)
            pos[idx, 0] = old_x
            pos[idx, 1] = old_y

        recent_costs.append(cur_cost)

        if iter_i % 1000 == 0 and len(recent_costs) == 1000:
            recent_snapshot = list(recent_costs)
            ma_curr = sum(recent_snapshot[500:]) / 500.0
            ma_prev = sum(recent_snapshot[:500]) / 500.0
            grad = ma_curr - ma_prev
            last_grad_abs = abs(grad)
            if last_grad_abs < GRAD_STOP_THRESHOLD:
                consec_below += 1
            else:
                consec_below = 0
            if math.isfinite(last_grad_abs):
                if last_grad_abs >= grad_ref:
                    frac = 0.0
                elif last_grad_abs <= GRAD_STOP_THRESHOLD:
                    frac = 1.0
                else:
                    frac = math.log(grad_ref / last_grad_abs) / math.log(
                        grad_ref / GRAD_STOP_THRESHOLD
                    )
                    frac = min(1.0, max(0.0, frac))
                temp = temp_start * temp_ratio ** frac
                hard_sigma = canvas * (
                    hard_sigma_start * (1.0 - frac) + 0.001 * frac
                )
                soft_sigma = canvas * (
                    soft_sigma_start * (1.0 - frac) + 0.002 * frac
                )

            if consec_below >= GRAD_CONSEC_REQUIRED:
                iter_i += 1
                break

        iter_i += 1

    return best_pos

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
        sizes_np, cw, ch, n_total,
    ):
        movable_step3 = large_mask & ~fixed_np
        return _adam_phase(
            benchmark, positions, movable_step3, sizes_np, cw, ch, n_total,
            device=self.device,
            iters=STEP3_ITERS,
            lr_frac=STEP3_LR_FRAC,
            lambda_start=STEP3_LAMBDA_START,
            lambda_end=STEP3_LAMBDA_END,
        )

    def _step6_adam_small_and_exact(
        self, benchmark, positions, large_mask, fixed_np,
        sizes_np, cw, ch, n_total,
    ):
        movable_step6 = (~large_mask) & (~fixed_np)
        positions = _adam_phase(
            benchmark, positions, movable_step6, sizes_np, cw, ch, n_total,
            device=self.device,
            iters=STEP6_ITERS,
            lr_frac=STEP6_LR_FRAC,
            lambda_start=STEP6_LAMBDA_START,
            lambda_end=STEP6_LAMBDA_END,
        )

        positions = _adam_exact_density_phase(
            benchmark, positions, movable_step6, sizes_np,
            cw, ch, n_total, device=self.device,
            iters=EXACT_DENSITY_ITERS,
            lr_frac=EXACT_DENSITY_LR_FRAC,
            lambda_start=EXACT_DENSITY_LAMBDA_START,
            lambda_end=EXACT_DENSITY_LAMBDA_END,
            density_target=EXACT_DENSITY_TARGET,
            overflow_weight=EXACT_DENSITY_OVERFLOW_WEIGHT,
        )
        return positions

    def _global_analytical_place(
        self, benchmark, positions, sizes_np, fixed_np, large_mask,
        n_hard, n_total, cw, ch,
    ):
        if GLOBAL_PLACE_ITERS <= 0:
            return positions

        movable_all = ~fixed_np
        out = _adam_phase(
            benchmark, positions, movable_all, sizes_np,
            cw, ch, n_total, device=self.device,
            iters=GLOBAL_PLACE_ITERS,
            lr_frac=GLOBAL_PLACE_LR_FRAC,
            lambda_start=GLOBAL_PLACE_LAMBDA_START,
            lambda_end=GLOBAL_PLACE_LAMBDA_END,
            density_target_slack=GLOBAL_PLACE_DENSITY_TARGET_SLACK,
            cong_weight=GLOBAL_PLACE_CONG_WEIGHT,
        )
        out[:n_hard] = _legalize_large_then_small(
            out[:n_hard], sizes_np[:n_hard], fixed_np[:n_hard],
            large_mask[:n_hard], cw, ch, gap=GAP,
        )
        return out

    def _all_macro_sa_polish(
        self, benchmark, positions, sizes_np, fixed_np, large_mask,
        n_hard, n_total, cw, ch, half_w, half_h, original_pos,
    ):
        pin_owner_np, pin_offset_np, pin_net_np, port_pos_np = (
            _build_pin_arrays_np(benchmark)
        )
        macro_to_nets = _build_macro_to_nets(pin_owner_np, pin_net_np, n_total)
        num_nets = len(benchmark.net_pin_nodes)
        nets_pin_indices = _net_pin_index_table(pin_net_np, num_nets)

        positions = self._finalize_positions(
            positions, half_w, half_h, fixed_np, original_pos, cw, ch
        )

        candidate_positions = _sa_all_macro_incremental(
            pos=positions.copy(),
            n_hard=n_hard,
            sizes_np=sizes_np,
            fixed_np=fixed_np,
            cw=cw, ch=ch,
            pin_owner_np=pin_owner_np,
            pin_offset_np=pin_offset_np,
            pin_net_np=pin_net_np,
            macro_to_nets=macro_to_nets,
            nets_pin_indices=nets_pin_indices,
            port_pos_np=port_pos_np,
            benchmark=benchmark,
            seed=self.seed + 505,
            congestion_weight=ALL_MACRO_SA_CONG_WEIGHT,
            density_weight=ALL_MACRO_SA_DENSITY_WEIGHT,
            max_iters=ALL_MACRO_SA_MAX_ITERS,
        )
        candidate_positions = self._finalize_positions(
            candidate_positions, half_w, half_h, fixed_np, original_pos, cw, ch,
        )
        if _has_hard_overlaps(
            candidate_positions[:n_hard], sizes_np[:n_hard],
            gap=0.0, use_float32=True,
        ):
            candidate_positions[:n_hard] = _legalize_large_then_small(
                candidate_positions[:n_hard],
                sizes_np[:n_hard],
                fixed_np[:n_hard],
                large_mask[:n_hard],
                cw, ch, gap=GAP,
            )
        return candidate_positions

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.use_deterministic_algorithms(self.deterministic)
        if self.deterministic:
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.set_num_threads(self.num_threads)
        try:
            torch.set_num_interop_threads(min(2, self.num_threads))
        except RuntimeError:
            pass

        n_hard = benchmark.num_hard_macros
        n_total = benchmark.num_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        core_area = cw * ch
        sizes_np = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
        fixed_np = benchmark.macro_fixed.cpu().numpy().astype(bool)
        original_pos = (
            benchmark.macro_positions.cpu().numpy().astype(np.float64).copy()
        )
        half_w = sizes_np[:, 0] / 2
        half_h = sizes_np[:, 1] / 2

        threshold = LARGE_MACRO_AREA_FRAC * core_area
        large_mask = np.zeros(n_total, dtype=bool)
        for i in range(n_hard):
            if not fixed_np[i] and (sizes_np[i, 0] * sizes_np[i, 1]) > threshold:
                large_mask[i] = True
        large_indices = np.where(large_mask)[0]

        positions = original_pos.copy()

        positions = self._step1_center_init(
            positions, fixed_np, half_w, half_h, n_total, cw, ch,
        )

        positions = self._step3_adam_large(
            benchmark, positions, large_mask, fixed_np,
            sizes_np, cw, ch, n_total,
        )

        positions = _legalize_to_boundary(
            positions, large_indices, sizes_np, cw, ch, gap=GAP,
        )

        positions = self._step6_adam_small_and_exact(
            benchmark, positions, large_mask, fixed_np,
            sizes_np, cw, ch, n_total,
        )

        positions[:n_hard] = _legalize_large_then_small(
            positions[:n_hard], sizes_np[:n_hard], fixed_np[:n_hard],
            large_mask[:n_hard], cw, ch, gap=GAP,
        )

        positions = self._global_analytical_place(
            benchmark, positions, sizes_np, fixed_np, large_mask,
            n_hard, n_total, cw, ch,
        )

        positions = self._all_macro_sa_polish(
            benchmark, positions, sizes_np, fixed_np, large_mask,
            n_hard, n_total, cw, ch, half_w, half_h, original_pos,
        )

        positions[:n_hard] = _legalize_large_then_small(
            positions[:n_hard], sizes_np[:n_hard], fixed_np[:n_hard],
            large_mask[:n_hard], cw, ch, gap=GAP,
        )
        positions[:, 0] = np.clip(
            positions[:, 0], half_w + 1e-6, cw - half_w - 1e-6
        )
        positions[:, 1] = np.clip(
            positions[:, 1], half_h + 1e-6, ch - half_h - 1e-6
        )
        for i in np.where(fixed_np)[0]:
            positions[i] = original_pos[i]

        return torch.tensor(positions, dtype=torch.float32)
