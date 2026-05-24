#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace {

constexpr double GAP = 0.0005;
constexpr double EPS = 1e-12;
constexpr int LEGALIZE_RING = 80;

constexpr double LARGE_MACRO_AREA_FRAC = 0.005;
constexpr double DENSITY_TOP_FRAC = 0.10;
constexpr double CONGESTION_TOP_FRAC = 0.05;
constexpr double HROUTING_ALLOC = 30.304;
constexpr double VROUTING_ALLOC = 71.304;
constexpr int SMOOTH_RANGE = 2;

constexpr int ALL_MACRO_SA_MAX_ITERS_DEFAULT = 200000;
constexpr int ALL_MACRO_SA_SHORT_ITERS_DEFAULT = 10000;
constexpr double ALL_MACRO_SA_CONG_WEIGHT = 0.8;
constexpr double ALL_MACRO_SA_DENSITY_WEIGHT = 0.5;

constexpr int CONGESTION_POLISH_ITERS_DEFAULT = 10000;
constexpr double CONGESTION_POLISH_ACCEPT_TOL = 1e-6;
constexpr double CONGESTION_POLISH_MIN_CONGESTION = 1.40;

constexpr int HOT_CONGESTION_SA_ITERS_DEFAULT = 200000;
constexpr double HOT_CONGESTION_SA_CONG_WEIGHT = 2.0;
constexpr double HOT_CONGESTION_SA_DENSITY_WEIGHT = 0.5;
constexpr double HOT_CONGESTION_ACCEPT_TOL = 1e-6;

constexpr int SWAP_SA_ITERS_DEFAULT = 10000;
constexpr int SWAP_SA_LOG_INTERVAL = 5000;
constexpr double SWAP_SA_ACCEPT_TOL = 1e-12;
constexpr double SWAP_SA_MIN_SUCCESS_RATE = 0.01;

constexpr double PROXY_WL_WEIGHT = 1.0;
constexpr double PROXY_DENSITY_WEIGHT = 0.5;
constexpr double PROXY_CONGESTION_WEIGHT = 0.5;

struct Point {
  double x = 0.0;
  double y = 0.0;
};

struct Macro {
  Point pos;
  double w = 0.0;
  double h = 0.0;
  bool fixed = false;
};

struct Pin {
  int owner = 0;
  double off_x = 0.0;
  double off_y = 0.0;
  int net = 0;
};

struct BenchmarkData {
  std::string name;
  double cw = 0.0;
  double ch = 0.0;
  int n_total = 0;
  int n_hard = 0;
  int n_ports = 0;
  int n_nets = 0;
  int grid_rows = 1;
  int grid_cols = 1;
  double hroutes_per_micron = 11.285;
  double vroutes_per_micron = 12.605;
  double wl_net_count = 0.0;
  std::vector<Macro> macros;
  std::vector<Point> ports;
  std::vector<Pin> pins;
  std::vector<double> net_weights;
  std::vector<std::vector<int>> nets_pin_indices;
  std::vector<std::vector<int>> macro_to_nets;
};

struct ProxyComponents {
  double wirelength = 0.0;
  double density = 0.0;
  double congestion = 0.0;
  double proxy = 0.0;
};

double clamp(double v, double lo, double hi) {
  if (lo > hi) {
    return 0.5 * (lo + hi);
  }
  return std::max(lo, std::min(hi, v));
}

int env_int(const char *name, int fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !*raw) {
    return fallback;
  }
  try {
    return std::max(0, std::stoi(raw));
  } catch (...) {
    return fallback;
  }
}

int default_thread_count() {
  unsigned int hc = std::thread::hardware_concurrency();
  if (hc == 0) {
    hc = 1;
  }
  return std::max(1, std::min(2, static_cast<int>(hc)));
}

int vibe_thread_count() {
  return std::max(1, env_int("VIBECPP_THREADS", default_thread_count()));
}

double mean_top(std::vector<double> values, int count) {
  if (values.empty()) {
    return 0.0;
  }
  count = std::max(1, std::min(count, static_cast<int>(values.size())));
  auto nth = values.end() - count;
  std::nth_element(values.begin(), nth, values.end());
  double sum = std::accumulate(nth, values.end(), 0.0);
  return sum / static_cast<double>(count);
}

double proxy_overall(double wl_norm, double density, double congestion) {
  return PROXY_WL_WEIGHT * wl_norm + PROXY_DENSITY_WEIGHT * density +
         PROXY_CONGESTION_WEIGHT * congestion;
}

BenchmarkData read_benchmark(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open input: " + path);
  }
  std::string magic;
  in >> magic;
  if (magic != "VIBECPP1") {
    throw std::runtime_error("unknown input format");
  }

  BenchmarkData b;
  std::string key;
  int pin_count = 0;
  while (in >> key) {
    if (key == "END") {
      break;
    }
    if (key == "name") {
      in >> b.name;
    } else if (key == "canvas") {
      in >> b.cw >> b.ch;
    } else if (key == "counts") {
      in >> b.n_total >> b.n_hard >> b.n_ports >> b.n_nets >> pin_count >>
          b.grid_rows >> b.grid_cols >> b.hroutes_per_micron >>
          b.vroutes_per_micron;
      b.macros.resize(b.n_total);
      b.ports.resize(b.n_ports);
      b.pins.resize(pin_count);
      b.net_weights.assign(b.n_nets, 1.0);
      b.wl_net_count = static_cast<double>(std::max(1, b.n_nets));
    } else if (key == "net_count") {
      in >> b.wl_net_count;
    } else if (key == "macros") {
      for (int i = 0; i < b.n_total; ++i) {
        int fixed = 0;
        in >> b.macros[i].pos.x >> b.macros[i].pos.y >> b.macros[i].w >>
            b.macros[i].h >> fixed;
        b.macros[i].fixed = fixed != 0;
      }
    } else if (key == "ports") {
      for (int i = 0; i < b.n_ports; ++i) {
        in >> b.ports[i].x >> b.ports[i].y;
      }
    } else if (key == "pins") {
      for (int i = 0; i < pin_count; ++i) {
        in >> b.pins[i].owner >> b.pins[i].off_x >> b.pins[i].off_y >>
            b.pins[i].net;
      }
    } else if (key == "net_weights") {
      for (int i = 0; i < b.n_nets; ++i) {
        in >> b.net_weights[i];
      }
    } else {
      throw std::runtime_error("unknown input key: " + key);
    }
  }

  b.grid_rows = std::max(1, b.grid_rows);
  b.grid_cols = std::max(1, b.grid_cols);
  b.nets_pin_indices.assign(b.n_nets, {});
  b.macro_to_nets.assign(b.n_total, {});
  std::vector<std::set<int>> seen(std::max(0, b.n_total));
  for (int pin_i = 0; pin_i < static_cast<int>(b.pins.size()); ++pin_i) {
    const Pin &p = b.pins[pin_i];
    if (0 <= p.net && p.net < b.n_nets) {
      b.nets_pin_indices[p.net].push_back(pin_i);
    }
    if (0 <= p.owner && p.owner < b.n_total && 0 <= p.net && p.net < b.n_nets) {
      if (seen[p.owner].insert(p.net).second) {
        b.macro_to_nets[p.owner].push_back(p.net);
      }
    }
  }
  return b;
}

std::vector<Point> macro_positions(const BenchmarkData &b) {
  std::vector<Point> pos(b.n_total);
  for (int i = 0; i < b.n_total; ++i) {
    pos[i] = b.macros[i].pos;
  }
  return pos;
}

void clamp_positions(const BenchmarkData &b, std::vector<Point> &pos,
                     const std::vector<Point> &original) {
  for (int i = 0; i < b.n_total; ++i) {
    const auto &m = b.macros[i];
    pos[i].x = clamp(pos[i].x, m.w * 0.5 + 1e-6, b.cw - m.w * 0.5 - 1e-6);
    pos[i].y = clamp(pos[i].y, m.h * 0.5 + 1e-6, b.ch - m.h * 0.5 - 1e-6);
    if (m.fixed) {
      pos[i] = original[i];
    }
  }
}

bool has_hard_overlaps(const BenchmarkData &b, const std::vector<Point> &pos,
                       double gap = 0.0) {
  for (int i = 0; i < b.n_hard; ++i) {
    for (int j = i + 1; j < b.n_hard; ++j) {
      double sep_x = (b.macros[i].w + b.macros[j].w) * 0.5 + gap;
      double sep_y = (b.macros[i].h + b.macros[j].h) * 0.5 + gap;
      if (std::abs(pos[i].x - pos[j].x) < sep_x &&
          std::abs(pos[i].y - pos[j].y) < sep_y) {
        return true;
      }
    }
  }
  return false;
}

bool has_hard_overlaps_float32(const BenchmarkData &b,
                               const std::vector<Point> &pos,
                               float gap = 0.0f) {
  for (int i = 0; i < b.n_hard; ++i) {
    for (int j = i + 1; j < b.n_hard; ++j) {
      float dx =
          std::abs(static_cast<float>(pos[i].x) - static_cast<float>(pos[j].x));
      float dy =
          std::abs(static_cast<float>(pos[i].y) - static_cast<float>(pos[j].y));
      float sep_x = (static_cast<float>(b.macros[i].w) +
                     static_cast<float>(b.macros[j].w)) *
                        0.5f +
                    gap;
      float sep_y = (static_cast<float>(b.macros[i].h) +
                     static_cast<float>(b.macros[j].h)) *
                        0.5f +
                    gap;
      if (dx < sep_x && dy < sep_y) {
        return true;
      }
    }
  }
  return false;
}

bool has_output_hard_overlaps(const BenchmarkData &b,
                              const std::vector<Point> &pos) {
  return has_hard_overlaps(b, pos, 0.0) ||
         has_hard_overlaps_float32(b, pos, 0.0f);
}

bool overlaps_any(
    double x, double y, double w, double h,
    const std::vector<std::tuple<double, double, double, double>> &placed,
    double gap) {
  double hw = w * 0.5;
  double hh = h * 0.5;
  for (const auto &box : placed) {
    double xl, xr, yb, yt;
    std::tie(xl, xr, yb, yt) = box;
    if (x + hw > xl - gap && x - hw < xr + gap && y + hh > yb - gap &&
        y - hh < yt + gap) {
      return true;
    }
  }
  return false;
}

bool try_place_on_boundary(
    int bound, double cx, double cy, double w, double h, double cw, double ch,
    const std::vector<std::tuple<double, double, double, double>> &placed,
    double gap, Point &out) {
  double hw = w * 0.5;
  double hh = h * 0.5;
  double fixed_coord = 0.0;
  double target = 0.0;
  double coord_min = 0.0;
  double coord_max = 0.0;
  bool slide_y = true;
  if (bound == 0) {
    fixed_coord = hw + gap;
    target = cy;
    coord_min = hh + gap;
    coord_max = ch - hh - gap;
    slide_y = true;
  } else if (bound == 1) {
    fixed_coord = cw - hw - gap;
    target = cy;
    coord_min = hh + gap;
    coord_max = ch - hh - gap;
    slide_y = true;
  } else if (bound == 2) {
    fixed_coord = hh + gap;
    target = cx;
    coord_min = hw + gap;
    coord_max = cw - hw - gap;
    slide_y = false;
  } else {
    fixed_coord = ch - hh - gap;
    target = cx;
    coord_min = hw + gap;
    coord_max = cw - hw - gap;
    slide_y = false;
  }
  target = clamp(target, coord_min, coord_max);
  double step = std::max(std::min(w, h) * 0.10, 0.05);
  for (int k = 0; k < 5000; ++k) {
    for (int sign : (k == 0 ? std::vector<int>{0} : std::vector<int>{1, -1})) {
      double slide = target + sign * k * step;
      if (slide < coord_min || slide > coord_max) {
        continue;
      }
      double x = slide_y ? fixed_coord : slide;
      double y = slide_y ? slide : fixed_coord;
      if (!overlaps_any(x, y, w, h, placed, gap)) {
        out = {x, y};
        return true;
      }
    }
  }
  return false;
}

void legalize_to_boundary(const BenchmarkData &b, std::vector<Point> &pos,
                          const std::vector<int> &large_indices,
                          double gap = GAP) {
  std::vector<int> order = large_indices;
  std::sort(order.begin(), order.end(), [&](int a, int c) {
    return b.macros[a].w * b.macros[a].h > b.macros[c].w * b.macros[c].h;
  });
  std::vector<std::tuple<double, double, double, double>> placed;
  for (int idx : order) {
    double cx = pos[idx].x;
    double cy = pos[idx].y;
    double w = b.macros[idx].w;
    double h = b.macros[idx].h;
    std::vector<std::pair<double, int>> bounds = {
        {cx - w * 0.5, 0},
        {b.cw - cx - w * 0.5, 1},
        {cy - h * 0.5, 2},
        {b.ch - cy - h * 0.5, 3},
    };
    std::sort(bounds.begin(), bounds.end());
    bool done = false;
    for (const auto &item : bounds) {
      Point p;
      if (try_place_on_boundary(item.second, cx, cy, w, h, b.cw, b.ch, placed,
                                gap, p)) {
        pos[idx] = p;
        placed.emplace_back(p.x - w * 0.5, p.x + w * 0.5, p.y - h * 0.5,
                            p.y + h * 0.5);
        done = true;
        break;
      }
    }
    if (!done) {
      pos[idx] = {b.cw * 0.5, b.ch * 0.5};
    }
  }
}

void spiral_legalize(const BenchmarkData &b, std::vector<Point> &pos,
                     const std::vector<bool> &fixed, int n, double gap,
                     int ring_cap) {
  std::vector<Point> legal = pos;
  std::vector<int> placed_idx;
  std::vector<bool> placed(n, false);
  for (int i = 0; i < n; ++i) {
    if (fixed[i]) {
      placed[i] = true;
      placed_idx.push_back(i);
    }
  }
  std::vector<int> order;
  for (int i = 0; i < n; ++i) {
    if (!fixed[i]) {
      order.push_back(i);
    }
  }
  std::sort(order.begin(), order.end(), [&](int a, int c) {
    return b.macros[a].w * b.macros[a].h > b.macros[c].w * b.macros[c].h;
  });

  auto overlaps = [&](int idx, double x, double y) {
    for (int j : placed_idx) {
      double sep_x = (b.macros[idx].w + b.macros[j].w) * 0.5 + gap;
      double sep_y = (b.macros[idx].h + b.macros[j].h) * 0.5 + gap;
      if (std::abs(x - legal[j].x) < sep_x &&
          std::abs(y - legal[j].y) < sep_y) {
        return true;
      }
    }
    return false;
  };

  for (int idx : order) {
    double hw = b.macros[idx].w * 0.5;
    double hh = b.macros[idx].h * 0.5;
    double x_min = hw + gap;
    double x_max = b.cw - hw - gap;
    double y_min = hh + gap;
    double y_max = b.ch - hh - gap;
    double x0 = clamp(pos[idx].x, x_min, x_max);
    double y0 = clamp(pos[idx].y, y_min, y_max);
    if (!overlaps(idx, x0, y0)) {
      legal[idx] = {x0, y0};
      placed[idx] = true;
      placed_idx.push_back(idx);
      continue;
    }
    double step =
        std::max(std::min(b.macros[idx].w, b.macros[idx].h) * 0.25, 0.05);
    Point best{x0, y0};
    double best_d = std::numeric_limits<double>::infinity();
    bool found = false;
    for (int r = 1; r <= ring_cap; ++r) {
      for (int dxm = -r; dxm <= r; ++dxm) {
        if (dxm == -r || dxm == r) {
          for (int dym = -r; dym <= r; ++dym) {
            double cx = clamp(x0 + dxm * step, x_min, x_max);
            double cy = clamp(y0 + dym * step, y_min, y_max);
            if (overlaps(idx, cx, cy)) {
              continue;
            }
            double d = (cx - x0) * (cx - x0) + (cy - y0) * (cy - y0);
            if (d < best_d) {
              best_d = d;
              best = {cx, cy};
              found = true;
            }
          }
        } else {
          for (int dym : {-r, r}) {
            double cx = clamp(x0 + dxm * step, x_min, x_max);
            double cy = clamp(y0 + dym * step, y_min, y_max);
            if (overlaps(idx, cx, cy)) {
              continue;
            }
            double d = (cx - x0) * (cx - x0) + (cy - y0) * (cy - y0);
            if (d < best_d) {
              best_d = d;
              best = {cx, cy};
              found = true;
            }
          }
        }
      }
      if (found) {
        break;
      }
    }
    legal[idx] = best;
    placed[idx] = true;
    placed_idx.push_back(idx);
  }
  pos = legal;
}

void push_legalize(const BenchmarkData &b, std::vector<Point> &pos,
                   const std::vector<bool> &fixed, int n, double gap,
                   int max_iters) {
  std::vector<Point> fixed_pos = pos;
  for (int iter = 0; iter < max_iters; ++iter) {
    bool any = false;
    std::vector<Point> force(n);
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        double dx = pos[i].x - pos[j].x;
        double dy = pos[i].y - pos[j].y;
        double sep_x = (b.macros[i].w + b.macros[j].w) * 0.5 + gap;
        double sep_y = (b.macros[i].h + b.macros[j].h) * 0.5 + gap;
        double ovr_x = sep_x - std::abs(dx);
        double ovr_y = sep_y - std::abs(dy);
        if (ovr_x <= 1e-12 || ovr_y <= 1e-12) {
          continue;
        }
        any = true;
        if (ovr_x <= ovr_y) {
          double s = dx >= 0.0 ? 1.0 : -1.0;
          force[i].x += 0.5 * ovr_x * s;
          force[j].x -= 0.5 * ovr_x * s;
        } else {
          double s = dy >= 0.0 ? 1.0 : -1.0;
          force[i].y += 0.5 * ovr_y * s;
          force[j].y -= 0.5 * ovr_y * s;
        }
      }
    }
    if (!any) {
      break;
    }
    for (int i = 0; i < n; ++i) {
      if (fixed[i]) {
        pos[i] = fixed_pos[i];
        continue;
      }
      pos[i].x = clamp(pos[i].x + force[i].x, b.macros[i].w * 0.5 + gap,
                       b.cw - b.macros[i].w * 0.5 - gap);
      pos[i].y = clamp(pos[i].y + force[i].y, b.macros[i].h * 0.5 + gap,
                       b.ch - b.macros[i].h * 0.5 - gap);
    }
  }
}

void robust_legalize_hard(const BenchmarkData &b, std::vector<Point> &pos,
                          double gap = GAP) {
  std::vector<bool> fixed(b.n_hard, false);
  for (int i = 0; i < b.n_hard; ++i) {
    fixed[i] = b.macros[i].fixed;
  }
  spiral_legalize(b, pos, fixed, b.n_hard, gap, LEGALIZE_RING);
  if (has_hard_overlaps(b, pos, 0.0)) {
    push_legalize(b, pos, fixed, b.n_hard, gap, 80);
    spiral_legalize(b, pos, fixed, b.n_hard, gap, 20);
  }
}

void legalize_large_then_small(const BenchmarkData &b, std::vector<Point> &pos,
                               const std::vector<bool> &large_mask,
                               double gap = GAP) {
  (void)large_mask;
  robust_legalize_hard(b, pos, gap);
  for (int attempt = 0; attempt < 3 && has_hard_overlaps(b, pos, 0.0);
       ++attempt) {
    std::vector<bool> fixed(b.n_hard, false);
    for (int i = 0; i < b.n_hard; ++i) {
      fixed[i] = b.macros[i].fixed;
    }
    push_legalize(b, pos, fixed, b.n_hard, gap, 300);
    spiral_legalize(b, pos, fixed, b.n_hard, gap, 160);
  }
}

Point pin_xy(const BenchmarkData &b, const std::vector<Point> &pos,
             int pin_idx) {
  const Pin &p = b.pins[pin_idx];
  if (p.owner < b.n_total) {
    return {pos[p.owner].x + p.off_x, pos[p.owner].y + p.off_y};
  }
  int port = p.owner - b.n_total;
  if (0 <= port && port < b.n_ports) {
    return b.ports[port];
  }
  return {};
}

double net_hpwl(const BenchmarkData &b, const std::vector<Point> &pos,
                int net_i) {
  if (net_i < 0 || net_i >= b.n_nets) {
    return 0.0;
  }
  const auto &pins = b.nets_pin_indices[net_i];
  if (pins.size() < 2) {
    return 0.0;
  }
  double min_x = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  for (int pin_idx : pins) {
    Point p = pin_xy(b, pos, pin_idx);
    min_x = std::min(min_x, p.x);
    max_x = std::max(max_x, p.x);
    min_y = std::min(min_y, p.y);
    max_y = std::max(max_y, p.y);
  }
  return (max_x - min_x) + (max_y - min_y);
}

class IncrementalDensityCost {
public:
  struct Token {
    double old_cost = 0.0;
    double new_cost = 0.0;
    int r0 = 0;
    int r1 = -1;
    int c0 = 0;
    int c1 = -1;
    std::vector<double> saved;
  };

  IncrementalDensityCost(const BenchmarkData &bench,
                         const std::vector<Point> &start_pos)
      : b(bench), pos(start_pos) {
    grid_w = b.cw / b.grid_cols;
    grid_h = b.ch / b.grid_rows;
    grid_area = std::max(grid_w * grid_h, EPS);
    occupied.assign(b.grid_rows * b.grid_cols, 0.0);
    for (int i = 0; i < b.n_total; ++i) {
      add_macro(i, pos[i].x, pos[i].y, 1.0);
    }
    top_count = std::max(1, static_cast<int>(std::floor(
                                b.grid_rows * b.grid_cols * DENSITY_TOP_FRAC)));
    current_cost = calc_cost();
  }

  double current() const { return current_cost; }

  std::pair<double, Token> begin_single_update(int idx, double nx, double ny) {
    Token tok;
    tok.old_cost = current_cost;
    auto old_rng = macro_cell_range(idx, pos[idx].x, pos[idx].y);
    auto new_rng = macro_cell_range(idx, nx, ny);
    if (old_rng.valid || new_rng.valid) {
      tok.r0 = std::min(old_rng.valid ? old_rng.r0 : new_rng.r0,
                        new_rng.valid ? new_rng.r0 : old_rng.r0);
      tok.r1 = std::max(old_rng.valid ? old_rng.r1 : new_rng.r1,
                        new_rng.valid ? new_rng.r1 : old_rng.r1);
      tok.c0 = std::min(old_rng.valid ? old_rng.c0 : new_rng.c0,
                        new_rng.valid ? new_rng.c0 : old_rng.c0);
      tok.c1 = std::max(old_rng.valid ? old_rng.c1 : new_rng.c1,
                        new_rng.valid ? new_rng.c1 : old_rng.c1);
      tok.saved.reserve((tok.r1 - tok.r0 + 1) * (tok.c1 - tok.c0 + 1));
      for (int r = tok.r0; r <= tok.r1; ++r) {
        for (int c = tok.c0; c <= tok.c1; ++c) {
          tok.saved.push_back(occupied[r * b.grid_cols + c]);
        }
      }
    }
    add_macro(idx, pos[idx].x, pos[idx].y, -1.0);
    add_macro(idx, nx, ny, 1.0);
    tok.new_cost = calc_cost();
    return {tok.new_cost, tok};
  }

  std::pair<double, Token> begin_two_update(int a, double ax, double ay, int c,
                                            double cx, double cy) {
    Token tok;
    tok.old_cost = current_cost;
    auto update_bounds = [&](const Range &rng) {
      if (!rng.valid) {
        return;
      }
      if (tok.r1 < tok.r0 || tok.c1 < tok.c0) {
        tok.r0 = rng.r0;
        tok.r1 = rng.r1;
        tok.c0 = rng.c0;
        tok.c1 = rng.c1;
        return;
      }
      tok.r0 = std::min(tok.r0, rng.r0);
      tok.r1 = std::max(tok.r1, rng.r1);
      tok.c0 = std::min(tok.c0, rng.c0);
      tok.c1 = std::max(tok.c1, rng.c1);
    };

    update_bounds(macro_cell_range(a, pos[a].x, pos[a].y));
    update_bounds(macro_cell_range(a, ax, ay));
    update_bounds(macro_cell_range(c, pos[c].x, pos[c].y));
    update_bounds(macro_cell_range(c, cx, cy));
    if (tok.r1 >= tok.r0 && tok.c1 >= tok.c0) {
      tok.saved.reserve((tok.r1 - tok.r0 + 1) * (tok.c1 - tok.c0 + 1));
      for (int r = tok.r0; r <= tok.r1; ++r) {
        for (int col = tok.c0; col <= tok.c1; ++col) {
          tok.saved.push_back(occupied[r * b.grid_cols + col]);
        }
      }
    }
    add_macro(a, pos[a].x, pos[a].y, -1.0);
    add_macro(c, pos[c].x, pos[c].y, -1.0);
    add_macro(a, ax, ay, 1.0);
    add_macro(c, cx, cy, 1.0);
    tok.new_cost = calc_cost();
    return {tok.new_cost, tok};
  }

  void accept_single(const Token &tok, int idx, double nx, double ny) {
    current_cost = tok.new_cost;
    pos[idx] = {nx, ny};
  }

  void accept_two(const Token &tok, int a, double ax, double ay, int c,
                  double cx, double cy) {
    current_cost = tok.new_cost;
    pos[a] = {ax, ay};
    pos[c] = {cx, cy};
  }

  void reject(const Token &tok) {
    if (tok.r1 >= tok.r0 && tok.c1 >= tok.c0) {
      size_t k = 0;
      for (int r = tok.r0; r <= tok.r1; ++r) {
        for (int c = tok.c0; c <= tok.c1; ++c) {
          occupied[r * b.grid_cols + c] = tok.saved[k++];
        }
      }
    }
    current_cost = tok.old_cost;
  }

private:
  struct Range {
    bool valid = false;
    int r0 = 0;
    int r1 = 0;
    int c0 = 0;
    int c1 = 0;
  };

  const BenchmarkData &b;
  std::vector<Point> pos;
  double grid_w = 1.0;
  double grid_h = 1.0;
  double grid_area = 1.0;
  int top_count = 1;
  std::vector<double> occupied;
  double current_cost = 0.0;

  Range macro_cell_range(int idx, double cx, double cy) const {
    double w = b.macros[idx].w;
    double h = b.macros[idx].h;
    double lx = cx - w * 0.5;
    double ux = cx + w * 0.5;
    double ly = cy - h * 0.5;
    double uy = cy + h * 0.5;
    if (w <= 0.0 || h <= 0.0 || ux <= 0.0 || uy <= 0.0 || lx >= b.cw ||
        ly >= b.ch) {
      return {};
    }
    Range out;
    out.valid = true;
    out.c0 = std::max(0, std::min(b.grid_cols - 1,
                                  static_cast<int>(std::floor(lx / grid_w))));
    out.c1 = std::max(0, std::min(b.grid_cols - 1,
                                  static_cast<int>(std::floor(ux / grid_w))));
    out.r0 = std::max(0, std::min(b.grid_rows - 1,
                                  static_cast<int>(std::floor(ly / grid_h))));
    out.r1 = std::max(0, std::min(b.grid_rows - 1,
                                  static_cast<int>(std::floor(uy / grid_h))));
    return out;
  }

  void add_macro(int idx, double cx, double cy, double sign) {
    double w = b.macros[idx].w;
    double h = b.macros[idx].h;
    double lx = cx - w * 0.5;
    double ux = cx + w * 0.5;
    double ly = cy - h * 0.5;
    double uy = cy + h * 0.5;
    if (w <= 0.0 || h <= 0.0 || ux <= 0.0 || uy <= 0.0 || lx >= b.cw ||
        ly >= b.ch) {
      return;
    }
    int c0 = std::max(0, std::min(b.grid_cols - 1,
                                  static_cast<int>(std::floor(lx / grid_w))));
    int c1 = std::max(0, std::min(b.grid_cols - 1,
                                  static_cast<int>(std::floor(ux / grid_w))));
    int r0 = std::max(0, std::min(b.grid_rows - 1,
                                  static_cast<int>(std::floor(ly / grid_h))));
    int r1 = std::max(0, std::min(b.grid_rows - 1,
                                  static_cast<int>(std::floor(uy / grid_h))));
    for (int r = r0; r <= r1; ++r) {
      double y0 = r * grid_h;
      double y1 = y0 + grid_h;
      double oy = std::max(0.0, std::min(uy, y1) - std::max(ly, y0));
      if (oy <= 0.0) {
        continue;
      }
      for (int c = c0; c <= c1; ++c) {
        double x0 = c * grid_w;
        double x1 = x0 + grid_w;
        double ox = std::max(0.0, std::min(ux, x1) - std::max(lx, x0));
        if (ox > 0.0) {
          occupied[r * b.grid_cols + c] += sign * ox * oy;
        }
      }
    }
  }

  double calc_cost() const {
    std::vector<double> density;
    density.reserve(occupied.size());
    for (double area : occupied) {
      double d = area / grid_area;
      if (d > 1e-9) {
        density.push_back(d);
      }
    }
    if (density.empty()) {
      return 0.0;
    }
    if (occupied.size() < 10) {
      double sum = std::accumulate(density.begin(), density.end(), 0.0);
      return 0.5 * sum / density.size();
    }
    int take = std::min(top_count, static_cast<int>(density.size()));
    auto nth = density.end() - take;
    std::nth_element(density.begin(), nth, density.end());
    double sum = std::accumulate(nth, density.end(), 0.0);
    return 0.5 * sum / static_cast<double>(top_count);
  }
};

struct VSeg {
  int row = 0;
  int c_lo = 0;
  int c_hi = 0;
  double val = 0.0;
};

struct HSeg {
  int r_lo = 0;
  int r_hi = 0;
  int col = 0;
  double val = 0.0;
};

struct PSeg {
  int row = 0;
  int col = 0;
  double val = 0.0;
};

struct NetContrib {
  std::vector<VSeg> v;
  std::vector<HSeg> h;
};

struct MacroContrib {
  std::vector<PSeg> v;
  std::vector<PSeg> h;
};

struct RouteKey {
  bool valid = false;
  std::pair<int, int> source{0, 0};
  std::vector<std::pair<int, int>> cells;

  bool operator==(const RouteKey &other) const {
    return valid == other.valid && source == other.source &&
           cells == other.cells;
  }
};

class IncrementalCongestionCost {
public:
  struct Token {
    double old_cost = 0.0;
    double new_cost = 0.0;
    std::vector<std::tuple<int, NetContrib, NetContrib, RouteKey, RouteKey>>
        nets;
    std::vector<std::tuple<int, MacroContrib, MacroContrib>> macros;
    std::vector<double> v_delta;
    std::vector<double> h_delta;
    bool has_route_delta = false;
  };

  IncrementalCongestionCost(const BenchmarkData &bench,
                            std::vector<Point> &positions)
      : b(bench), pos(positions) {
    grid_w = b.cw / b.grid_cols;
    grid_h = b.ch / b.grid_rows;
    grid_v_routes = std::max(grid_w * b.vroutes_per_micron, EPS);
    grid_h_routes = std::max(grid_h * b.hroutes_per_micron, EPS);
    inv_grid_v_routes = 1.0 / grid_v_routes;
    inv_grid_h_routes = 1.0 / grid_h_routes;
    grid_size = b.grid_rows * b.grid_cols;
    v_route.assign(grid_size, 0.0);
    h_route.assign(grid_size, 0.0);
    v_macro.assign(grid_size, 0.0);
    h_macro.assign(grid_size, 0.0);
    abu_top_count = std::max(
        1, static_cast<int>(std::floor(2.0 * grid_size * CONGESTION_TOP_FRAC)));
    for (int c = 0; c < b.grid_cols; ++c) {
      int lo = std::max(0, c - SMOOTH_RANGE);
      int hi = std::min(b.grid_cols, c + SMOOTH_RANGE + 1);
      v_smooth.emplace_back(lo, hi, inv_grid_v_routes / std::max(1, hi - lo));
    }
    for (int r = 0; r < b.grid_rows; ++r) {
      int lo = std::max(0, r - SMOOTH_RANGE);
      int hi = std::min(b.grid_rows, r + SMOOTH_RANGE + 1);
      h_smooth.emplace_back(lo, hi, inv_grid_h_routes / std::max(1, hi - lo));
    }
    net_contribs.resize(b.n_nets);
    net_route_keys.resize(b.n_nets);
    for (int ni = 0; ni < b.n_nets; ++ni) {
      auto pair = net_contrib(ni);
      net_contribs[ni] = pair.first;
      net_route_keys[ni] = pair.second;
      apply_net(net_contribs[ni], 1.0);
    }
    macro_contribs.resize(b.n_hard);
    for (int mi = 0; mi < b.n_hard; ++mi) {
      macro_contribs[mi] = macro_contrib(mi);
      apply_macro(macro_contribs[mi], 1.0);
    }
    current_cost = abu_cost();
  }

  double current() const { return current_cost; }

  std::pair<double, Token>
  begin_single_update(int moved_macro, const std::vector<int> &affected_nets) {
    Token tok;
    tok.old_cost = current_cost;
    std::vector<double> v_delta(grid_size, 0.0);
    std::vector<double> h_delta(grid_size, 0.0);
    std::vector<int> nets = affected_nets;
    std::sort(nets.begin(), nets.end());
    nets.erase(std::unique(nets.begin(), nets.end()), nets.end());
    for (int ni : nets) {
      if (ni < 0 || ni >= b.n_nets) {
        continue;
      }
      RouteKey old_key = net_route_keys[ni];
      auto key_data = net_route_key_data(ni);
      RouteKey new_key = std::get<0>(key_data);
      if (new_key == old_key) {
        continue;
      }
      NetContrib old = net_contribs[ni];
      NetContrib next;
      if (new_key.valid) {
        next = net_contrib_from_gcells(ni, std::get<1>(key_data),
                                       std::get<2>(key_data));
      }
      accum_net_delta(old, -1.0, v_delta, h_delta);
      accum_net_delta(next, 1.0, v_delta, h_delta);
      tok.nets.emplace_back(ni, old, next, old_key, new_key);
    }
    if (!tok.nets.empty()) {
      for (int i = 0; i < grid_size; ++i) {
        v_route[i] += v_delta[i];
        h_route[i] += h_delta[i];
      }
      tok.has_route_delta = true;
      tok.v_delta = std::move(v_delta);
      tok.h_delta = std::move(h_delta);
    }
    if (0 <= moved_macro && moved_macro < b.n_hard) {
      MacroContrib old = macro_contribs[moved_macro];
      MacroContrib next = macro_contrib(moved_macro);
      apply_macro(old, -1.0);
      apply_macro(next, 1.0);
      tok.macros.emplace_back(moved_macro, old, next);
    }
    tok.new_cost = abu_cost();
    return {tok.new_cost, tok};
  }

  std::pair<double, Token>
  begin_multi_update(std::vector<int> moved_macros,
                     const std::vector<int> &affected_nets) {
    Token tok;
    tok.old_cost = current_cost;
    std::vector<double> v_delta(grid_size, 0.0);
    std::vector<double> h_delta(grid_size, 0.0);
    std::vector<int> nets = affected_nets;
    std::sort(nets.begin(), nets.end());
    nets.erase(std::unique(nets.begin(), nets.end()), nets.end());
    for (int ni : nets) {
      if (ni < 0 || ni >= b.n_nets) {
        continue;
      }
      RouteKey old_key = net_route_keys[ni];
      auto key_data = net_route_key_data(ni);
      RouteKey new_key = std::get<0>(key_data);
      if (new_key == old_key) {
        continue;
      }
      NetContrib old = net_contribs[ni];
      NetContrib next;
      if (new_key.valid) {
        next = net_contrib_from_gcells(ni, std::get<1>(key_data),
                                       std::get<2>(key_data));
      }
      accum_net_delta(old, -1.0, v_delta, h_delta);
      accum_net_delta(next, 1.0, v_delta, h_delta);
      tok.nets.emplace_back(ni, old, next, old_key, new_key);
    }
    if (!tok.nets.empty()) {
      for (int i = 0; i < grid_size; ++i) {
        v_route[i] += v_delta[i];
        h_route[i] += h_delta[i];
      }
      tok.has_route_delta = true;
      tok.v_delta = std::move(v_delta);
      tok.h_delta = std::move(h_delta);
    }

    std::sort(moved_macros.begin(), moved_macros.end());
    moved_macros.erase(std::unique(moved_macros.begin(), moved_macros.end()),
                       moved_macros.end());
    for (int macro_i : moved_macros) {
      if (0 <= macro_i && macro_i < b.n_hard) {
        MacroContrib old = macro_contribs[macro_i];
        MacroContrib next = macro_contrib(macro_i);
        apply_macro(old, -1.0);
        apply_macro(next, 1.0);
        tok.macros.emplace_back(macro_i, old, next);
      }
    }
    tok.new_cost = abu_cost();
    return {tok.new_cost, tok};
  }

  void accept(const Token &tok) {
    for (const auto &item : tok.nets) {
      int ni;
      NetContrib old_c, new_c;
      RouteKey old_k, new_k;
      std::tie(ni, old_c, new_c, old_k, new_k) = item;
      (void)old_c;
      (void)old_k;
      net_contribs[ni] = new_c;
      net_route_keys[ni] = new_k;
    }
    for (const auto &item : tok.macros) {
      int mi;
      MacroContrib old_c, new_c;
      std::tie(mi, old_c, new_c) = item;
      (void)old_c;
      macro_contribs[mi] = new_c;
    }
    current_cost = tok.new_cost;
  }

  void reject(const Token &tok) {
    if (tok.has_route_delta) {
      for (int i = 0; i < grid_size; ++i) {
        v_route[i] -= tok.v_delta[i];
        h_route[i] -= tok.h_delta[i];
      }
    }
    for (const auto &item : tok.macros) {
      int mi;
      MacroContrib old_c, new_c;
      std::tie(mi, old_c, new_c) = item;
      (void)mi;
      apply_macro(new_c, -1.0);
      apply_macro(old_c, 1.0);
    }
    current_cost = tok.old_cost;
  }

  std::vector<int> hot_congestion_nets() const {
    std::vector<double> combined;
    combined.reserve(2 * grid_size);
    for (int i = 0; i < grid_size; ++i) {
      combined.push_back(v_route[i] + v_macro[i]);
    }
    for (int i = 0; i < grid_size; ++i) {
      combined.push_back(h_route[i] + h_macro[i]);
    }

    int take = std::min(abu_top_count, static_cast<int>(combined.size()));
    if (take <= 0) {
      return {};
    }
    std::vector<int> order(combined.size());
    std::iota(order.begin(), order.end(), 0);
    auto hotter = [&](int a, int c) { return combined[a] > combined[c]; };
    if (take < static_cast<int>(order.size())) {
      std::nth_element(order.begin(), order.begin() + take, order.end(),
                       hotter);
    } else {
      std::sort(order.begin(), order.end(), hotter);
    }

    std::vector<char> hot_v(grid_size, 0);
    std::vector<char> hot_h(grid_size, 0);
    for (int i = 0; i < take; ++i) {
      int flat = order[i];
      if (flat < grid_size) {
        hot_v[flat] = 1;
      } else {
        hot_h[flat - grid_size] = 1;
      }
    }

    std::vector<int> hot_nets;
    hot_nets.reserve(b.n_nets);
    for (int ni = 0; ni < b.n_nets; ++ni) {
      bool touches_hot = false;
      for (const auto &s : net_contribs[ni].v) {
        for (int c = s.c_lo; c < s.c_hi; ++c) {
          if (hot_v[idx(s.row, c)]) {
            touches_hot = true;
            break;
          }
        }
        if (touches_hot)
          break;
      }
      if (!touches_hot) {
        for (const auto &s : net_contribs[ni].h) {
          for (int r = s.r_lo; r < s.r_hi; ++r) {
            if (hot_h[idx(r, s.col)]) {
              touches_hot = true;
              break;
            }
          }
          if (touches_hot)
            break;
        }
      }
      if (touches_hot) {
        hot_nets.push_back(ni);
      }
    }
    return hot_nets;
  }

private:
  const BenchmarkData &b;
  std::vector<Point> &pos;
  double grid_w = 1.0;
  double grid_h = 1.0;
  double grid_v_routes = 1.0;
  double grid_h_routes = 1.0;
  double inv_grid_v_routes = 1.0;
  double inv_grid_h_routes = 1.0;
  int grid_size = 1;
  int abu_top_count = 1;
  std::vector<std::tuple<int, int, double>> v_smooth;
  std::vector<std::tuple<int, int, double>> h_smooth;
  std::vector<double> v_route;
  std::vector<double> h_route;
  std::vector<double> v_macro;
  std::vector<double> h_macro;
  std::vector<NetContrib> net_contribs;
  std::vector<RouteKey> net_route_keys;
  std::vector<MacroContrib> macro_contribs;
  double current_cost = 0.0;

  int idx(int r, int c) const { return r * b.grid_cols + c; }

  std::pair<int, int> grid_cell(double x, double y) const {
    int row = std::max(
        0, std::min(b.grid_rows - 1, static_cast<int>(std::floor(y / grid_h))));
    int col = std::max(
        0, std::min(b.grid_cols - 1, static_cast<int>(std::floor(x / grid_w))));
    return {row, col};
  }

  void add_v_route(NetContrib &contrib, int row, int col, double weight) const {
    if (row < 0 || row >= b.grid_rows || col < 0 || col >= b.grid_cols) {
      return;
    }
    int lo, hi;
    double scale;
    std::tie(lo, hi, scale) = v_smooth[col];
    contrib.v.push_back({row, lo, hi, weight * scale});
  }

  void add_h_route(NetContrib &contrib, int row, int col, double weight) const {
    if (row < 0 || row >= b.grid_rows || col < 0 || col >= b.grid_cols) {
      return;
    }
    int lo, hi;
    double scale;
    std::tie(lo, hi, scale) = h_smooth[row];
    contrib.h.push_back({lo, hi, col, weight * scale});
  }

  void route_two(NetContrib &contrib, std::pair<int, int> source,
                 const std::vector<std::pair<int, int>> &cells,
                 double weight) const {
    if (cells.size() < 2) {
      return;
    }
    std::pair<int, int> sink = cells[0] == source ? cells[1] : cells[0];
    int row_min = std::min(sink.first, source.first);
    int row_max = std::max(sink.first, source.first);
    int col_min = std::min(sink.second, source.second);
    int col_max = std::max(sink.second, source.second);
    for (int col = col_min; col < col_max; ++col) {
      add_h_route(contrib, source.first, col, weight);
    }
    for (int row = row_min; row < row_max; ++row) {
      add_v_route(contrib, row, sink.second, weight);
    }
  }

  void route_l(NetContrib &contrib, std::vector<std::pair<int, int>> cells,
               double weight) const {
    std::sort(cells.begin(), cells.end(), [](auto a, auto c) {
      return std::make_pair(a.second, a.first) <
             std::make_pair(c.second, c.first);
    });
    int y1 = cells[0].first, x1 = cells[0].second;
    int y2 = cells[1].first, x2 = cells[1].second;
    int y3 = cells[2].first, x3 = cells[2].second;
    for (int col = x1; col < x2; ++col)
      add_h_route(contrib, y1, col, weight);
    for (int col = x2; col < x3; ++col)
      add_h_route(contrib, y2, col, weight);
    for (int row = std::min(y1, y2); row < std::max(y1, y2); ++row)
      add_v_route(contrib, row, x2, weight);
    for (int row = std::min(y2, y3); row < std::max(y2, y3); ++row)
      add_v_route(contrib, row, x3, weight);
  }

  void route_t(NetContrib &contrib, std::vector<std::pair<int, int>> cells,
               double weight) const {
    std::sort(cells.begin(), cells.end());
    int y1 = cells[0].first, x1 = cells[0].second;
    int y2 = cells[1].first, x2 = cells[1].second;
    int y3 = cells[2].first, x3 = cells[2].second;
    int xmin = std::min({x1, x2, x3});
    int xmax = std::max({x1, x2, x3});
    for (int col = xmin; col < xmax; ++col)
      add_h_route(contrib, y2, col, weight);
    for (int row = std::min(y1, y2); row < std::max(y1, y2); ++row)
      add_v_route(contrib, row, x1, weight);
    for (int row = std::min(y2, y3); row < std::max(y2, y3); ++row)
      add_v_route(contrib, row, x3, weight);
  }

  void route_three(NetContrib &contrib, std::vector<std::pair<int, int>> cells,
                   double weight) const {
    std::sort(cells.begin(), cells.end(), [](auto a, auto c) {
      return std::make_pair(a.second, a.first) <
             std::make_pair(c.second, c.first);
    });
    int y1 = cells[0].first, x1 = cells[0].second;
    int y2 = cells[1].first, x2 = cells[1].second;
    int y3 = cells[2].first, x3 = cells[2].second;
    if (x1 < x2 && x2 < x3 && std::min(y1, y3) < y2 && std::max(y1, y3) > y2) {
      route_l(contrib, cells, weight);
    } else if (x2 == x3 && x1 < x2 && y1 < std::min(y2, y3)) {
      for (int col = x1; col < x2; ++col)
        add_h_route(contrib, y1, col, weight);
      for (int row = y1; row < std::max(y2, y3); ++row)
        add_v_route(contrib, row, x2, weight);
    } else if (y2 == y3) {
      for (int col = x1; col < x2; ++col)
        add_h_route(contrib, y1, col, weight);
      for (int col = x2; col < x3; ++col)
        add_h_route(contrib, y2, col, weight);
      for (int row = std::min(y2, y1); row < std::max(y2, y1); ++row)
        add_v_route(contrib, row, x2, weight);
    } else {
      route_t(contrib, cells, weight);
    }
  }

  std::pair<std::pair<int, int>, std::vector<std::pair<int, int>>>
  get_net_gcells(int net_i) const {
    std::set<std::pair<int, int>> unique_cells;
    std::pair<int, int> source{0, 0};
    bool have_source = false;
    for (int pin_idx : b.nets_pin_indices[net_i]) {
      Point p = pin_xy(b, pos, pin_idx);
      auto cell = grid_cell(p.x, p.y);
      if (!have_source) {
        source = cell;
        have_source = true;
      }
      unique_cells.insert(cell);
    }
    return {source, std::vector<std::pair<int, int>>(unique_cells.begin(),
                                                     unique_cells.end())};
  }

  std::tuple<RouteKey, std::pair<int, int>, std::vector<std::pair<int, int>>>
  net_route_key_data(int net_i) const {
    if (b.nets_pin_indices[net_i].size() < 2) {
      return {RouteKey{}, {0, 0}, {}};
    }
    auto data = get_net_gcells(net_i);
    RouteKey key;
    key.valid = true;
    key.source = data.first;
    key.cells = data.second;
    return {key, data.first, data.second};
  }

  NetContrib
  net_contrib_from_gcells(int net_i, std::pair<int, int> source,
                          const std::vector<std::pair<int, int>> &cells) const {
    NetContrib contrib;
    double weight =
        (0 <= net_i && net_i < static_cast<int>(b.net_weights.size()))
            ? b.net_weights[net_i]
            : 1.0;
    if (cells.size() == 2) {
      route_two(contrib, source, cells, weight);
    } else if (cells.size() == 3) {
      route_three(contrib, cells, weight);
    } else if (cells.size() > 3) {
      for (auto cell : cells) {
        if (cell != source) {
          std::vector<std::pair<int, int>> pair_cells{source, cell};
          route_two(contrib, source, pair_cells, weight);
        }
      }
    }
    return contrib;
  }

  std::pair<NetContrib, RouteKey> net_contrib(int net_i) const {
    auto data = net_route_key_data(net_i);
    RouteKey key = std::get<0>(data);
    if (!key.valid) {
      return {NetContrib{}, key};
    }
    return {
        net_contrib_from_gcells(net_i, std::get<1>(data), std::get<2>(data)),
        key};
  }

  MacroContrib macro_contrib(int macro_i) const {
    MacroContrib out;
    double x = pos[macro_i].x;
    double y = pos[macro_i].y;
    double w = b.macros[macro_i].w;
    double h = b.macros[macro_i].h;
    double x0 = x - w * 0.5;
    double x1 = x + w * 0.5;
    double y0 = y - h * 0.5;
    double y1 = y + h * 0.5;
    auto bl = grid_cell(x0, y0);
    auto ur = grid_cell(x1, y1);
    bool partial_vertical = false;
    bool partial_horizontal = false;
    for (int row = bl.first; row <= ur.first; ++row) {
      for (int col = bl.second; col <= ur.second; ++col) {
        double gx0 = col * grid_w;
        double gx1 = (col + 1) * grid_w;
        double gy0 = row * grid_h;
        double gy1 = (row + 1) * grid_h;
        double x_dist = std::max(0.0, std::min(x1, gx1) - std::max(x0, gx0));
        double y_dist = std::max(0.0, std::min(y1, gy1) - std::max(y0, gy0));
        if (ur.first != bl.first && (row == bl.first || row == ur.first) &&
            std::abs(y_dist - grid_h) > 1e-5) {
          partial_vertical = true;
        }
        if (ur.second != bl.second && (col == bl.second || col == ur.second) &&
            std::abs(x_dist - grid_w) > 1e-5) {
          partial_horizontal = true;
        }
        out.v.push_back(
            {row, col, x_dist * VROUTING_ALLOC * inv_grid_v_routes});
        out.h.push_back(
            {row, col, y_dist * HROUTING_ALLOC * inv_grid_h_routes});
      }
    }
    if (partial_vertical) {
      int row = ur.first;
      for (int col = bl.second; col <= ur.second; ++col) {
        double gx0 = col * grid_w;
        double gx1 = (col + 1) * grid_w;
        double x_dist = std::max(0.0, std::min(x1, gx1) - std::max(x0, gx0));
        out.v.push_back(
            {row, col, -x_dist * VROUTING_ALLOC * inv_grid_v_routes});
      }
    }
    if (partial_horizontal) {
      int col = ur.second;
      for (int row = bl.first; row <= ur.first; ++row) {
        double gy0 = row * grid_h;
        double gy1 = (row + 1) * grid_h;
        double y_dist = std::max(0.0, std::min(y1, gy1) - std::max(y0, gy0));
        out.h.push_back(
            {row, col, -y_dist * HROUTING_ALLOC * inv_grid_h_routes});
      }
    }
    return out;
  }

  void apply_net(const NetContrib &contrib, double sign) {
    for (const auto &s : contrib.v) {
      for (int c = s.c_lo; c < s.c_hi; ++c) {
        v_route[idx(s.row, c)] += sign * s.val;
      }
    }
    for (const auto &s : contrib.h) {
      for (int r = s.r_lo; r < s.r_hi; ++r) {
        h_route[idx(r, s.col)] += sign * s.val;
      }
    }
  }

  void accum_net_delta(const NetContrib &contrib, double sign,
                       std::vector<double> &vd, std::vector<double> &hd) const {
    for (const auto &s : contrib.v) {
      for (int c = s.c_lo; c < s.c_hi; ++c) {
        vd[idx(s.row, c)] += sign * s.val;
      }
    }
    for (const auto &s : contrib.h) {
      for (int r = s.r_lo; r < s.r_hi; ++r) {
        hd[idx(r, s.col)] += sign * s.val;
      }
    }
  }

  void apply_macro(const MacroContrib &contrib, double sign) {
    for (const auto &p : contrib.v) {
      v_macro[idx(p.row, p.col)] += sign * p.val;
    }
    for (const auto &p : contrib.h) {
      h_macro[idx(p.row, p.col)] += sign * p.val;
    }
  }

  double abu_cost() const {
    std::vector<double> combined;
    combined.reserve(2 * grid_size);
    for (int i = 0; i < grid_size; ++i) {
      combined.push_back(v_route[i] + v_macro[i]);
    }
    for (int i = 0; i < grid_size; ++i) {
      combined.push_back(h_route[i] + h_macro[i]);
    }
    return mean_top(std::move(combined), abu_top_count);
  }
};

ProxyComponents compute_proxy(const BenchmarkData &b, std::vector<Point> &pos) {
  double weighted_hpwl = 0.0;
  for (int ni = 0; ni < b.n_nets; ++ni) {
    double w =
        ni < static_cast<int>(b.net_weights.size()) ? b.net_weights[ni] : 1.0;
    weighted_hpwl += net_hpwl(b, pos, ni) * w;
  }
  double norm_net_count = b.wl_net_count > 0.0
                              ? b.wl_net_count
                              : static_cast<double>(std::max(1, b.n_nets));
  double norm = std::max((b.cw + b.ch) * norm_net_count, EPS);
  IncrementalDensityCost den(b, pos);
  IncrementalCongestionCost cong(b, pos);
  ProxyComponents out;
  out.wirelength = weighted_hpwl / norm;
  out.density = den.current();
  out.congestion = cong.current();
  out.proxy = proxy_overall(out.wirelength, out.density, out.congestion);
  return out;
}

struct Bounds {
  double x0 = 0.0;
  double x1 = 0.0;
  double y0 = 0.0;
  double y1 = 0.0;
  bool enabled = false;
};

std::vector<Point> sa_all_macro_incremental(
    const BenchmarkData &b, std::vector<Point> pos,
    const std::vector<bool> &fixed_override, int seed, double congestion_weight,
    double density_weight, double hard_move_prob, double hard_sigma_start,
    double soft_sigma_start, int max_iters, Bounds bounds, bool log_progress,
    const std::vector<bool> *allowed_mask = nullptr,
    const std::vector<double> *move_scales_override = nullptr) {
  std::mt19937_64 rng(static_cast<uint64_t>(seed));
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::normal_distribution<double> normal(0.0, 1.0);

  std::vector<int> movable_hard;
  std::vector<int> movable_soft;
  for (int i = 0; i < b.n_hard; ++i) {
    bool fixed = fixed_override.empty() ? b.macros[i].fixed : fixed_override[i];
    bool allowed =
        allowed_mask == nullptr ||
        (i < static_cast<int>(allowed_mask->size()) && (*allowed_mask)[i]);
    if (!fixed && allowed)
      movable_hard.push_back(i);
  }
  for (int i = b.n_hard; i < b.n_total; ++i) {
    bool fixed = fixed_override.empty() ? b.macros[i].fixed : fixed_override[i];
    bool allowed =
        allowed_mask == nullptr ||
        (i < static_cast<int>(allowed_mask->size()) && (*allowed_mask)[i]);
    if (!fixed && allowed)
      movable_soft.push_back(i);
  }
  if (movable_hard.empty() && movable_soft.empty()) {
    return pos;
  }

  std::vector<double> cur_hpwl(b.n_nets, 0.0);
  for (int ni = 0; ni < b.n_nets; ++ni) {
    double w =
        ni < static_cast<int>(b.net_weights.size()) ? b.net_weights[ni] : 1.0;
    cur_hpwl[ni] = net_hpwl(b, pos, ni) * w;
  }
  IncrementalDensityCost den_eval(b, pos);
  IncrementalCongestionCost cong_eval(b, pos);

  double norm_net_count = b.wl_net_count > 0.0
                              ? b.wl_net_count
                              : static_cast<double>(std::max(1, b.n_nets));
  double wl_norm = std::max((b.cw + b.ch) * norm_net_count, EPS);
  double cur_cost = std::accumulate(cur_hpwl.begin(), cur_hpwl.end(), 0.0) +
                    (congestion_weight * cong_eval.current() +
                     density_weight * den_eval.current()) *
                        wl_norm;
  double best_cost = cur_cost;
  std::vector<Point> best_pos = pos;

  double canvas = std::max(b.cw, b.ch);
  double temp_start = canvas * 0.1;
  double temp_end = canvas * 0.00005;
  double temp_ratio = temp_end / temp_start;
  double temp = temp_start;
  double hard_sigma = canvas * hard_sigma_start;
  double soft_sigma = canvas * soft_sigma_start;

  const double grad_stop_threshold = 50.0;
  const double grad_ref = grad_stop_threshold * 10.0;
  int consec_below = 0;
  std::deque<double> recent_costs;
  static const std::vector<double> default_move_scales{1.0, 1.5, 0.5};
  const std::vector<double> &move_scales =
      (move_scales_override != nullptr && !move_scales_override->empty())
          ? *move_scales_override
          : default_move_scales;
  bool gate_default_rescue_scales =
      move_scales_override == nullptr || move_scales_override->empty();

  auto choice = [&](const std::vector<int> &v) -> int {
    std::uniform_int_distribution<int> pick(0, static_cast<int>(v.size()) - 1);
    return v[pick(rng)];
  };

  auto log_proxy = [&](const std::string &label) {
    if (!log_progress)
      return;
    double hpwl = std::accumulate(cur_hpwl.begin(), cur_hpwl.end(), 0.0);
    double wl = hpwl / wl_norm;
    double proxy = proxy_overall(wl, den_eval.current(), cong_eval.current());
    std::cerr << "[vibeCpp] " << label << " proxy=" << proxy << " WL=" << wl
              << " density=" << den_eval.current()
              << " congestion=" << cong_eval.current() << "\n";
  };

  for (int iter = 0; iter < max_iters; ++iter) {
    if (log_progress && iter % 5000 == 0) {
      log_proxy("SA iter=" + std::to_string(iter));
    }

    int idx;
    double sigma;
    if (!movable_hard.empty() &&
        (movable_soft.empty() || uni(rng) < hard_move_prob)) {
      idx = choice(movable_hard);
      sigma = hard_sigma;
    } else {
      idx = choice(movable_soft);
      sigma = soft_sigma;
    }

    double old_x = pos[idx].x;
    double old_y = pos[idx].y;
    double hw = b.macros[idx].w * 0.5;
    double hh = b.macros[idx].h * 0.5;
    double lo_x = hw + 1e-6;
    double hi_x = b.cw - hw - 1e-6;
    double lo_y = hh + 1e-6;
    double hi_y = b.ch - hh - 1e-6;
    if (bounds.enabled) {
      lo_x = std::max(lo_x, bounds.x0 + hw + 1e-6);
      hi_x = std::min(hi_x, bounds.x1 - hw - 1e-6);
      lo_y = std::max(lo_y, bounds.y0 + hh + 1e-6);
      hi_y = std::min(hi_y, bounds.y1 - hh - 1e-6);
    }
    if (lo_x > hi_x || lo_y > hi_y) {
      continue;
    }
    double step_x = normal(rng) * sigma;
    double step_y = normal(rng) * sigma;
    bool accepted = false;
    bool cost_rejected = false;
    bool initial_overlap_rejected = false;

    for (double move_scale : move_scales) {
      if (gate_default_rescue_scales && move_scale != 1.0 && !cost_rejected) {
        continue;
      }
      double nx = clamp(old_x + step_x * move_scale, lo_x, hi_x);
      double ny = clamp(old_y + step_y * move_scale, lo_y, hi_y);
      pos[idx] = {nx, ny};

      if (idx < b.n_hard) {
        bool bad = false;
        for (int j = 0; j < b.n_hard; ++j) {
          if (j == idx)
            continue;
          double sep_x = (b.macros[idx].w + b.macros[j].w) * 0.5 + 1e-6;
          double sep_y = (b.macros[idx].h + b.macros[j].h) * 0.5 + 1e-6;
          if (std::abs(nx - pos[j].x) < sep_x &&
              std::abs(ny - pos[j].y) < sep_y) {
            bad = true;
            break;
          }
        }
        if (bad) {
          pos[idx] = {old_x, old_y};
          if (gate_default_rescue_scales && move_scale == 1.0) {
            initial_overlap_rejected = true;
            break;
          }
          continue;
        }
      }

      auto den_pair = den_eval.begin_single_update(idx, nx, ny);
      double new_den = den_pair.first;
      auto den_token = den_pair.second;

      double old_wl = 0.0;
      double new_wl = 0.0;
      std::vector<std::pair<int, double>> updates;
      for (int net_i : b.macro_to_nets[idx]) {
        double w = net_i < static_cast<int>(b.net_weights.size())
                       ? b.net_weights[net_i]
                       : 1.0;
        double hpwl = net_hpwl(b, pos, net_i) * w;
        old_wl += cur_hpwl[net_i];
        new_wl += hpwl;
        updates.push_back({net_i, hpwl});
      }
      auto cong_pair = cong_eval.begin_single_update(idx, b.macro_to_nets[idx]);
      double new_cong = cong_pair.first;
      auto cong_token = cong_pair.second;

      double delta = new_wl - old_wl;
      delta += (congestion_weight * (new_cong - cong_token.old_cost) +
                density_weight * (new_den - den_token.old_cost)) *
               wl_norm;

      if (delta < 0.0 || uni(rng) < std::exp(-delta / std::max(temp, EPS))) {
        cur_cost += delta;
        for (const auto &item : updates) {
          cur_hpwl[item.first] = item.second;
        }
        den_eval.accept_single(den_token, idx, nx, ny);
        cong_eval.accept(cong_token);
        accepted = true;
        if (cur_cost < best_cost - 1e-12) {
          best_cost = cur_cost;
          best_pos = pos;
        }
        break;
      }

      den_eval.reject(den_token);
      cong_eval.reject(cong_token);
      pos[idx] = {old_x, old_y};
      cost_rejected = true;
    }

    if (initial_overlap_rejected) {
      continue;
    }
    if (!accepted) {
      pos[idx] = {old_x, old_y};
    }

    recent_costs.push_back(cur_cost);
    if (recent_costs.size() > 1000) {
      recent_costs.pop_front();
    }
    if (iter % 1000 == 0 && recent_costs.size() == 1000) {
      double prev = 0.0;
      double curr = 0.0;
      for (int i = 0; i < 500; ++i)
        prev += recent_costs[i];
      for (int i = 500; i < 1000; ++i)
        curr += recent_costs[i];
      prev /= 500.0;
      curr /= 500.0;
      double grad_abs = std::abs(curr - prev);
      if (grad_abs < grad_stop_threshold) {
        ++consec_below;
      } else {
        consec_below = 0;
      }
      double frac = 0.0;
      if (std::isfinite(grad_abs)) {
        if (grad_abs >= grad_ref) {
          frac = 0.0;
        } else if (grad_abs <= grad_stop_threshold) {
          frac = 1.0;
        } else {
          frac = std::log(grad_ref / grad_abs) /
                 std::log(grad_ref / grad_stop_threshold);
          frac = clamp(frac, 0.0, 1.0);
        }
        temp = temp_start * std::pow(temp_ratio, frac);
        hard_sigma = canvas * (hard_sigma_start * (1.0 - frac) + 0.001 * frac);
        soft_sigma = canvas * (soft_sigma_start * (1.0 - frac) + 0.002 * frac);
      }
      if (consec_below >= 10) {
        break;
      }
    }
  }
  log_proxy("SA final");
  return best_pos;
}

std::vector<Point> parallel_sa_all_macro(
    const BenchmarkData &b, const std::vector<Point> &start_pos,
    const std::vector<bool> &fixed_override, int seed, double congestion_weight,
    double density_weight, double hard_move_prob, double hard_sigma_start,
    double soft_sigma_start, int max_iters, Bounds bounds, bool log_progress,
    const std::string &label, const std::vector<bool> *allowed_mask = nullptr,
    const std::vector<double> *move_scales_override = nullptr) {
  const int threads = std::min(vibe_thread_count(), std::max(1, max_iters));
  if (threads <= 1 || max_iters < 2000) {
    return sa_all_macro_incremental(
        b, start_pos, fixed_override, seed, congestion_weight, density_weight,
        hard_move_prob, hard_sigma_start, soft_sigma_start, max_iters, bounds,
        log_progress, allowed_mask, move_scales_override);
  }

  if (log_progress) {
    std::cerr << "[vibeCpp] " << label << " parallel SA workers=" << threads
              << " total_iters=" << max_iters << "\n";
  }

  std::vector<std::future<std::vector<Point>>> futures;
  futures.reserve(threads);
  int remaining = max_iters;
  for (int worker = 0; worker < threads; ++worker) {
    int worker_count = threads - worker;
    int worker_iters = (remaining + worker_count - 1) / worker_count;
    remaining -= worker_iters;
    int worker_seed = seed + worker * 9973;
    futures.push_back(std::async(std::launch::async, [&, worker_iters,
                                                      worker_seed]() {
      return sa_all_macro_incremental(
          b, start_pos, fixed_override, worker_seed, congestion_weight,
          density_weight, hard_move_prob, hard_sigma_start, soft_sigma_start,
          worker_iters, bounds, false, allowed_mask, move_scales_override);
    }));
  }

  std::vector<Point> best = start_pos;
  ProxyComponents best_proxy = compute_proxy(b, best);
  for (auto &fut : futures) {
    std::vector<Point> candidate = fut.get();
    if (has_output_hard_overlaps(b, candidate)) {
      continue;
    }
    ProxyComponents after = compute_proxy(b, candidate);
    if (after.proxy < best_proxy.proxy) {
      best = std::move(candidate);
      best_proxy = after;
    }
  }

  if (log_progress) {
    std::cerr << "[vibeCpp] " << label
              << " parallel SA final proxy=" << best_proxy.proxy
              << " WL=" << best_proxy.wirelength
              << " density=" << best_proxy.density
              << " congestion=" << best_proxy.congestion << "\n";
  }
  return best;
}

bool macro_center_inside_canvas(const BenchmarkData &b, int idx, Point p) {
  double hw = b.macros[idx].w * 0.5;
  double hh = b.macros[idx].h * 0.5;
  return p.x >= hw + 1e-6 && p.x <= b.cw - hw - 1e-6 && p.y >= hh + 1e-6 &&
         p.y <= b.ch - hh - 1e-6;
}

bool hard_pair_overlaps_at(const BenchmarkData &b, int a, Point pa, int c,
                           Point pc) {
  double sep_x = (b.macros[a].w + b.macros[c].w) * 0.5;
  double sep_y = (b.macros[a].h + b.macros[c].h) * 0.5;
  if (std::abs(pa.x - pc.x) < sep_x && std::abs(pa.y - pc.y) < sep_y) {
    return true;
  }
  float dx = std::abs(static_cast<float>(pa.x) - static_cast<float>(pc.x));
  float dy = std::abs(static_cast<float>(pa.y) - static_cast<float>(pc.y));
  float fsep_x =
      (static_cast<float>(b.macros[a].w) + static_cast<float>(b.macros[c].w)) *
      0.5f;
  float fsep_y =
      (static_cast<float>(b.macros[a].h) + static_cast<float>(b.macros[c].h)) *
      0.5f;
  return dx < fsep_x && dy < fsep_y;
}

bool swap_has_hard_overlap(const BenchmarkData &b,
                           const std::vector<Point> &pos, int a, int c) {
  Point pa = pos[c];
  Point pc = pos[a];
  if (!macro_center_inside_canvas(b, a, pa) ||
      !macro_center_inside_canvas(b, c, pc)) {
    return true;
  }
  if (a >= b.n_hard && c >= b.n_hard) {
    return false;
  }

  auto swapped_pos = [&](int idx) -> Point {
    if (idx == a)
      return pa;
    if (idx == c)
      return pc;
    return pos[idx];
  };
  std::vector<int> moved_hard;
  if (a < b.n_hard)
    moved_hard.push_back(a);
  if (c < b.n_hard)
    moved_hard.push_back(c);
  for (int idx : moved_hard) {
    Point pi = swapped_pos(idx);
    for (int j = 0; j < b.n_hard; ++j) {
      if (j == idx) {
        continue;
      }
      if (hard_pair_overlaps_at(b, idx, pi, j, swapped_pos(j))) {
        return true;
      }
    }
  }
  return false;
}

std::vector<int> affected_nets_for_pair(const BenchmarkData &b, int a, int c) {
  std::vector<int> affected = b.macro_to_nets[a];
  affected.insert(affected.end(), b.macro_to_nets[c].begin(),
                  b.macro_to_nets[c].end());
  std::sort(affected.begin(), affected.end());
  affected.erase(std::unique(affected.begin(), affected.end()), affected.end());
  return affected;
}

std::vector<Point> swap_sa_polish(const BenchmarkData &b,
                                  std::vector<Point> pos, int seed,
                                  int max_iters) {
  if (max_iters <= 0) {
    return pos;
  }

  std::vector<int> by_area;
  by_area.reserve(b.n_total);
  for (int i = 0; i < b.n_total; ++i) {
    if (!b.macros[i].fixed) {
      by_area.push_back(i);
    }
  }
  std::sort(by_area.begin(), by_area.end(), [&](int a, int c) {
    double area_a = b.macros[a].w * b.macros[a].h;
    double area_c = b.macros[c].w * b.macros[c].h;
    if (area_a != area_c) {
      return area_a < area_c;
    }
    return a < c;
  });
  if (by_area.size() < 2) {
    std::cerr << "[vibeCpp] swap SA skipped movable_macros=" << by_area.size()
              << "\n";
    return pos;
  }

  std::mt19937_64 rng(static_cast<uint64_t>(seed));
  std::vector<double> cur_hpwl(b.n_nets, 0.0);
  for (int ni = 0; ni < b.n_nets; ++ni) {
    double w =
        ni < static_cast<int>(b.net_weights.size()) ? b.net_weights[ni] : 1.0;
    cur_hpwl[ni] = net_hpwl(b, pos, ni) * w;
  }
  double cur_total_wl = std::accumulate(cur_hpwl.begin(), cur_hpwl.end(), 0.0);
  double norm_net_count = b.wl_net_count > 0.0
                              ? b.wl_net_count
                              : static_cast<double>(std::max(1, b.n_nets));
  double wl_norm = std::max((b.cw + b.ch) * norm_net_count, EPS);
  IncrementalDensityCost den_eval(b, pos);
  IncrementalCongestionCost cong_eval(b, pos);
  ProxyComponents current;
  current.wirelength = cur_total_wl / wl_norm;
  current.density = den_eval.current();
  current.congestion = cong_eval.current();
  current.proxy =
      proxy_overall(current.wirelength, current.density, current.congestion);

  int group_size = std::min(6, static_cast<int>(by_area.size()));
  int window_count =
      std::max(1, static_cast<int>(by_area.size()) - group_size + 1);
  std::uniform_int_distribution<int> window_pick(0, window_count - 1);
  int accepted = 0;
  int valid_rounds = 0;
  int completed_iters = 0;

  std::cerr << "[vibeCpp] swap SA movable_macros=" << by_area.size()
            << " group_size=" << group_size << " iters=" << max_iters
            << "\n";

  for (int iter = 1; iter <= max_iters; ++iter) {
    completed_iters = iter;
    int start = window_pick(rng);
    ProxyComponents best_proxy;
    best_proxy.proxy = std::numeric_limits<double>::infinity();
    int best_a = -1;
    int best_c = -1;
    double best_total_wl = cur_total_wl;
    std::vector<std::pair<int, double>> best_hpwl_updates;

    for (int u = 0; u < group_size; ++u) {
      for (int v = u + 1; v < group_size; ++v) {
        int a = by_area[start + u];
        int c = by_area[start + v];
        if (swap_has_hard_overlap(b, pos, a, c)) {
          continue;
        }

        Point old_a = pos[a];
        Point old_c = pos[c];
        std::swap(pos[a], pos[c]);
        std::vector<int> affected = affected_nets_for_pair(b, a, c);
        double old_wl = 0.0;
        double new_wl = 0.0;
        std::vector<std::pair<int, double>> hpwl_updates;
        hpwl_updates.reserve(affected.size());
        for (int net_i : affected) {
          double w = net_i < static_cast<int>(b.net_weights.size())
                         ? b.net_weights[net_i]
                         : 1.0;
          double hpwl = net_hpwl(b, pos, net_i) * w;
          old_wl += cur_hpwl[net_i];
          new_wl += hpwl;
          hpwl_updates.push_back({net_i, hpwl});
        }

        auto den_pair = den_eval.begin_two_update(a, pos[a].x, pos[a].y, c,
                                                  pos[c].x, pos[c].y);
        auto den_token = den_pair.second;
        auto cong_pair = cong_eval.begin_multi_update({a, c}, affected);
        auto cong_token = cong_pair.second;

        double candidate_total_wl = cur_total_wl + new_wl - old_wl;
        ProxyComponents candidate;
        candidate.wirelength = candidate_total_wl / wl_norm;
        candidate.density = den_pair.first;
        candidate.congestion = cong_pair.first;
        candidate.proxy = proxy_overall(candidate.wirelength, candidate.density,
                                        candidate.congestion);

        cong_eval.reject(cong_token);
        den_eval.reject(den_token);
        pos[a] = old_a;
        pos[c] = old_c;

        if (candidate.proxy < best_proxy.proxy) {
          best_proxy = candidate;
          best_a = a;
          best_c = c;
          best_total_wl = candidate_total_wl;
          best_hpwl_updates = std::move(hpwl_updates);
        }
      }
    }

    if (best_a >= 0) {
      ++valid_rounds;
    }
    if (best_a >= 0 && best_proxy.proxy < current.proxy - SWAP_SA_ACCEPT_TOL) {
      std::swap(pos[best_a], pos[best_c]);
      auto den_pair =
          den_eval.begin_two_update(best_a, pos[best_a].x, pos[best_a].y,
                                    best_c, pos[best_c].x, pos[best_c].y);
      den_eval.accept_two(den_pair.second, best_a, pos[best_a].x, pos[best_a].y,
                          best_c, pos[best_c].x, pos[best_c].y);
      std::vector<int> affected = affected_nets_for_pair(b, best_a, best_c);
      auto cong_pair = cong_eval.begin_multi_update({best_a, best_c}, affected);
      cong_eval.accept(cong_pair.second);
      for (const auto &item : best_hpwl_updates) {
        cur_hpwl[item.first] = item.second;
      }
      cur_total_wl = best_total_wl;
      current = best_proxy;
      ++accepted;
    }

    if (iter % SWAP_SA_LOG_INTERVAL == 0 || iter == max_iters) {
      double success_rate =
          static_cast<double>(accepted) / static_cast<double>(iter);
      double valid_rate =
          static_cast<double>(valid_rounds) / static_cast<double>(iter);
      std::cerr << "[vibeCpp] swap SA iter=" << iter
                << " success_rate=" << success_rate
                << " valid_rate=" << valid_rate << " proxy=" << current.proxy
                << " WL=" << current.wirelength
                << " density=" << current.density
                << " congestion=" << current.congestion << "\n";
      if (iter >= SWAP_SA_LOG_INTERVAL &&
          success_rate < SWAP_SA_MIN_SUCCESS_RATE) {
        std::cerr << "[vibeCpp] swap SA early stop success_rate="
                  << success_rate << " threshold=" << SWAP_SA_MIN_SUCCESS_RATE
                  << " iter=" << iter << "\n";
        break;
      }
    }
  }

  int denom = std::max(1, completed_iters);
  std::cerr << "[vibeCpp] swap SA final proxy=" << current.proxy
            << " WL=" << current.wirelength << " density=" << current.density
            << " congestion=" << current.congestion << " success_rate="
            << static_cast<double>(accepted) / static_cast<double>(denom)
            << "\n";
  return pos;
}

std::vector<bool> hot_congestion_macro_mask(const BenchmarkData &b,
                                            std::vector<Point> &pos,
                                            int &hot_net_count, int &hard_count,
                                            int &soft_count) {
  IncrementalCongestionCost congestion(b, pos);
  std::vector<int> hot_nets = congestion.hot_congestion_nets();
  hot_net_count = static_cast<int>(hot_nets.size());

  std::vector<bool> selected(b.n_total, false);
  for (int net_i : hot_nets) {
    if (net_i < 0 || net_i >= static_cast<int>(b.nets_pin_indices.size())) {
      continue;
    }
    for (int pin_idx : b.nets_pin_indices[net_i]) {
      if (pin_idx < 0 || pin_idx >= static_cast<int>(b.pins.size())) {
        continue;
      }
      int owner = b.pins[pin_idx].owner;
      if (0 <= owner && owner < b.n_total && !b.macros[owner].fixed) {
        selected[owner] = true;
      }
    }
  }

  hard_count = 0;
  soft_count = 0;
  for (int i = 0; i < b.n_total; ++i) {
    if (!selected[i]) {
      continue;
    }
    if (i < b.n_hard) {
      ++hard_count;
    } else {
      ++soft_count;
    }
  }
  return selected;
}

std::vector<Point> hot_congestion_sa_polish(const BenchmarkData &b,
                                            const std::vector<Point> &positions,
                                            const std::vector<Point> &original,
                                            const std::vector<bool> &large_mask,
                                            int seed, int max_iters) {
  if (max_iters <= 0) {
    return positions;
  }

  std::vector<Point> start = positions;
  int hot_net_count = 0;
  int hot_hard_count = 0;
  int hot_soft_count = 0;
  std::vector<bool> hot_macros = hot_congestion_macro_mask(
      b, start, hot_net_count, hot_hard_count, hot_soft_count);
  int hot_macro_count = hot_hard_count + hot_soft_count;
  if (hot_macro_count == 0) {
    std::cerr << "[vibeCpp] hot congestion SA skipped hot_nets="
              << hot_net_count << " hot_macros=0\n";
    return positions;
  }

  std::vector<Point> current_pos = positions;
  ProxyComponents current = compute_proxy(b, current_pos);
  static const std::vector<double> hot_move_scales{0.4, 0.7, 1.0,
                                                   1.3, 1.7, 2.0};
  std::cerr << "[vibeCpp] hot congestion SA hot_nets=" << hot_net_count
            << " hot_macros=" << hot_macro_count << " hard=" << hot_hard_count
            << " soft=" << hot_soft_count << " iters=" << max_iters
            << " move_scales=0.4,0.7,1.0,1.3,1.7,2.0\n";

  std::vector<Point> candidate = parallel_sa_all_macro(
      b, positions, {}, seed, HOT_CONGESTION_SA_CONG_WEIGHT,
      HOT_CONGESTION_SA_DENSITY_WEIGHT, 0.55, 0.015, 0.045, max_iters, Bounds{},
      true, "hot_congestion_sa", &hot_macros, &hot_move_scales);
  clamp_positions(b, candidate, original);
  if (has_output_hard_overlaps(b, candidate)) {
    legalize_large_then_small(b, candidate, large_mask, GAP);
    clamp_positions(b, candidate, original);
  }

  bool overlaps = has_output_hard_overlaps(b, candidate);
  ProxyComponents after = compute_proxy(b, candidate);
  bool accept = !overlaps &&
                after.proxy <= current.proxy + HOT_CONGESTION_ACCEPT_TOL &&
                after.congestion <= current.congestion + EPS;
  std::cerr << "[vibeCpp] hot congestion SA "
            << (accept ? "accepted" : "rejected") << " proxy " << current.proxy
            << " -> " << after.proxy << " congestion " << current.congestion
            << " -> " << after.congestion << " overlaps=" << (overlaps ? 1 : 0)
            << "\n";
  if (accept) {
    return candidate;
  }
  return positions;
}

std::vector<Point> place(const BenchmarkData &b, int seed) {
  std::vector<Point> original = macro_positions(b);
  std::vector<Point> positions = original;
  clamp_positions(b, positions, original);

  std::vector<bool> large_mask(b.n_total, false);
  std::vector<int> large_indices;
  double threshold = LARGE_MACRO_AREA_FRAC * b.cw * b.ch;
  for (int i = 0; i < b.n_hard; ++i) {
    if (!b.macros[i].fixed && b.macros[i].w * b.macros[i].h > threshold) {
      large_mask[i] = true;
      large_indices.push_back(i);
    }
  }

  std::cerr << "[vibeCpp] start " << b.name << " macros=" << b.n_total
            << " hard=" << b.n_hard << " nets=" << b.n_nets << "\n";
  ProxyComponents start = compute_proxy(b, positions);
  std::cerr << "[vibeCpp] initial proxy=" << start.proxy
            << " WL=" << start.wirelength << " density=" << start.density
            << " congestion=" << start.congestion << "\n";

  if (has_output_hard_overlaps(b, positions)) {
    legalize_to_boundary(b, positions, large_indices, GAP);
    legalize_large_then_small(b, positions, large_mask, GAP);
    clamp_positions(b, positions, original);
  }

  int short_iters =
      env_int("VIBECPP_SA_SHORT_ITERS", ALL_MACRO_SA_SHORT_ITERS_DEFAULT);
  int full_iters =
      env_int("VIBECPP_SA_FULL_ITERS", ALL_MACRO_SA_MAX_ITERS_DEFAULT);
  int cong_iters = env_int("VIBECPP_CONGESTION_POLISH_ITERS",
                           CONGESTION_POLISH_ITERS_DEFAULT);
  int hot_cong_iters = env_int("VIBECPP_HOT_CONGESTION_SA_ITERS",
                               HOT_CONGESTION_SA_ITERS_DEFAULT);
  int swap_iters = env_int("VIBECPP_SWAP_SA_ITERS", SWAP_SA_ITERS_DEFAULT);
  int final_fix_iters =
      env_int("VIBECPP_FINAL_FIX_SA_ITERS", ALL_MACRO_SA_MAX_ITERS_DEFAULT);

  positions = parallel_sa_all_macro(
      b, positions, {}, seed + 505, ALL_MACRO_SA_CONG_WEIGHT,
      ALL_MACRO_SA_DENSITY_WEIGHT, 0.35, 0.03, 0.08, short_iters, Bounds{},
      true, "short_all_macro");
  clamp_positions(b, positions, original);
  legalize_large_then_small(b, positions, large_mask, GAP);

  positions = parallel_sa_all_macro(
      b, positions, {}, seed + 1505, ALL_MACRO_SA_CONG_WEIGHT,
      ALL_MACRO_SA_DENSITY_WEIGHT, 0.35, 0.03, 0.08, full_iters, Bounds{}, true,
      "full_all_macro");
  clamp_positions(b, positions, original);

  ProxyComponents current = compute_proxy(b, positions);
  if (cong_iters > 0 &&
      current.congestion >= CONGESTION_POLISH_MIN_CONGESTION) {
    for (int pass = 0; pass < 2; ++pass) {
      double cong_weight = pass == 0 ? 0.9 : 1.2;
      std::vector<Point> candidate = parallel_sa_all_macro(
          b, positions, {}, seed + 2505 + pass * 101, cong_weight,
          PROXY_DENSITY_WEIGHT, 0.45, 0.015, 0.045, cong_iters, Bounds{}, false,
          "congestion_polish");
      clamp_positions(b, candidate, original);
      if (has_output_hard_overlaps(b, candidate)) {
        legalize_large_then_small(b, candidate, large_mask, GAP);
        clamp_positions(b, candidate, original);
      }
      ProxyComponents after = compute_proxy(b, candidate);
      bool accept =
          after.proxy <= current.proxy + CONGESTION_POLISH_ACCEPT_TOL &&
          after.congestion <= current.congestion + EPS;
      std::cerr << "[vibeCpp] cong pass=" << pass + 1
                << (accept ? " accepted" : " rejected") << " proxy "
                << current.proxy << " -> " << after.proxy << " congestion "
                << current.congestion << " -> " << after.congestion << "\n";
      if (accept) {
        positions = std::move(candidate);
        current = after;
      }
    }
  }

  positions = hot_congestion_sa_polish(b, positions, original, large_mask,
                                       seed + 3505, hot_cong_iters);

  positions = swap_sa_polish(b, positions, seed + 4005, swap_iters);
  clamp_positions(b, positions, original);

  positions = parallel_sa_all_macro(
      b, positions, {}, seed + 4505, ALL_MACRO_SA_CONG_WEIGHT,
      ALL_MACRO_SA_DENSITY_WEIGHT, 0.35, 0.03, 0.08, final_fix_iters, Bounds{},
      true, "full_all_macro");
  clamp_positions(b, positions, original);

  for (int attempt = 0; attempt < 3 && has_output_hard_overlaps(b, positions);
       ++attempt) {
    legalize_large_then_small(b, positions, large_mask, GAP);
    clamp_positions(b, positions, original);
  }
  clamp_positions(b, positions, original);
  ProxyComponents final = compute_proxy(b, positions);
  std::cerr << "[vibeCpp] final proxy=" << final.proxy
            << " WL=" << final.wirelength << " density=" << final.density
            << " congestion=" << final.congestion
            << " overlaps=" << (has_output_hard_overlaps(b, positions) ? 1 : 0)
            << "\n";
  return positions;
}

void write_output(const std::string &path, const std::vector<Point> &pos) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open output: " + path);
  }
  out << std::setprecision(17);
  for (const auto &p : pos) {
    out << p.x << ' ' << p.y << '\n';
  }
}

} // namespace

int main(int argc, char **argv) {
  std::string input;
  std::string output;
  int seed = 7;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      input = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output = argv[++i];
    } else if (arg == "--seed" && i + 1 < argc) {
      seed = std::stoi(argv[++i]);
    } else {
      std::cerr << "usage: vibe_placer --input benchmark.txt --output "
                   "placement.txt [--seed N]\n";
      return 2;
    }
  }
  if (input.empty() || output.empty()) {
    std::cerr << "usage: vibe_placer --input benchmark.txt --output "
                 "placement.txt [--seed N]\n";
    return 2;
  }
  try {
    BenchmarkData b = read_benchmark(input);
    std::vector<Point> positions = place(b, seed);
    write_output(output, positions);
  } catch (const std::exception &exc) {
    std::cerr << "vibe_placer error: " << exc.what() << "\n";
    return 1;
  }
  return 0;
}
