#pragma once
#include <cmath>
#include <algorithm>

namespace wasrt {

// Python's round() is round-half-to-even; the default FP rounding mode makes nearbyint match it.
inline long round_half_even(double x) { return static_cast<long>(std::nearbyint(x)); }

// Python's % on ints follows the sign of the divisor (floored modulo); C++ % follows the dividend.
inline long floored_mod(long a, long b) {
	long r = a % b;
	if (r != 0 && ((r < 0) != (b < 0))) r += b;
	return r;
}

struct Geometry {
	double sx, sy;
	long x_offset, y_offset, out_w, out_h, h_after;
};

inline Geometry compute_geometry(long orig_w, long orig_h, double scale, double top_pct, double bottom_pct, long divisor) {
	long scaled_w = std::max(divisor, round_half_even(orig_w * scale));
	long scaled_h = std::max(divisor, round_half_even(orig_h * scale));
	double sx = static_cast<double>(scaled_w) / static_cast<double>(orig_w);
	double sy = static_cast<double>(scaled_h) / static_cast<double>(orig_h);
	long top_cut = round_half_even(scaled_h * top_pct);
	long bottom_cut = round_half_even(scaled_h * bottom_pct);
	long h_after = scaled_h - top_cut - bottom_cut;

	long extra_w = floored_mod(scaled_w, divisor);
	long left_extra = extra_w / 2;
	long out_w = scaled_w - extra_w;
	long extra_h = floored_mod(h_after, divisor);
	long top_extra = extra_h / 2;
	long out_h = h_after - extra_h;

	Geometry g;
	g.sx = sx;
	g.sy = sy;
	g.x_offset = left_extra;
	g.y_offset = top_cut + top_extra;
	g.out_w = out_w;
	g.out_h = out_h;
	g.h_after = h_after;
	return g;
}

// Rectified stream: R stays identity, D stays as-is; only K and P are rescaled by the crop geometry.
inline void scale_intrinsics(const double* K_in, const double* P_in, const Geometry& g, double* K_out, double* P_out) {
	for (int i = 0; i < 9; ++i) K_out[i] = K_in[i];
	K_out[0] = K_in[0] * g.sx;
	K_out[2] = K_in[2] * g.sx - g.x_offset;
	K_out[4] = K_in[4] * g.sy;
	K_out[5] = K_in[5] * g.sy - g.y_offset;

	for (int i = 0; i < 12; ++i) P_out[i] = P_in[i];
	P_out[0] = P_in[0] * g.sx;
	P_out[2] = P_in[2] * g.sx - g.x_offset;
	P_out[3] = P_in[3] * g.sx;
	P_out[5] = P_in[5] * g.sy;
	P_out[6] = P_in[6] * g.sy - g.y_offset;
}

}
