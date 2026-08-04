// Host-only unit tests for the adaptive filter's error model / selection logic.
// These construct SpectralSketch objects by hand (no GPU needed) and check that
// select_filter picks cheap filters for easy spectra and accurate ones for hard
// spectra.  GPU end-to-end validation lives in the benchmark/ development tree
// (bench_md_adap.cu), which is not part of this repository.
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "psd_projection/adaptive_filter.h"
#include "psd_projection/spectral_sketch.h"
#include "psd_projection/filter_bank.h"


// Build a synthetic sketch from an explicit list of (scaled) eigenvalues,
// giving each an equal quadrature weight, as if n_probes == 1.
static SpectralSketch make_sketch(const std::vector<double>& y, double scale = 1.0) {
    SpectralSketch S;
    S.n = (int)y.size();
    S.n_probes = 1;
    S.scale = scale;
    double lo = 1e300, hi = -1e300;
    for (double v : y) {
        S.theta.push_back(v);
        S.tau.push_back(1.0 / (double)y.size());   // weights sum to 1 per "probe"
        lo = std::min(lo, v); hi = std::max(hi, v);
    }
    S.lam_min_scaled = lo; S.lam_max_scaled = hi;
    return S;
}

TEST(SelectFilter, GappedSpectrumPicksCheapFilter) {
    // eigenvalues in [-1,-0.3] U [0.3,1] -> large gap, few stages should suffice
    std::vector<double> y;
    for (int i = 0; i < 500; ++i) {
        double t = (double)i / 499.0;
        y.push_back(-1.0 + t * 0.7);        // [-1,-0.3]
        y.push_back(0.3 + t * 0.7);         // [0.3,1]
    }
    SpectralSketch S = make_sketch(y);
    AdaptiveOptions opts; opts.tol = 1e-3; opts.enable_deflation = false;
    double pred; bool qual;
    int idx = select_filter(S, opts, &pred, &qual);
    ASSERT_GE(idx, 0);
    EXPECT_TRUE(qual);
    EXPECT_LE(FILTER_BANK[idx].gemms, 16);   // far below the fixed 31
    EXPECT_LE(opts.safety * pred, opts.tol);
}

TEST(SelectFilter, ClusterAtZeroNeedsMoreStages) {
    // eigenvalues packed toward 0 -> small effective gap, needs a longer filter
    std::vector<double> y;
    for (int i = 0; i < 1000; ++i) {
        double mag = std::pow(10.0, -3.0 * (i % 500) / 500.0);
        y.push_back((i < 500) ? -mag : mag);
    }
    SpectralSketch S = make_sketch(y);
    AdaptiveOptions opts; opts.tol = 1e-3; opts.enable_deflation = false;
    double pred_hard; bool qual;
    int idx_hard = select_filter(S, opts, &pred_hard, &qual);
    ASSERT_GE(idx_hard, 0);

    // gapped spectrum should be selectable with no more GEMMs than the clustered one
    std::vector<double> yg;
    for (int i = 0; i < 1000; ++i) {
        double t = (double)i / 999.0;
        yg.push_back((i < 500) ? (-1.0 + t * 0.7) : (0.3 + t * 0.7));
    }
    SpectralSketch Sg = make_sketch(yg);
    double pred_easy; bool qe;
    int idx_easy = select_filter(Sg, opts, &pred_easy, &qe);
    EXPECT_LE(FILTER_BANK[idx_easy].gemms, FILTER_BANK[idx_hard].gemms);
}

TEST(SelectFilter, TighterToleranceCostsMoreOrEqualGemms) {
    std::vector<double> y;
    for (int i = 0; i < 800; ++i) {
        double t = (double)i / 799.0;
        y.push_back(-1.0 + 2.0 * t);        // uniform [-1,1]
    }
    SpectralSketch S = make_sketch(y);
    AdaptiveOptions loose; loose.tol = 1e-2; loose.enable_deflation = false;
    AdaptiveOptions tight; tight.tol = 1e-5; tight.enable_deflation = false;
    double p; bool q;
    int i_loose = select_filter(S, loose, &p, &q);
    int i_tight = select_filter(S, tight, &p, &q);
    EXPECT_GE(FILTER_BANK[i_tight].gemms, FILTER_BANK[i_loose].gemms);
}

TEST(FilterBank, NonEmptyAndMonotoneMetadata) {
    ASSERT_GT(FILTER_BANK_SIZE, 0);
    for (int i = 0; i < FILTER_BANK_SIZE; ++i) {
        EXPECT_EQ(FILTER_BANK[i].gemms, 3 * FILTER_BANK[i].T + 1);
        EXPECT_GT(FILTER_BANK[i].e_sign, 0.0);
        EXPECT_LE(FILTER_BANK[i].e_relu, FILTER_BANK[i].e_sign);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
