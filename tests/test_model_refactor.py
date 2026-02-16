import importlib
import warnings
import unittest

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import model.core as core
from model.core import STDPModel, STDPParams
from model.data import load_data
from model.fit import FitConfig, fit_stdp
from model.objective import _hill_lambda_constraint_penalty, _pack_params
from model.plotting_basic import evaluate_model, plot_fraction_comparison
from model.uncertainty import compute_prediction_ci


def make_dataset():
    params = STDPParams(
        k_lag=3,
        mu0=0.5,
        mu24p=0.1,
        kT=0.6,
        kS_kT_ratio=4.0,
        kST=1.2,
        n=1.5,
        nST=1.2,
        a50=20.0,
        r0=0.02,
    )
    model = STDPModel(params)
    rows = []
    for idx, (C, tau, age) in enumerate([(0.1, 1.0, 24.0), (1.0, 2.0, 24.0), (10.0, 4.0, 48.0)]):
        pred = model.predict_condition(C, tau, age)
        rows.append(
            (
                f"cond_{idx}",
                C,
                tau,
                age,
                {
                    "incomplete": pred.incomplete,
                    "induced": pred.induced,
                    "preexisting": pred.preexisting,
                },
            )
        )
    return rows


class ModelRefactorTests(unittest.TestCase):
    def test_prediction_probabilities_sum(self):
        model = STDPModel(STDPParams())
        pred = model.predict_condition(5.0, 2.0, 24.0)
        total = pred.incomplete + pred.induced + pred.preexisting + pred.dead
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_non_log_hazard_prediction_is_finite_at_high_dose(self):
        original = core.USE_LOG_HAZARDS
        core.USE_LOG_HAZARDS = False
        try:
            model = STDPModel(STDPParams())
            pred = model.predict_condition(1000.0, 10.0, 24.0)
            values = np.array([pred.incomplete, pred.induced, pred.preexisting, pred.dead], dtype=float)
            self.assertTrue(np.isfinite(values).all())
            self.assertGreaterEqual(values.min(), 0.0)
            self.assertLessEqual(values.max(), 1.0)
        finally:
            core.USE_LOG_HAZARDS = original

    def test_independent_log_hazard_overrides(self):
        old_flags = core.get_log_hazard_flags()
        try:
            C = 10.0
            a = 24.0
            p = STDPParams(kT=0.5, kS_kT_ratio=5.0, kST=1.35, n=2.0, nST=1.5, a50=20.0, r0=0.01)

            hs_linear = p.kT * p.kS_kT_ratio * (C**p.n)
            ht_linear = p.kT * (C**p.n)
            rst_linear = core.m(a, p.a50) * (p.r0 + p.kST * (C**p.nST))

            core.set_log_hazard_flags(global_default=False, hS=True, hT=False, rST=False)
            self.assertAlmostEqual(core.hS(C, p.kT, p.kS_kT_ratio, p.n), np.log1p(hs_linear), places=12)
            self.assertAlmostEqual(core.hT(C, p.kT, p.n), ht_linear, places=12)
            self.assertAlmostEqual(
                core.rST(C, a, p.kST, p.nST, p.a50, p.r0),
                rst_linear,
                places=12,
            )

            core.set_log_hazard_flags(hS=None, hT=None, rST=None)
            core.set_log_hazard_flags(global_default=True)
            self.assertAlmostEqual(core.hS(C, p.kT, p.kS_kT_ratio, p.n), np.log1p(hs_linear), places=12)
            self.assertAlmostEqual(core.hT(C, p.kT, p.n), np.log1p(ht_linear), places=12)
        finally:
            core.USE_LOG_HAZARDS = old_flags["global_default"]
            core.USE_LOG_HAZARDS_HS = old_flags["hS_override"]
            core.USE_LOG_HAZARDS_HT = old_flags["hT_override"]
            core.USE_LOG_HAZARDS_RST = old_flags["rST_override"]

    def test_hill_hazard_forms_and_log_layering(self):
        old_log = core.get_log_hazard_flags()
        old_hill = core.get_hill_hazard_flags()
        try:
            C = 10.0
            a = 24.0
            p = STDPParams(
                kT=2.0,
                kS_kT_ratio=3.0,
                kST=1.5,
                K=5.0,
                KST=7.0,
                n=2.0,
                nST=3.0,
                a50=20.0,
                r0=0.1,
            )

            core.set_hill_hazard_flags(global_default=False, hS=True, hT=True, rST=True)
            core.set_log_hazard_flags(global_default=False, hS=False, hT=False, rST=False)

            ht_hill = p.kT * (C**p.n) / ((p.K**p.n) + (C**p.n))
            hs_hill = p.kT * p.kS_kT_ratio * (C**p.n) / ((p.K**p.n) + (C**p.n))
            rst_hill_inner = p.kST * (C**p.nST) / ((p.KST**p.nST) + (C**p.nST))
            rst_hill = core.m(a, p.a50) * (p.r0 + rst_hill_inner)

            self.assertAlmostEqual(core.hT(C, p.kT, p.n, p.K), ht_hill, places=12)
            self.assertAlmostEqual(core.hS(C, p.kT, p.kS_kT_ratio, p.n, p.K), hs_hill, places=12)
            self.assertAlmostEqual(
                core.rST(C, a, p.kST, p.nST, p.a50, p.r0, p.KST),
                rst_hill,
                places=12,
            )

            core.set_log_hazard_flags(hT=True)
            self.assertAlmostEqual(core.hT(C, p.kT, p.n, p.K), np.log1p(ht_hill), places=12)
        finally:
            core.USE_LOG_HAZARDS = old_log["global_default"]
            core.USE_LOG_HAZARDS_HS = old_log["hS_override"]
            core.USE_LOG_HAZARDS_HT = old_log["hT_override"]
            core.USE_LOG_HAZARDS_RST = old_log["rST_override"]
            core.USE_HILL_HAZARD = old_hill["global_default"]
            core.USE_HILL_HAZARD_HS = old_hill["hS_override"]
            core.USE_HILL_HAZARD_HT = old_hill["hT_override"]
            core.USE_HILL_HAZARD_RST = old_hill["rST_override"]

    def test_hill_lambda_constraint_penalty(self):
        old_hill = core.get_hill_hazard_flags()
        try:
            core.set_hill_hazard_flags(global_default=False, hS=False, hT=True, rST=False)

            p_bad = STDPParams(k_lag=6, mu0=0.2, mu24p=0.2, kT=1.0)
            bad_pen = _hill_lambda_constraint_penalty(p_bad, ages=(24.0,), weight=1e6)
            self.assertGreater(bad_pen, 0.0)

            p_good = STDPParams(k_lag=6, mu0=2.0, mu24p=2.0, kT=2.0)
            good_pen = _hill_lambda_constraint_penalty(p_good, ages=(24.0,), weight=1e6)
            self.assertEqual(good_pen, 0.0)

            core.set_hill_hazard_flags(hT=False)
            off_pen = _hill_lambda_constraint_penalty(p_bad, ages=(24.0,), weight=1e6)
            self.assertEqual(off_pen, 0.0)
        finally:
            core.USE_HILL_HAZARD = old_hill["global_default"]
            core.USE_HILL_HAZARD_HS = old_hill["hS_override"]
            core.USE_HILL_HAZARD_HT = old_hill["hT_override"]
            core.USE_HILL_HAZARD_RST = old_hill["rST_override"]

    def test_load_data_from_canonical_dataframe(self):
        df = pd.DataFrame(
            {
                "condition": ["a"],
                "C": [1.0],
                "tau": [2.0],
                "age": [24.0],
                "incomplete": [0.1],
                "induced": [0.2],
                "preexisting": [0.3],
            }
        )
        data = load_data(df)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0][0], "a")

    def test_load_data_rejects_invalid_fraction_sum(self):
        df = pd.DataFrame(
            {
                "condition": ["bad"],
                "C": [1.0],
                "tau": [2.0],
                "age": [24.0],
                "incomplete": [0.6],
                "induced": [0.4],
                "preexisting": [0.2],
            }
        )
        with self.assertRaises(ValueError):
            load_data(df)

    def test_fit_smoke_fast_profile(self):
        data = make_dataset()
        start = STDPParams()
        cfg = FitConfig(profile="custom", n_starts=2, maxiter=40, seed=7, jitter_sigma=0.2)
        result = fit_stdp(data=data, start=start, config=cfg)
        self.assertTrue(np.isfinite(result.best_value))
        self.assertEqual(len(result.free_keys), len(result.best_x))

    def test_prediction_ci_schema(self):
        data = make_dataset()
        params = STDPParams()
        free_keys = ["mu0", "mu24p", "kT", "kS_kT_ratio", "kST", "n", "nST", "a50", "r0"]
        x = _pack_params(params, free_keys)
        draws = np.tile(x, (50, 1))
        ci_df = compute_prediction_ci(draws, params, free_keys, data)
        self.assertIn("pred_ci_lo_incomplete", ci_df.columns)
        self.assertIn("pred_ci_hi_preexisting", ci_df.columns)
        self.assertEqual(len(ci_df), len(data))

    def test_plot_fraction_returns_fig_and_ax(self):
        data = make_dataset()
        model = STDPModel(STDPParams())
        df_fit = evaluate_model(model, data)
        fig, ax = plot_fraction_comparison("incomplete", df_fit=df_fit)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_import_side_effect_free(self):
        pkg = importlib.import_module("model")
        shim = importlib.import_module("model.model")
        self.assertTrue(hasattr(pkg, "ModelWorkflow"))
        self.assertIsNone(getattr(shim, "_COMPAT_DATA"))

    def test_compat_shim_fit_signature(self):
        shim = importlib.import_module("model.model")
        data = make_dataset()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            p, val, info = shim.fit_stdp(
                data=data,
                start=STDPParams(),
                config=FitConfig(profile="custom", n_starts=1, maxiter=10, seed=1, jitter_sigma=0.1),
            )
        self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))
        self.assertIsInstance(p, STDPParams)
        self.assertTrue(np.isfinite(val))


if __name__ == "__main__":
    unittest.main()
