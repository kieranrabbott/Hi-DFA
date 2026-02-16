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
from model.objective import _pack_params
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
