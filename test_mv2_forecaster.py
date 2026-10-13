"""Unit tests for the pure parts of mv2_forecaster (no network).

Run from this folder with `python -m unittest -v`, or from anywhere with
`python path/to/test_mv2_forecaster.py -v`.
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mv2_forecaster as mf  # noqa: E402


def numeric_q(rmin, rmax, lo_open, hi_open, zero_point=None, qtype="numeric", count=None):
    scaling = {"range_min": rmin, "range_max": rmax, "zero_point": zero_point}
    if count is not None:
        scaling["inbound_outcome_count"] = count
    return {"id": 7, "title": "How many?", "type": qtype, "scaling": scaling,
            "open_lower_bound": lo_open, "open_upper_bound": hi_open}


WIDE = {5: 30, 10: 35, 20: 40, 30: 45, 40: 48, 50: 50, 60: 52, 70: 55, 80: 60, 90: 65, 95: 70}


class ParseTests(unittest.TestCase):
    def test_binary(self):
        self.assertAlmostEqual(mf.parse_binary("blah\nProbability: 23%"), 0.23)
        self.assertAlmostEqual(mf.parse_binary("Probability: 12.5 %"), 0.125)
        self.assertAlmostEqual(mf.parse_binary("Probability: 10%\n...\nprobability: 40%"), 0.40)
        self.assertIsNone(mf.parse_binary("a 5% base rate, no final line"))
        self.assertIsNone(mf.parse_binary("Probability: 140%"))

    def test_multiple_choice(self):
        text = "Thinking...\n**Yes**: 50%\nYes, partially: 30%\nNo: 20%"
        got = mf.parse_multiple_choice(text, ["Yes", "Yes, partially", "No"])
        self.assertAlmostEqual(got["Yes"], 0.5)
        self.assertAlmostEqual(got["Yes, partially"], 0.3)
        self.assertAlmostEqual(sum(got.values()), 1.0)
        normalized = mf.parse_multiple_choice("A: 2\nB: 2", ["A", "B"])
        self.assertAlmostEqual(normalized["A"], 0.5)
        self.assertIsNone(mf.parse_multiple_choice("A: 50%", ["A", "B"]))

    def test_percentiles(self):
        got = mf.parse_percentiles("Percentile 10: 1,200\nPercentile 50: $3,500.5\npercentile 90: 1e4")
        self.assertEqual(got, {10: 1200.0, 50: 3500.5, 90: 10000.0})
        self.assertIsNone(mf.parse_percentiles("Percentile 10: 1"))


class AggregateTests(unittest.TestCase):
    def test_binary_log_odds(self):
        self.assertAlmostEqual(mf.aggregate_binary([0.2, 0.8]), 0.5, places=4)
        self.assertAlmostEqual(mf.aggregate_binary([0.3, 0.5]), 0.3956, places=3)
        self.assertEqual(mf.aggregate_binary([0.999, 0.999]), mf.BINARY_CEIL)
        self.assertEqual(mf.aggregate_binary([0.0]), mf.BINARY_FLOOR)

    def test_multiple_choice(self):
        got = mf.aggregate_multiple_choice([{"A": 1.0, "B": 0.0, "C": 0.0},
                                            {"A": 0.8, "B": 0.2, "C": 0.0}], ["A", "B", "C"])
        self.assertAlmostEqual(sum(got.values()), 1.0, places=6)
        self.assertGreaterEqual(min(got.values()), mf.MC_FLOOR * 0.95)
        self.assertGreater(got["A"], got["B"])

    def test_cdf_mean(self):
        self.assertEqual(mf.aggregate_cdfs([[0, 0.5, 1], [0, 0.25, 1]]), [0, 0.375, 1])


class CdfTests(unittest.TestCase):
    def test_location_to_value(self):
        self.assertAlmostEqual(mf.location_to_value(0, 10, 20, None), 10)
        self.assertAlmostEqual(mf.location_to_value(0.5, 10, 20, None), 15)
        self.assertAlmostEqual(mf.location_to_value(0, 1, 1e6, 0), 1)
        self.assertAlmostEqual(mf.location_to_value(1, 1, 1e6, 0), 1e6)
        self.assertAlmostEqual(mf.location_to_value(0.5, 1, 1e6, 0), 1000, places=6)

    def test_all_bound_combinations_valid(self):
        for lo_open in (False, True):
            for hi_open in (False, True):
                q = numeric_q(0, 100, lo_open, hi_open)
                cdf = mf.cdf_from_percentiles(WIDE, q)
                self.assertEqual(mf.cdf_problems(cdf, q), [], (lo_open, hi_open))

    def test_median_preserved(self):
        q = numeric_q(0, 100, True, True)
        cdf = mf.cdf_from_percentiles(WIDE, q)
        self.assertAlmostEqual(float(mf.median_text(cdf, q)), 50, delta=1)

    def test_narrow_forecast_is_capped(self):
        q = numeric_q(0, 1000, False, False)
        cdf = mf.cdf_from_percentiles({10: 500, 50: 500.2, 90: 500.4}, q)
        self.assertEqual(mf.cdf_problems(cdf, q), [])

    def test_percentiles_outside_range(self):
        q = numeric_q(0, 10, False, True)
        cdf = mf.cdf_from_percentiles({10: 20, 50: 40, 90: 80}, q)
        self.assertEqual(mf.cdf_problems(cdf, q), [])
        q = numeric_q(0, 10, True, False)
        cdf = mf.cdf_from_percentiles({10: -50, 50: -20, 90: -5}, q)
        self.assertEqual(mf.cdf_problems(cdf, q), [])

    def test_unsorted_and_tied_values(self):
        q = numeric_q(0, 100, True, True)
        cdf = mf.cdf_from_percentiles({10: 60, 50: 40, 90: 40}, q)
        self.assertEqual(mf.cdf_problems(cdf, q), [])

    def test_log_scaled(self):
        q = numeric_q(1, 1e6, False, True, zero_point=0)
        cdf = mf.cdf_from_percentiles({10: 100, 50: 1000, 90: 20000}, q)
        self.assertEqual(mf.cdf_problems(cdf, q), [])
        self.assertTrue(900 < float(mf.median_text(cdf, q)) < 1300)

    def test_discrete(self):
        q = numeric_q(-0.5, 10.5, False, False, qtype="discrete", count=11)
        cdf = mf.cdf_from_percentiles({10: 2, 50: 4, 90: 7}, q)
        self.assertEqual(len(cdf), 12)
        self.assertEqual(mf.cdf_problems(cdf, q), [])

    def test_problems_detected(self):
        q = numeric_q(0, 100, False, False)
        self.assertTrue(mf.cdf_problems([0.0] * 201, q))
        self.assertTrue(mf.cdf_problems([0.0, 1.0], q))


class ApiShapeTests(unittest.TestCase):
    def test_payloads(self):
        p = mf.forecast_payload(5, "binary", 0.3)
        self.assertEqual(p["probability_yes"], 0.3)
        self.assertIsNone(p["continuous_cdf"])
        p = mf.forecast_payload(5, "multiple_choice", {"A": 1.0})
        self.assertEqual(p["probability_yes_per_category"], {"A": 1.0})
        p = mf.forecast_payload(5, "discrete", [0, 1])
        self.assertEqual(p["continuous_cdf"], [0, 1])
        with self.assertRaises(ValueError):
            mf.forecast_payload(5, "date", 1)

    def test_already_forecast(self):
        self.assertTrue(mf.already_forecast({"my_forecasts": {"latest": {"forecast_values": [0.7, 0.3]}}}))
        self.assertFalse(mf.already_forecast({"my_forecasts": {"latest": None}}))
        self.assertFalse(mf.already_forecast({}))

    def test_pick_models(self):
        listing = {"data": [
            {"id": "anthropic/claude-sonnet-4.5", "created": 10},
            {"id": "anthropic/claude-sonnet-5", "created": 20},
            {"id": "anthropic/claude-opus-5", "created": 30},
            {"id": "openai/gpt-6", "created": 25},
            {"id": "openai/gpt-6-mini", "created": 40},
            {"id": "openai/gpt-7:free", "created": 50},
            {"id": "google/gemini-3-pro", "created": 5},
            {"id": "google/gemini-3-pro-image", "created": 9},
            {"id": "meta/llama-5", "created": 99},
        ]}
        self.assertEqual(mf.pick_models(listing),
                         ["anthropic/claude-sonnet-5", "openai/gpt-6", "google/gemini-3-pro"])
        self.assertEqual(mf.pick_models({}), [])


def fake_llm(replies):
    def llm(model, prompt, base, key, temperature=0.4, web=False):
        if web:
            return "Recent news: nothing has changed."
        return replies[model]
    return llm


class ForecastQuestionTests(unittest.TestCase):
    cfg = {"base_url": "http://x", "api_key": "k", "runs_per_model": 1, "today": "2026-10-13"}

    def test_binary(self):
        q = {"id": 1, "title": "Will X happen?", "type": "binary"}
        llm = fake_llm({"a": "Probability: 30%", "b": "Probability: 50%"})
        value, comment, record = mf.forecast_question(q, ["a", "b"], "a", self.cfg, llm, lambda t: [])
        self.assertAlmostEqual(value, 0.3956, places=3)
        self.assertIn("an AI", comment)
        self.assertIn("No human in the loop", comment)
        self.assertEqual(len(record["runs"]), 2)

    def test_multiple_choice(self):
        q = {"id": 2, "title": "Which?", "type": "multiple_choice", "options": ["A", "B", "C"]}
        llm = fake_llm({"a": "A: 50%\nB: 30%\nC: 20%"})
        value, _, _ = mf.forecast_question(q, ["a"], "a", self.cfg, llm, lambda t: [])
        self.assertAlmostEqual(sum(value.values()), 1.0, places=6)

    def test_numeric(self):
        q = numeric_q(0, 100, True, True)
        reply = "\n".join(f"Percentile {p}: {v}" for p, v in WIDE.items())
        value, _, _ = mf.forecast_question(q, ["a"], "a", self.cfg, fake_llm({"a": reply}), lambda t: [])
        self.assertEqual(mf.cdf_problems(value, q), [])

    def test_no_usable_answers(self):
        q = {"id": 1, "title": "Will X happen?", "type": "binary"}
        with self.assertRaises(ValueError):
            mf.forecast_question(q, ["a"], "a", self.cfg, fake_llm({"a": "no idea"}), lambda t: [])

    def test_prompts_mention_bounds_and_format(self):
        q = numeric_q(0, 100, False, True)
        prompt = mf.forecast_prompt(q, "2026-10-13", "", [])
        self.assertIn("closed: the outcome cannot be lower", prompt)
        self.assertIn("open: the outcome can be higher", prompt)
        self.assertIn("Percentile 95: X", prompt)


class MainTests(unittest.TestCase):
    def test_without_credentials_does_nothing(self):
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=True), redirect_stdout(buf):
            self.assertEqual(mf.main([]), 0)
        self.assertIn("nothing to do", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
