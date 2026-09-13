"""Unit tests for GenAI LER / yield / TND preparation rules."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.experimentutils.ler_prep import (  # noqa: E402
    calculate_ler_from_yields,
    calculate_tnd,
    convert_yield_to_t_ha,
    prepare_ler_row,
)


class YieldConversionTests(unittest.TestCase):
    def test_common_mass_per_area_units(self):
        cases = [
            (1.0, "t/ha", 1.0),
            (1.0, "Mg/ha", 1.0),
            (1000.0, "kg/ha", 1.0),
            (100.0, "g/m²", 1.0),
            (100.0, "g DM m-2", 1.0),
            (10.0, "q/ha", 1.0),
        ]
        for mag, unit, expected in cases:
            t_ha, canon, _, err = convert_yield_to_t_ha(mag, unit)
            self.assertIsNone(err, msg=(mag, unit, err))
            self.assertIsNotNone(canon)
            self.assertAlmostEqual(t_ha, expected, places=6)

    def test_incompatible_and_missing_units(self):
        self.assertEqual(convert_yield_to_t_ha(1.0, None)[3], "missing_yield_unit")
        self.assertEqual(convert_yield_to_t_ha(1.0, "g plant-1")[3], "incompatible_yield_unit")
        self.assertEqual(convert_yield_to_t_ha(1.0, "kg N ha-1")[3], "incompatible_yield_unit")


class LERDecisionTests(unittest.TestCase):
    def test_reported_ler_precedence(self):
        row = {
            "LER": "1.25",
            "unified yield sc 1": 4,
            "unified yield sc 2": 4,
            "unified yield ic 1": 3,
            "unified yield ic 2": 3,
            "Yield unit": "t/ha",
            "TND": "0.2",
        }
        prep = prepare_ler_row(row)
        self.assertEqual(prep.LER_source, "reported")
        self.assertAlmostEqual(prep.LER_final, 1.25)
        self.assertAlmostEqual(prep.LER_from_yields, 1.5)
        self.assertEqual(prep.LER_agreement, "disagree")

    def test_calculate_when_reported_missing(self):
        row = {
            "LER": "",
            "unified yield sc 1": 4,
            "unified yield sc 2": 4,
            "unified yield ic 1": 3,
            "unified yield ic 2": 3,
            "Yield unit": "t/ha",
        }
        prep = prepare_ler_row(row)
        self.assertEqual(prep.LER_source, "calculated_from_yields")
        self.assertAlmostEqual(prep.LER_final, 1.5)

    def test_exclude_when_both_missing(self):
        prep = prepare_ler_row({"LER": "", "Yield unit": ""})
        self.assertEqual(prep.LER_source, "excluded")
        self.assertIsNotNone(prep.exclusion_reason)

    def test_range_and_zero_sole(self):
        self.assertEqual(prepare_ler_row({"LER": "1.1–1.4"}).exclusion_reason, "ambiguous_numeric_range")
        ler, reason = calculate_ler_from_yields(0.0, 4.0, 1.0, 1.0)
        self.assertIsNone(ler)
        self.assertEqual(reason, "zero_sole_crop_yield")

    def test_no_partial_ler(self):
        ler, reason = calculate_ler_from_yields(4.0, None, 2.0, 1.0)
        self.assertIsNone(ler)
        self.assertEqual(reason, "missing_sole_yield_2")


class TNDTests(unittest.TestCase):
    def test_reported_tnd_preferred(self):
        prep = prepare_ler_row(
            {
                "LER": "1.1",
                "TND": "0.4",
                "Sowing date 1": 35509,
                "Sowing date 2": 35536,
                "Harvest date 1": 35626,
                "Harvest date 2": 35697,
            }
        )
        self.assertEqual(prep.TND_source, "reported")
        self.assertAlmostEqual(prep.TND_final, 0.4)
        self.assertIsNotNone(prep.TND_from_dates)

    def test_calculate_tnd_from_excel_serials(self):
        # Study 4 style dates from the Yu spreadsheet.
        tnd, err = calculate_tnd(35509, 35536, 35626, 35697)
        self.assertIsNone(err)
        self.assertAlmostEqual(tnd, 98 / 189, places=6)

    def test_simultaneous_relative_harvest(self):
        tnd, err = calculate_tnd(29327, 29327, "13-15 weeks later", "13-15 weeks later")
        self.assertIsNone(err)
        self.assertAlmostEqual(tnd, 0.0, places=6)


class RowCountReconcileTests(unittest.TestCase):
    def test_prepare_preserves_row_count(self):
        from src.experimentutils.ler_prep import prepare_ler_records

        rows = [
            {"LER": "1.1", "TND": "0"},
            {"LER": "", "Yield unit": "t/ha"},
            {"LER": "1.1-1.4"},
        ]
        out = prepare_ler_records(rows)
        self.assertEqual(len(out), len(rows))


class EligibilityAndMatchingTests(unittest.TestCase):
    def test_only_eligible_rows_are_matched(self):
        from src.experimentutils.ler_reproduction import (
            attach_experimental_ids,
            mark_regression_eligibility,
        )

        prepared = pd.DataFrame(
            [
                {
                    "study_id": 1,
                    "LER_final": 1.2,
                    "LER_source": "reported",
                    "TND_final": 0.1,
                    "TND_source": "reported",
                    "Crop species 1": "pea",
                    "Crop species 2": "barley",
                    "Year of data": "1996",
                    "Data source": "table 1",
                },
                {
                    "study_id": 1,
                    "LER_final": 1.3,
                    "LER_source": "reported",
                    "TND_final": None,
                    "TND_source": "excluded",
                    "Crop species 1": "pea",
                    "Crop species 2": "barley",
                    "Year of data": "1996",
                    "Data source": "table 1",
                },
            ]
        )
        flagged = mark_regression_eligibility(prepared)
        self.assertTrue(bool(flagged.loc[0, "regression_eligible"]))
        self.assertFalse(bool(flagged.loc[1, "regression_eligible"]))

        gt_by_id = {
            1: [
                {
                    "Experimental ID": "E1",
                    "Crop species 1": "pea",
                    "Crop species 2": "barley",
                    "Year of data": "1996",
                    "Data source": "table 1",
                    "LER": 9.9,
                    "TND": 0.9,
                }
            ]
        }
        matched = attach_experimental_ids(prepared, gt_by_id)
        self.assertTrue(bool(matched.loc[0, "matched_gt"]))
        self.assertEqual(matched.loc[0, "Experimental ID"], "E1")
        self.assertFalse(bool(matched.loc[1, "matched_gt"]))
        self.assertEqual(matched.loc[1, "match_exclusion_reason"], "missing_final_tnd")

    def test_shared_experimental_id_allowed_across_gt_rows(self):
        from src.experimentutils.ler_reproduction import attach_experimental_ids

        prepared = pd.DataFrame(
            [
                {
                    "study_id": 1,
                    "LER_final": 1.1,
                    "LER_source": "reported",
                    "TND_final": 0.0,
                    "TND_source": "reported",
                    "Crop species 1": "pea",
                    "Crop species 2": "barley",
                    "Year of data": "1996",
                    "N input SC1": "0",
                    "Data source": "a",
                },
                {
                    "study_id": 1,
                    "LER_final": 1.2,
                    "LER_source": "reported",
                    "TND_final": 0.0,
                    "TND_source": "reported",
                    "Crop species 1": "pea",
                    "Crop species 2": "barley",
                    "Year of data": "1996",
                    "N input SC1": "100",
                    "Data source": "b",
                },
            ]
        )
        gt_by_id = {
            1: [
                {
                    "Experimental ID": "SAME",
                    "Crop species 1": "pea",
                    "Crop species 2": "barley",
                    "Year of data": "1996",
                    "N input SC1": "0",
                    "Data source": "a",
                },
                {
                    "Experimental ID": "SAME",
                    "Crop species 1": "pea",
                    "Crop species 2": "barley",
                    "Year of data": "1996",
                    "N input SC1": "100",
                    "Data source": "b",
                },
            ]
        }
        matched = attach_experimental_ids(prepared, gt_by_id)
        self.assertTrue(bool(matched.loc[0, "matched_gt"]))
        self.assertTrue(bool(matched.loc[1, "matched_gt"]))
        self.assertEqual(matched.loc[0, "Experimental ID"], "SAME")
        self.assertEqual(matched.loc[1, "Experimental ID"], "SAME")
        self.assertNotEqual(
            matched.loc[0, "matched_gt_row_index"],
            matched.loc[1, "matched_gt_row_index"],
        )


if __name__ == "__main__":
    unittest.main()
