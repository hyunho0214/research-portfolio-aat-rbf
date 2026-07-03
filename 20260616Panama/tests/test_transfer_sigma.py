import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.transfer_sigma import fit_transfer_table, gaussian_with_baseline, load_sigma_values, parse_vd_from_column


class TransferSigmaTests(unittest.TestCase):
    def test_parse_vd_from_column(self):
        self.assertEqual(parse_vd_from_column("VD_5V_ID"), 5.0)
        self.assertEqual(parse_vd_from_column("ID at VD=2.5 V"), 2.5)
        self.assertIsNone(parse_vd_from_column("current"))

    def test_fit_transfer_table_recovers_sigma(self):
        vg = np.arange(0.0, 30.0001, 0.25)
        y1 = gaussian_with_baseline(vg, 2.0e-9, 18.0, 2.5, 1.0e-12)
        y2 = gaussian_with_baseline(vg, 1.0e-9, 19.0, 3.2, 2.0e-12)
        df = pd.DataFrame({"VG": vg, "VD_1V_ID": y1, "VD_2V_ID": y2})

        result, vg_column, id_columns = fit_transfer_table(df)

        self.assertEqual(vg_column, "VG")
        self.assertEqual(id_columns, ["VD_1V_ID", "VD_2V_ID"])
        self.assertTrue(result["success"].all())
        self.assertGreater(result["r2"].min(), 0.999)
        sigma_by_col = dict(zip(result["column"], result["sigma"]))
        self.assertAlmostEqual(sigma_by_col["VD_1V_ID"], 2.5, places=2)
        self.assertAlmostEqual(sigma_by_col["VD_2V_ID"], 3.2, places=2)

    def test_load_sigma_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sigmas.xlsx"
            pd.DataFrame({"sigma": [0.3, 1.0, np.nan, -2.0]}).to_excel(path, index=False)
            values = load_sigma_values(path)
        np.testing.assert_allclose(values, [0.3, 1.0])


if __name__ == "__main__":
    unittest.main()
