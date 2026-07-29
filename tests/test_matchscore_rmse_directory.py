import math
import unittest

from src.matchscore import compute_rmse_for_directory


class ComputeRmseForDirectoryTests(unittest.TestCase):
    def test_compute_rmse_for_directory(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "a.csv").write_text("1,2,3\n", encoding="utf-8")
            (tmp_path / "b.txt").write_text("4\n5\n6\n", encoding="utf-8")

            results = compute_rmse_for_directory(tmp_path)

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0][0], "a.csv")
            self.assertEqual(results[1][0], "b.txt")
            self.assertAlmostEqual(results[0][1], math.sqrt(14.0 / 3.0))
            self.assertAlmostEqual(results[1][1], math.sqrt(77.0 / 3.0))


if __name__ == "__main__":
    unittest.main()
