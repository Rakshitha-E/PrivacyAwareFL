import unittest

import numpy as np

import run_iid_noniid


class PartitionAuditTests(unittest.TestCase):
    def test_partition_audit_reports_exact_coverage(self):
        partitions = {
            1: np.array([0, 2]),
            2: np.array([1, 3]),
        }

        audit = run_iid_noniid._verify_partition(partitions, sample_count=4)

        self.assertEqual(audit["training_rows"], 4)
        self.assertEqual(audit["assigned_rows"], 4)
        self.assertEqual(audit["unique_assigned_rows"], 4)
        self.assertEqual(audit["duplicate_assignments"], 0)
        self.assertEqual(audit["missing_training_rows"], 0)

    def test_partition_audit_rejects_duplicate_and_missing_rows(self):
        partitions = {
            1: np.array([0, 1]),
            2: np.array([1, 3]),
        }

        with self.assertRaisesRegex(RuntimeError, "lost or duplicated"):
            run_iid_noniid._verify_partition(partitions, sample_count=4)


if __name__ == "__main__":
    unittest.main()
