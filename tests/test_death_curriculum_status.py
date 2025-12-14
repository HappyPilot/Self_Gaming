import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

from agents.curriculum_status import DeathCurriculumStatus
import tools.death_curriculum_status as cli


class DeathCurriculumCliTest(unittest.TestCase):
    def setUp(self):
        self.orig_argv = sys.argv

    def tearDown(self):
        sys.argv = self.orig_argv

    @mock.patch.object(cli, "compute_death_curriculum_status")
    @mock.patch.object(cli, "MemRPC")
    def test_cli_ready_output(self, mock_memrpc, mock_compute):
        mock_instance = mock.Mock()
        mock_memrpc.return_value = mock_instance
        mock_instance.close.return_value = None
        mock_compute.return_value = DeathCurriculumStatus(
            scope="critical_dialog:death",
            profile="poe_default",
            death_count=10,
            resurrect_count=8,
            other_count=0,
            success_rate=0.8,
            rules_count=3,
            calibration_count=2,
            ready=True,
            reason="ready",
        )
        sys.argv = ["prog", "--scope", "critical_dialog:death"]
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main()
        output = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("READY        : YES", output)
        self.assertIn("Success rate : 0.80", output)

    @mock.patch.object(cli, "compute_death_curriculum_status")
    @mock.patch.object(cli, "MemRPC")
    def test_cli_not_ready_note(self, mock_memrpc, mock_compute):
        mock_instance = mock.Mock()
        mock_memrpc.return_value = mock_instance
        mock_instance.close.return_value = None
        mock_compute.return_value = DeathCurriculumStatus(
            scope="critical_dialog:death",
            profile=None,
            death_count=2,
            resurrect_count=0,
            other_count=0,
            success_rate=0.0,
            rules_count=1,
            calibration_count=0,
            ready=False,
            reason="no_calibration",
        )
        sys.argv = ["prog"]
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main()
        output = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("READY        : NO", output)
        self.assertIn("NOTE: requirements", output)


if __name__ == "__main__":
    unittest.main()

