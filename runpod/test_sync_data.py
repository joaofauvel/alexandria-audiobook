import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("sync_data.sh")


class SyncDataTests(unittest.TestCase):
    def test_dry_run_pushes_data_without_deleting_remote_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "alexandria"
            data.mkdir()
            (data / "config.json").write_text("{}", encoding="utf-8")

            result = subprocess.run(
                [
                    str(SCRIPT),
                    "push",
                    "--host",
                    "example.invalid",
                    "--port",
                    "1234",
                    "--local-data",
                    str(data),
                    "--dry-run",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("root@example.invalid:/workspace/alexandria/", result.stdout)
        self.assertNotIn("--delete", result.stdout)

    def test_dry_run_pull_includes_cache_only_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "alexandria"
            cache = root / "huggingface"
            data.mkdir()
            cache.mkdir()

            result = subprocess.run(
                [
                    str(SCRIPT),
                    "pull",
                    "--host",
                    "example.invalid",
                    "--port",
                    "1234",
                    "--local-data",
                    str(data),
                    "--local-cache",
                    str(cache),
                    "--cache",
                    "--dry-run",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("/workspace/.cache/huggingface/", result.stdout)
        self.assertIn(str(cache), result.stdout)


if __name__ == "__main__":
    unittest.main()
