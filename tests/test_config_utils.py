import tempfile
import unittest
from pathlib import Path

from script.config_utils import get_frame_ratio, get_output_base_dir, resolve_output_dir


class TestConfigUtils(unittest.TestCase):
    def test_get_output_base_dir_relative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            cfg = {"output": {"base_dir": "output"}}
            self.assertEqual(get_output_base_dir(base_dir, cfg), (base_dir / "output").resolve())

    def test_resolve_output_dir_new_relative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            cfg = {"output": {"base_dir": "output"}}
            self.assertEqual(
                resolve_output_dir(base_dir, cfg, "step_TF"),
                (base_dir / "output" / "step_TF").resolve(),
            )

    def test_resolve_output_dir_legacy_prefixed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            cfg = {"output": {"base_dir": "output"}}
            self.assertEqual(
                resolve_output_dir(base_dir, cfg, "output/step_TF"),
                (base_dir / "output" / "step_TF").resolve(),
            )

    def test_resolve_output_dir_legacy_nested_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            cfg = {"output": {"base_dir": "out/base"}}
            self.assertEqual(
                resolve_output_dir(base_dir, cfg, "out/base/step_TF"),
                (base_dir / "out" / "base" / "step_TF").resolve(),
            )

    def test_resolve_output_dir_absolute_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            cfg = {"output": {"base_dir": "/tmp/agg_viz_out"}}
            self.assertEqual(
                resolve_output_dir(base_dir, cfg, "step_TF"),
                Path("/tmp/agg_viz_out/step_TF").resolve(),
            )

    def test_resolve_output_dir_absolute_mode_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            cfg = {"output": {"base_dir": "output"}}
            self.assertEqual(
                resolve_output_dir(base_dir, cfg, "/tmp/agg_viz_custom"),
                Path("/tmp/agg_viz_custom"),
            )

    def test_get_frame_ratio_configured(self) -> None:
        cfg = {"device_sample_rate": 1000, "mocap_sample_rate": 100, "frame_ratio": 20}
        self.assertEqual(get_frame_ratio(cfg), 20)

    def test_get_frame_ratio_fallback(self) -> None:
        cfg = {"device_sample_rate": 1000, "mocap_sample_rate": 100}
        self.assertEqual(get_frame_ratio(cfg), 10)

    def test_get_frame_ratio_invalid_configured(self) -> None:
        cfg = {"device_sample_rate": 1000, "mocap_sample_rate": 100, "frame_ratio": "bad"}
        self.assertEqual(get_frame_ratio(cfg), 10)


if __name__ == "__main__":
    unittest.main()

