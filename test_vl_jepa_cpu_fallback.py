import unittest
from unittest.mock import patch

import agents.vl_jepa_agent as vl


class _FakeEncoder:
    backend = "siglip2"
    model_id = "fake"

    def encode(self, _image):
        return None


class VlJepaCpuFallbackTest(unittest.TestCase):
    def setUp(self):
        self._old_backend = vl.BACKEND
        self._old_device_raw = vl.DEVICE_RAW

    def tearDown(self):
        vl.BACKEND = self._old_backend
        vl.DEVICE_RAW = self._old_device_raw

    def test_siglip_oom_falls_back_to_cpu_before_dummy(self):
        vl.BACKEND = "siglip2"
        vl.DEVICE_RAW = "cuda"

        calls = []
        
        class _FlakySiglip(_FakeEncoder):
            def __init__(self, _model_id, device, *_args, **_kwargs):
                calls.append(device)
                if device == "cuda":
                    raise RuntimeError("CUDA out of memory")

        with patch.object(vl, "SiglipEncoder", _FlakySiglip):
            agent = vl.VlJepaAgent()
            agent._ensure_encoder()

        self.assertEqual(calls, ["cuda", "cpu"])
        self.assertEqual(agent.encoder.backend, "siglip2")


if __name__ == "__main__":
    unittest.main()
