import unittest

import numpy as np

from world_state.adapter import WorldStateAdapter
from world_state.encoder import FrameEncoder


class WorldStateAdapterTest(unittest.TestCase):
    def test_encoder_output_dim(self):
        encoder = FrameEncoder(frame_size=8, latent_dim=16, seed=1)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        latent = encoder.encode_frame(frame)
        self.assertEqual(latent.shape, (16,))

    def test_build_state(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        obs = {
            "frame_id": 12,
            "timestamp": 123.4,
            "yolo_objects": [{"label": "enemy", "confidence": 0.8, "bbox": [0, 0, 1, 1]}],
            "text_zones": {"dialog": {"text": "Play", "confidence": 0.9, "bbox": [0, 0, 1, 1]}},
        }
        adapter = WorldStateAdapter(encoder=FrameEncoder(frame_size=8, latent_dim=8))
        state = adapter.build_state(observation=obs, frame=frame, reward=0.5)
        self.assertEqual(state["frame_id"], 12)
        self.assertEqual(state["object_count"], 1)
        self.assertEqual(state["text_count"], 1)
        self.assertEqual(state["reward"], 0.5)
        self.assertEqual(len(state["latent"]), 8)


if __name__ == "__main__":
    unittest.main()
