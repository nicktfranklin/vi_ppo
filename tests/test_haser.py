import unittest

import torch
from torch import nn

from vi_ppo.actor_critic import ActorCritic, ActorCriticConfig
from vi_ppo.buffer import RolloutBuffer
from vi_ppo.utils.vae import StateHasher, estimate_graph_laplacian


class TestStateHasher(unittest.TestCase):

    def setUp(self):
        self.z_dim = 3
        self.z_layers = 2
        self.hasher = StateHasher(self.z_dim, self.z_layers)

    def test_convert_from_onehot(self):
        onehot = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float32)
        expected = torch.tensor([1, 0])
        result = self.hasher.convert_from_onehot(onehot)
        self.assertTrue(torch.equal(result, expected))

    def test_call_new_state(self):
        onehot = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float32)
        index = self.hasher(onehot)
        self.assertEqual(index, 0)
        self.assertEqual(self.hasher.n, 1)

    def test_call_existing_state(self):
        onehot = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float32)
        self.hasher(onehot)
        index = self.hasher(onehot)
        self.assertEqual(index, 0)
        self.assertEqual(self.hasher.n, 1)

    def test_device(self):
        self.assertEqual(self.hasher.device, torch.device("cpu"))


# class TestEstimateGraphLaplacian(unittest.TestCase):

#     def setUp(self):
#         self.module = ActorCritic(ActorCriticConfig(), nn.Identity(), nn.Identity())
#         self.buffer = RolloutBuffer()
#         # Add mock data to buffer
#         self.buffer.observations = [torch.randn(3, 3) for _ in range(10)]
#         self.buffer.next_observations = [torch.randn(3, 3) for _ in range(10)]

#     def test_estimate_graph_laplacian(self):
#         laplacian_matrix = estimate_graph_laplacian(
#             self.module, self.buffer, normalized=True
#         )
#         self.assertIsInstance(laplacian_matrix, torch.Tensor)
#         self.assertEqual(laplacian_matrix.shape[0], laplacian_matrix.shape[1])


if __name__ == "__main__":
    unittest.main()
