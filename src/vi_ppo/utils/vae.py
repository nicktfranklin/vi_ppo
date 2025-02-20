import torch

from vi_ppo.actor_critic import ActorCritic
from vi_ppo.buffer import RolloutBuffer


class StateHasher:

    def __init__(self, z_dim, z_layers):
        self.z_dim = z_dim
        self.z_layers = z_layers

        self.observed_vectors = torch.empty((0, z_layers), dtype=torch.float32)

    def convert_from_onehot(self, x):
        return torch.argmax(x.view(self.z_layers, self.z_dim), dim=-1)

    def __call__(self, s):
        s = self.convert_from_onehot(s)
        exists = torch.any(torch.all(self.observed_vectors == s, dim=1))
        if exists:
            print(torch.where(exists))
            return torch.where(exists)[0][0]
        else:
            self.observed_vectors = torch.cat(
                [self.observed_vectors, s.unsqueeze(0)], dim=0
            )
            return len(self.observed_vectors) - 1


def estimate_graph_laplacian(
    module: ActorCritic, buffer: RolloutBuffer, normalized: bool = True
) -> torch.Tensor:
    module.train()

    # easy enough to loop through twice
    hasher = StateHasher(
        module.state_vae.config.z_dim, module.state_vae.config.z_layers
    )
    pairs = []
    for o, op in zip(buffer.observations, buffer.next_observations):
        o = torch.from_numpy(o).float()
        op = torch.from_numpy(op).float()
        with torch.no_grad():
            s = module.get_state(o)
            sp = module.get_state(op)
        s = hasher(s)
        sp = hasher(sp)
        pairs.append([s, sp])

    print(pairs)

    raise Exception("stop")
    adjacency_matrix = torch.zeros(hasher.n, hasher.n, device=hasher.device)
    for o, op in zip(buffer.observations, buffer.next_observations):
        with torch.no_grad():
            o = torch.from_numpy(o).float()
            op = torch.from_numpy(op).float()
            s = module.get_state(o)
            sp = module.get_state(op)
            print(hasher(s), hasher(sp))
        adjacency_matrix[hasher(s), hasher(sp)] = 1

    print(adjacency_matrix, adjacency_matrix.sum(dim=1))
    degree_matrix = torch.diag(adjacency_matrix.sum(dim=1))

    if normalized:
        sqrt_degree = torch.diag(
            1.0 / torch.sqrt(torch.clamp(degree_matrix.diag(), min=1e-10))
        )
        laplacian_matrix = torch.eye(hasher.n, device=hasher.device) - torch.matmul(
            sqrt_degree, torch.matmul(adjacency_matrix, sqrt_degree)
        )
    else:
        laplacian_matrix = degree_matrix - adjacency_matrix

    return laplacian_matrix
