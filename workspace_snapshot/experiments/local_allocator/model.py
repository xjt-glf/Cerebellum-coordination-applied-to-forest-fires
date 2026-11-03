"""obs_v2 set networks. No identities, privileged metrics or fixed fleet width."""
import numpy as np
import torch
from torch import nn


def tensor_obs(obs, device="cpu"):
    return {k: torch.as_tensor(v, device=device, dtype=torch.float32) for k, v in obs.items()}


def pool(encoded, mask):
    mask = mask.unsqueeze(-1)
    mean = (encoded * mask).sum(-2) / mask.sum(-2).clamp_min(1)
    maximum = encoded.masked_fill(mask == 0, -1e9).amax(-2)
    maximum = torch.where(mask.sum(-2) > 0, maximum, torch.zeros_like(maximum))
    count = mask.sum(-2)
    return torch.cat((mean, maximum, count / 16, (count == 0).float()), -1)


class SetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.uav = nn.Sequential(nn.Linear(16, 64), nn.ReLU())
        self.person = nn.Sequential(nn.Linear(8, 64), nn.ReLU())

    def forward(self, obs):
        return torch.cat((pool(self.uav(obs["mates"]), obs["mate_mask"]),
                          pool(self.person(obs["persons"]), obs["person_mask"]),
                          obs["context"]), -1)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.sets = SetEncoder()
        self.net = nn.Sequential(nn.Linear(16 + 260 + 72, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))

    def forward(self, obs):
        return self.net(torch.cat((obs["self"], self.sets(obs)), -1))

    @torch.no_grad()
    def act(self, local_observation):
        """Independent deployment call: one local observation -> integer action."""
        logits = self(tensor_obs(stack_obs([local_observation]), next(self.parameters()).device))
        return int(logits.argmax(-1).item())


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.sets = SetEncoder()
        self.net = nn.Sequential(nn.Linear(260 + 72, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, team):
        return self.net(self.sets(team)).squeeze(-1)


def stack_obs(items):
    """Pad each set independently; zero padding never contributes to pooling."""
    result = {}
    for k in items[0]:
        values = [x[k] for x in items]
        if k in ("mates", "persons", "mate_mask", "person_mask"):
            width = max(1, max(len(x) for x in values))
            padded = np.zeros((len(items), width, *values[0].shape[1:]), np.float32)
            for i, x in enumerate(values):
                padded[i, :len(x)] = x
            result[k] = padded
        else:
            result[k] = np.stack(values)
    return result
