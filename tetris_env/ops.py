from __future__ import annotations

import torch

from . import _C  # noqa: F401


def start(envs: int, envs_per_thread: int = 1) -> None:
    torch.ops.TetrisEnvBranchless.start(envs, envs_per_thread)


def step(
    actions: torch.Tensor,
    observations: torch.Tensor,
    rewards: torch.Tensor,
    done: torch.Tensor | None = None,
    image_observation: bool = False,
) -> torch.Tensor:
    if done is None:
        done = torch.empty_like(rewards, dtype=torch.bool)
    return torch.ops.TetrisEnvBranchless.step(actions, observations, rewards, done, image_observation)
