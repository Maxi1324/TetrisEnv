from __future__ import annotations

import torch

from . import _C  # noqa: F401


def start(
    envs: int,
    envs_per_thread: int = 1,
    drop_speed: int = 1,
    survival_reward: float = 0.01,
    placed_block_reward: float = 0.1,
    line_clear_reward: float = 1.0,
    game_over_reward: float = -1.0,
) -> None:
    torch.ops.TetrisEnvBranchless.start(
        envs,
        envs_per_thread,
        drop_speed,
        survival_reward,
        placed_block_reward,
        line_clear_reward,
        game_over_reward,
    )


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
