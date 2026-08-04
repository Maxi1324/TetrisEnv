from __future__ import annotations

import torch

from . import _C  # noqa: F401


def start(
    envs: int,
    envs_per_thread: int = 1,
    drop_speed: int = 1,
    survival_reward: float = 0.01,
    placed_block_reward: float = 0.1,
    clear_reward_1: float = 10.0,
    clear_reward_2: float = 20.0,
    clear_reward_3: float = 30.0,
    clear_reward_4: float = 40.0,
    game_over_reward: float = -1.0,
) -> None:
    torch.ops.TetrisEnvBranchless.start(
        envs,
        envs_per_thread,
        drop_speed,
        survival_reward,
        placed_block_reward,
        clear_reward_1,
        clear_reward_2,
        clear_reward_3,
        clear_reward_4,
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


def rolling_episode_data() -> tuple[torch.Tensor, int]:
    """Per-episode values of the most recent completed episodes for on-GPU quantile stats.

    Returns ``(data, valid)`` where ``data`` is an int32 CUDA tensor of shape
    ``[num_metrics, window]`` (metric-major; row order matches the METRIC_* indices below) and
    ``valid`` is how many leading columns are filled. Slice ``data[metric, :valid].float()`` and
    call ``torch.quantile`` on it.
    """
    return torch.ops.TetrisEnvBranchless.rolling_episode_data()


# Row indices into the tensor returned by rolling_episode_data() (must match the M_* macros in
# src/TetrisEnv.cu).
METRIC_LENGTH = 0
METRIC_ROWS = 1
METRIC_PIECES = 2
METRIC_HOLES = 3
METRIC_BUMPINESS = 4
METRIC_AGG_HEIGHT = 5
METRIC_MAX_HEIGHT = 6
METRIC_SINGLES = 7
METRIC_DOUBLES = 8
METRIC_TRIPLES = 9
METRIC_TETRIS = 10
