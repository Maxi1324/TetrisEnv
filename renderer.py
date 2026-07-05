from __future__ import annotations

import argparse
import time

import pygame
import torch

import tetris_env


ACTION_LEFT = 0
ACTION_ROTATE = 1
ACTION_RIGHT = 2
ACTION_IDLE = 4


def draw_board(
    screen: pygame.Surface,
    board: list[float],
    reward: float,
    done: bool,
    stats: list[float],
    cell_size: int,
) -> None:
    width = 10 * cell_size
    height = 20 * cell_size
    screen.fill((18, 20, 24))

    board_rect = pygame.Rect(20, 20, width, height)
    pygame.draw.rect(screen, (42, 46, 54), board_rect)

    for y in range(20):
        for x in range(10):
            value = board[y * 10 + x]
            rect = pygame.Rect(20 + x * cell_size, 20 + y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (31, 35, 42), rect, 1)
            if value > 0.5:
                inner = rect.inflate(-4, -4)
                pygame.draw.rect(screen, (77, 179, 255), inner, border_radius=3)

    font = pygame.font.Font(None, 24)
    lines = [
        f"reward: {reward:.1f}",
        f"done: {done}",
        f"avg len: {stats[0]:.2f}",
        f"avg rows: {stats[1]:.2f}",
        "left/right: move",
        "up: rotate",
        "esc: quit",
    ]
    x = 40 + width
    for i, text in enumerate(lines):
        color = (235, 238, 244) if i < 4 else (156, 164, 176)
        screen.blit(font.render(text, True, color), (x, 24 + i * 28))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop-speed", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--cell-size", type=int, default=28)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    pygame.init()
    screen = pygame.display.set_mode((20 + 10 * args.cell_size + 220, 40 + 20 * args.cell_size))
    pygame.display.set_caption("TetrisEnv CUDA Renderer")
    clock = pygame.time.Clock()

    envs = 1
    tetris_env.start(envs, envs_per_thread=1, drop_speed=args.drop_speed)
    actions = torch.empty((envs,), device="cuda", dtype=torch.uint32)
    image = torch.empty((envs, 200), device="cuda", dtype=torch.float32)
    rewards = torch.empty((envs,), device="cuda", dtype=torch.float32)
    done = torch.empty((envs,), device="cuda", dtype=torch.bool)

    running = True
    while running:
        action = ACTION_IDLE
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                action = ACTION_LEFT
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                action = ACTION_ROTATE
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                action = ACTION_RIGHT

        actions.fill_(action)
        stats = tetris_env.step(actions, image, rewards, done, image_observation=True)
        torch.cuda.synchronize()

        draw_board(
            screen,
            image[0].detach().cpu().tolist(),
            float(rewards[0].detach().cpu()),
            bool(done[0].detach().cpu()),
            stats.detach().cpu().tolist(),
            args.cell_size,
        )
        pygame.display.flip()
        clock.tick(60)
        if args.sleep > 0:
            time.sleep(args.sleep)

    pygame.quit()


if __name__ == "__main__":
    main()
