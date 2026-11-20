"""Pygame-only rendering for the Snake environment."""

import math

import pygame


BG_A = (9, 19, 28)
BG_B = (11, 22, 32)
GRID_LINE = (20, 35, 45)
TEXT = (231, 246, 250)
MUTED = (140, 166, 177)
NEON = (0, 245, 158)


def _pill(screen, rect, alpha=200):
    surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.rect(
        surface,
        (6, 14, 24, alpha),
        surface.get_rect(),
        border_radius=11,
    )
    screen.blit(surface, rect.topleft)


def _draw_food(screen, food, cell_size):
    if food is None:
        return

    cx = int((food[0] + 0.5) * cell_size)
    cy = int((food[1] + 0.5) * cell_size)
    pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2

    glow_size = cell_size * 4
    glow = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
    center = glow_size // 2

    pygame.draw.circle(glow, (255, 65, 95, 18), (center, center), int(28 + 4 * pulse))
    pygame.draw.circle(glow, (255, 65, 95, 38), (center, center), int(19 + 3 * pulse))
    screen.blit(glow, (cx - center, cy - center))

    pygame.draw.circle(screen, (255, 70, 92), (cx, cy), int(8 + pulse))
    pygame.draw.circle(screen, (255, 180, 185), (cx - 3, cy - 3), 2)


def _draw_snake(screen, env, previous_snake, progress, cell_size):
    snake = env.snake
    if not snake:
        return

    prior = previous_snake or snake
    progress = max(0.0, min(1.0, progress))

    def interpolated(index):
        target_x, target_y = snake[index]
        old_x, old_y = prior[index] if index < len(prior) else prior[-1]
        return (
            old_x + (target_x - old_x) * progress,
            old_y + (target_y - old_y) * progress,
        )

    # Tail first, so the head is always drawn on top.
    for index in range(len(snake) - 1, -1, -1):
        x, y = interpolated(index)
        fade = index / max(1, len(snake) - 1)

        if index == 0:
            color = (14, 255, 170)
        else:
            color = (0, int(230 - 76 * fade), int(145 - 55 * fade))

        rect = pygame.Rect(
            round(x * cell_size) + 2,
            round(y * cell_size) + 2,
            cell_size - 4,
            cell_size - 4,
        )
        pygame.draw.rect(screen, color, rect, border_radius=6)

    # Two eyes make it easy to see which segment is the head.
    head_x, head_y = interpolated(0)
    center_x = round((head_x + 0.5) * cell_size)
    center_y = round((head_y + 0.5) * cell_size)
    dx, dy = env.DIRECTIONS[env.direction_index]

    if dx:
        eyes = ((dx * 5, -5), (dx * 5, 5))
    else:
        eyes = ((-5, dy * 5), (5, dy * 5))

    for ox, oy in eyes:
        pygame.draw.circle(screen, (238, 255, 250), (center_x + ox, center_y + oy), 3)
        pygame.draw.circle(screen, (10, 45, 39), (center_x + ox + dx, center_y + oy + dy), 1)


def draw_board(
    screen,
    env,
    previous_snake,
    progress,
    *,
    cell_size=25,
    mode="HUMAN",
    paused=False,
    move_interval_ms=130,
    show_hud=True,
):
    """Render a polished board without modifying game state."""

    width = env.width * cell_size
    height = env.height * cell_size

    # Soft checkerboard plus subtle grid lines.
    for y in range(env.height):
        for x in range(env.width):
            rect = (x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, BG_A if (x + y) % 2 else BG_B, rect)

    for x in range(0, width + 1, cell_size):
        pygame.draw.line(screen, GRID_LINE, (x, 0), (x, height), 1)
    for y in range(0, height + 1, cell_size):
        pygame.draw.line(screen, GRID_LINE, (0, y), (width, y), 1)

    _draw_food(screen, env.food, cell_size)
    _draw_snake(screen, env, previous_snake, progress, cell_size)

    small_font = pygame.font.SysFont(None, 19)

    if show_hud:
        title_font = pygame.font.SysFont(None, 23, bold=True)

        top = pygame.Rect(12, 12, width - 24, 40)
        _pill(screen, top)
        screen.blit(title_font.render("SNAKE  /  LIVE", True, NEON), (25, 23))

        mode_label = small_font.render(mode, True, TEXT)
        screen.blit(mode_label, (width - 25 - mode_label.get_width(), 25))

        bottom = pygame.Rect(12, height - 48, width - 24, 36)
        _pill(screen, bottom)
        screen.blit(
            small_font.render("SPACE pause     + faster     - slower     R restart", True, TEXT),
            (22, height - 36),
        )

        speed_label = small_font.render(f"{1000 / move_interval_ms:.1f} moves/s", True, MUTED)
        screen.blit(speed_label, (width - speed_label.get_width() - 22, height - 36))

    if paused or (env.done and mode == "HUMAN"):
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((3, 8, 18, 145))
        screen.blit(overlay, (0, 0))

        headline = "PAUSED" if paused else "GAME OVER"
        subtitle = "Space to resume" if paused else "Press R to restart"
        headline_font = pygame.font.SysFont(None, 60, bold=True)
        headline_image = headline_font.render(headline, True, NEON)
        subtitle_image = small_font.render(subtitle, True, TEXT)

        screen.blit(headline_image, headline_image.get_rect(center=(width // 2, height // 2 - 10)))
        screen.blit(subtitle_image, subtitle_image.get_rect(center=(width // 2, height // 2 + 35)))
