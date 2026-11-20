import random
import pygame

from snake_env import SnakeEnv


# Game configuration
CELL_SIZE = 25
GRID_WIDTH = 20
GRID_HEIGHT = 20

BOARD_WIDTH = CELL_SIZE * GRID_WIDTH
BOARD_HEIGHT = CELL_SIZE * GRID_HEIGHT

PANEL_WIDTH = 260

WIDTH = BOARD_WIDTH + PANEL_WIDTH
HEIGHT = BOARD_HEIGHT

# Colors
BLACK = (20, 20, 20)
GREEN = (0, 255, 100)
RED = (255, 70, 70)
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
PANEL_BG = (30, 32, 45)

MOVE_EVENT = pygame.USEREVENT + 1


def arrow_to_action(env, key):
    """Convert an arrow key into a relative movement action."""

    keys = {
        pygame.K_RIGHT: (1, 0),
        pygame.K_DOWN: (0, 1),
        pygame.K_LEFT: (-1, 0),
        pygame.K_UP: (0, -1),
    }

    desired_direction = keys.get(key)

    if desired_direction is None:
        return None

    desired_index = env.DIRECTIONS.index(desired_direction)

    turn = (desired_index - env.direction_index) % 4

    # 0 = straight, 1 = right, 2 = left
    actions = {
        0: 0,
        1: 1,
        3: 2,
    }

    return actions.get(turn)


def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI Snake Training Lab")

    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont(None, 32)
    font = pygame.font.SysFont(None, 25)

    # Create our game environment
    env = SnakeEnv(
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
    )

    rng = random.Random()

    mode = "HUMAN"

    pending_action = 0
    episodes = 0
    best_score = 0
    last_reward = 0.0

    running = True

    pygame.time.set_timer(MOVE_EVENT, 150)

    while running:

        # Handle events
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                # Human mode
                if event.key == pygame.K_h:
                    mode = "HUMAN"
                    env.reset()
                    pending_action = 0
                    episodes = 0

                # Random AI mode
                elif event.key == pygame.K_a:
                    mode = "RANDOM AI"
                    env.reset()
                    pending_action = 0
                    episodes = 0

                # Restart
                elif event.key == pygame.K_r:
                    env.reset()
                    pending_action = 0

                # Human controls
                elif mode == "HUMAN" and not env.done:

                    action = arrow_to_action(env, event.key)

                    if action is not None:
                        pending_action = action

            # Advance game
            elif event.type == MOVE_EVENT:

                if mode == "RANDOM AI":

                    # Automatically restart after death
                    if env.done:
                        episodes += 1
                        env.reset()

                    else:
                        action = rng.randrange(3)

                        state, reward, done = env.step(action)

                        last_reward = reward

                elif not env.done:

                    state, reward, done = env.step(
                        pending_action
                    )

                    last_reward = reward
                    pending_action = 0

                best_score = max(best_score, env.score)

        # Clear screen
        screen.fill(BLACK)

        # Draw snake
        for x, y in env.snake:
            pygame.draw.rect(
                screen,
                GREEN,
                (
                    x * CELL_SIZE,
                    y * CELL_SIZE,
                    CELL_SIZE - 1,
                    CELL_SIZE - 1,
                ),
            )

        # Draw food
        if env.food is not None:
            pygame.draw.rect(
                screen,
                RED,
                (
                    env.food[0] * CELL_SIZE,
                    env.food[1] * CELL_SIZE,
                    CELL_SIZE - 1,
                    CELL_SIZE - 1,
                ),
            )

        # Draw dashboard panel
        pygame.draw.rect(
            screen,
            PANEL_BG,
            (BOARD_WIDTH, 0, PANEL_WIDTH, HEIGHT),
        )

        panel_x = BOARD_WIDTH + 20

        # Title
        title = title_font.render(
            "AI SNAKE LAB",
            True,
            GREEN,
        )

        screen.blit(title, (panel_x, 25))

        # Dashboard information
        stats = [
            f"Mode: {mode}",
            "",
            f"Score: {env.score}",
            f"Best Score: {best_score}",
            f"Episodes: {episodes}",
            "",
            f"Last Reward: {last_reward:.2f}",
            "",
            "Neural Network:",
            "Not connected yet",
            "",
            "CONTROLS",
            "H - Human Mode",
            "A - Random AI",
            "R - Restart",
            "Arrows - Movement",
        ]

        y = 90

        for line in stats:
            text = font.render(line, True, WHITE)

            screen.blit(text, (panel_x, y))

            y += 25

        # Display game over
        if env.done and mode == "HUMAN":

            message = font.render(
                "Game Over - Press R",
                True,
                WHITE,
            )

            rect = message.get_rect(
                center=(
                    BOARD_WIDTH // 2,
                    BOARD_HEIGHT // 2,
                )
            )

            screen.blit(message, rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.time.set_timer(MOVE_EVENT, 0)
    pygame.quit()


if __name__ == "__main__":
    main()