import random
import pygame

# Game configuration
CELL_SIZE = 25
GRID_WIDTH = 20
GRID_HEIGHT = 20

WIDTH = CELL_SIZE * GRID_WIDTH
HEIGHT = CELL_SIZE * GRID_HEIGHT

# Colors
BLACK = (20, 20, 20)
GREEN = (0, 255, 100)
RED = (255, 70, 70)
WHITE = (255, 255, 255)

# Directions
RIGHT = (1, 0)
LEFT = (-1, 0)
UP = (0, -1)
DOWN = (0, 1)

# Snake movement event
MOVE_EVENT = pygame.USEREVENT + 1


def spawn_food(snake):
    """Spawn food randomly in an unoccupied cell."""

    occupied = set(snake)

    available = [
        (x, y)
        for y in range(GRID_HEIGHT)
        for x in range(GRID_WIDTH)
        if (x, y) not in occupied
    ]

    if not available:
        return None

    return random.choice(available)


def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI Snake")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)

    # Starting snake
    snake = [(10, 10), (9, 10), (8, 10)]

    direction = RIGHT
    next_direction = RIGHT

    # Food and scoring
    food = spawn_food(snake)
    score = 0

    game_over = False
    won = False
    running = True

    # Move every 150 milliseconds
    pygame.time.set_timer(MOVE_EVENT, 150)

    while running:

        # Handle events
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            # Keyboard controls
            elif event.type == pygame.KEYDOWN:

                # Restart game
                if game_over and event.key == pygame.K_r:
                    snake = [(10, 10), (9, 10), (8, 10)]
                    direction = RIGHT
                    next_direction = RIGHT

                    food = spawn_food(snake)
                    score = 0

                    game_over = False
                    won = False

                elif not game_over:
                    keys = {
                        pygame.K_UP: UP,
                        pygame.K_DOWN: DOWN,
                        pygame.K_LEFT: LEFT,
                        pygame.K_RIGHT: RIGHT,
                    }

                    requested = keys.get(event.key)

                    if requested is not None:
                        opposite = (-direction[0], -direction[1])

                        # Prevent reversing directly into the body
                        if requested != opposite:
                            next_direction = requested

            # Move the snake
            elif event.type == MOVE_EVENT and not game_over:

                direction = next_direction

                head_x, head_y = snake[0]

                new_head = (
                    head_x + direction[0],
                    head_y + direction[1],
                )

                # Wall collision
                hit_wall = (
                    new_head[0] < 0
                    or new_head[0] >= GRID_WIDTH
                    or new_head[1] < 0
                    or new_head[1] >= GRID_HEIGHT
                )

                # Body collision
                hit_body = new_head in snake[:-1]

                if hit_wall or hit_body:
                    game_over = True

                else:
                    # Move head forward
                    snake.insert(0, new_head)

                    # Check whether snake ate food
                    if new_head == food:
                        score += 1

                        # Spawn new food
                        food = spawn_food(snake)

                        # Win if the board is full
                        if food is None:
                            game_over = True
                            won = True

                    else:
                        # Remove tail if no food was eaten
                        snake.pop()

        # Clear screen
        screen.fill(BLACK)

        # Draw snake
        for x, y in snake:
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
        if food is not None:
            pygame.draw.rect(
                screen,
                RED,
                (
                    food[0] * CELL_SIZE,
                    food[1] * CELL_SIZE,
                    CELL_SIZE - 1,
                    CELL_SIZE - 1,
                ),
            )

        # Draw score
        score_text = font.render(
            f"Score: {score}",
            True,
            WHITE,
        )

        screen.blit(score_text, (10, 10))

        # Game over message
        if game_over:

            message = "You Win! Press R" if won else "Game Over - Press R"

            text = font.render(
                message,
                True,
                WHITE,
            )

            rect = text.get_rect(
                center=(WIDTH // 2, HEIGHT // 2 - 60)
            )

            screen.blit(text, rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()