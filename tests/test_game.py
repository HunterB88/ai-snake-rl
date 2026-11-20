from game import spawn_food, GRID_WIDTH, GRID_HEIGHT


def test_food_does_not_spawn_on_snake():
    snake = [
        (x, y)
        for y in range(GRID_HEIGHT)
        for x in range(GRID_WIDTH)
        if (x, y) != (0, 0)
    ]

    food = spawn_food(snake)

    assert food == (0, 0)


def test_no_food_when_board_is_full():
    snake = [
        (x, y)
        for y in range(GRID_HEIGHT)
        for x in range(GRID_WIDTH)
    ]

    food = spawn_food(snake)

    assert food is None