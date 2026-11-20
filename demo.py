"""AI Snake: focused, recruiter-friendly demo of the trained DQN agent.

Run: python demo.py [--model PATH]

Training and evaluation remain separate workflows; this window is intentionally
an inference showcase, not a training dashboard.
"""

import argparse

import pygame
import torch

from board_renderer import draw_board
from model import load_trained_model
from network_explorer import NetworkExplorer
from snake_env import SnakeEnv
from state_encoder import encode_state


SIZE = (1220, 748)
BOARD_CELLS = 23
BG = (8, 14, 23)
SURFACE = (16, 26, 38)
RAISED = (21, 35, 49)
STROKE = (39, 58, 72)
WHITE = (235, 246, 250)
SOFT = (160, 180, 192)
DIM = (103, 129, 144)
GREEN = (12, 236, 157)
PINK = (219, 114, 186)
YELLOW = (254, 211, 120)

ACTIONS = ("Straight", "Right", "Left")
SAMPLE_NEURONS = (0, 25, 51, 76, 102, 127)


def font(size, bold=False):
    # Initialized by pygame.init(); cache to avoid rebuilding fonts each frame.
    key = (size, bold)
    if not hasattr(font, "cache"):
        font.cache = {}
    if key not in font.cache:
        font.cache[key] = pygame.font.SysFont("Arial", size, bold=bold)
    return font.cache[key]


def write(screen, message, x, y, size=16, color=WHITE, bold=False):
    rendered = font(size, bold).render(str(message), True, color)
    screen.blit(rendered, (round(x), round(y)))
    return rendered


def right(screen, message, right_x, y, size=16, color=WHITE, bold=False):
    rendered = font(size, bold).render(str(message), True, color)
    screen.blit(rendered, (round(right_x - rendered.get_width()), round(y)))


def card(screen, rect, color=SURFACE, radius=18):
    pygame.draw.rect(screen, color, rect, border_radius=radius)
    pygame.draw.rect(screen, STROKE, rect, 1, border_radius=radius)


def button(screen, rect, title, *, active=False):
    rect = pygame.Rect(rect)
    hovered = rect.collidepoint(pygame.mouse.get_pos())
    fill = (22, 88, 75) if active else (35, 54, 66) if hovered else RAISED
    pygame.draw.rect(screen, fill, rect, border_radius=10)
    pygame.draw.rect(screen, (29, 171, 122) if active else STROKE,
                     rect, 1, border_radius=10)
    caption = font(14, True).render(title, True, WHITE)
    screen.blit(caption, caption.get_rect(center=rect.center))
    return rect


def _column_positions(y_top, y_bottom, count):
    return [round(y_top + i * (y_bottom - y_top) / max(1, count - 1))
            for i in range(count)]


def _node(screen, point, value, *, picked=False):
    """Use actual activation magnitude for node brightness, not decorative noise."""
    strength = abs(float(value)) / (1 + abs(float(value)))
    shade = tuple(round(a + (b - a) * strength)
                  for a, b in zip((46, 72, 86), GREEN))
    if picked:
        pygame.draw.circle(screen, (35, 104, 82), point, 14)
    pygame.draw.circle(screen, (10, 22, 32), point, 9)
    pygame.draw.circle(screen, shade, point, 6)


def _links(screen, left_x, left_y, left_idx, left_vals,
           right_x, right_y, right_idx, weights, *, count=1):
    """Draw only the most influential visible connections for legibility."""
    for pos, out_idx in enumerate(right_idx):
        if out_idx >= weights.shape[0]:
            continue
        candidates = []
        for src_pos, src_idx in enumerate(left_idx):
            if src_idx >= weights.shape[1]:
                continue
            contribution = (float(left_vals[src_pos]) *
                            float(weights[out_idx, src_idx]))
            candidates.append((abs(contribution), contribution, src_pos))
        candidates.sort(reverse=True)
        for magnitude, contribution, source in candidates[:count]:
            intensity = magnitude / (1 + magnitude)
            base = GREEN if contribution >= 0 else PINK
            shade = tuple(round(38 + (base[c] - 38) * (0.25 + 0.52 * intensity))
                          for c in range(3))
            pygame.draw.line(screen, shade, (left_x, left_y[source]),
                             (right_x, right_y[pos]), 1)


def draw_neural_activity(screen, model, inputs, hidden_1, hidden_2, q_values):
    """Small, accurate sample of 11→128→128→3 with a deliberately sparse graph."""
    write(screen, "NETWORK ACTIVITY", 578, 375, 12, DIM, True)
    right(screen, "11  /  128  /  128  /  3", 1162, 375, 13, SOFT)

    x0, x1, x2, x3 = (602, 756, 910, 1061)
    y0 = _column_positions(419, 590, 11)
    y1 = _column_positions(431, 578, 6)
    y2 = _column_positions(431, 578, 6)
    y3 = _column_positions(442, 567, 3)
    active = model is not None and all(v is not None for v in
                                       (inputs, hidden_1, hidden_2, q_values))
    iv = [float(v) for v in inputs] if active else [0.0] * 11
    hv1 = [float(hidden_1[i]) for i in SAMPLE_NEURONS] if active else [0.0] * 6
    hv2 = [float(hidden_2[i]) for i in SAMPLE_NEURONS] if active else [0.0] * 6
    qv = [float(v) for v in q_values] if active else [0.0] * 3

    if active:
        linear = [layer for layer in model.modules()
                  if isinstance(layer, torch.nn.Linear)]
        if len(linear) >= 3:
            weight = [layer.weight.detach().cpu() for layer in linear[:3]]
            _links(screen, x0, y0, range(11), iv,
                   x1, y1, SAMPLE_NEURONS, weight[0], count=1)
            _links(screen, x1, y1, SAMPLE_NEURONS, hv1,
                   x2, y2, SAMPLE_NEURONS, weight[1], count=2)
            _links(screen, x2, y2, SAMPLE_NEURONS, hv2,
                   x3, y3, range(3), weight[2], count=2)

    for ys, x, values in ((y0, x0, iv), (y1, x1, hv1),
                          (y2, x2, hv2), (y3, x3, qv)):
        for y, value in zip(ys, values):
            _node(screen, (x, y), value)

    for label, x in (("INPUT", x0), ("HIDDEN", x1),
                     ("HIDDEN", x2), ("OUTPUT", x3)):
        image = font(11, True).render(label, True, DIM)
        screen.blit(image, (x - image.get_width() // 2, 609))
    write(screen, "6 / 128 hidden neurons shown per layer", 578, 641, 12, DIM)


class SnakeShowcase:
    def __init__(self, model_path=None):
        pygame.init()
        self.screen = pygame.display.set_mode(SIZE)
        pygame.display.set_caption("Snake AI — Deep Reinforcement Learning")
        self.clock = pygame.time.Clock()
        self.model_error = None
        try:
            self.model = load_trained_model(model_path) if model_path else load_trained_model()
        except Exception as exc:
            # Keep the demo window usable and show a helpful error in the
            # terminal if the checkpoint is corrupt or has incompatible keys.
            self.model = None
            self.model_error = str(exc)
            print(f"Could not load trained checkpoint: {exc}", flush=True)
        self.explorer = NetworkExplorer(self.model)
        self.exploring = False
        self.env = SnakeEnv()
        self.previous = list(self.env.snake)
        self.last_move = pygame.time.get_ticks()
        self.interval = 135
        self.paused = False
        self.best = 0
        self.games = 1
        self.steps_since_food = 0
        self.death_time = None
        self.inputs = self.hidden_1 = self.hidden_2 = self.q_values = None
        self.action = None
        self.buttons = {}

    def restart(self, *, manual=False):
        self.env.reset()
        self.previous = list(self.env.snake)
        self.last_move = pygame.time.get_ticks()
        self.steps_since_food = 0
        self.death_time = None
        self.inputs = self.hidden_1 = self.hidden_2 = self.q_values = None
        self.action = None
        if manual:
            # Restart this run without throwing away the best score already
            # achieved during the showcase session.
            self.paused = False
        self.games += 1

    def activate(self, name):
        if name == "pause":
            self.paused = not self.paused
        elif name == "restart":
            self.restart(manual=True)
        elif name == "slower":
            self.interval = min(360, self.interval + 25)
        elif name == "faster":
            self.interval = max(45, self.interval - 25)
        elif name == "explore" and self.model is not None:
            self.exploring = True

    def update(self, now):
        if self.paused or self.model is None or now - self.last_move < self.interval:
            return
        self.last_move = now
        if self.env.done:
            if self.death_time is not None and now - self.death_time > 750:
                self.restart()
            return

        state = encode_state(self.env)
        with torch.no_grad():
            q, activations = self.model(state.unsqueeze(0), return_activations=True)
        self.action = int(q.argmax(dim=1).item())
        self.inputs = state.tolist()
        self.hidden_1 = activations["hidden_1"][0].tolist()
        self.hidden_2 = activations["hidden_2"][0].tolist()
        self.q_values = q[0].tolist()

        self.previous = list(self.env.snake)
        _, reward, ended = self.env.step(self.action)
        self.best = max(self.best, self.env.score)
        self.steps_since_food = 0 if reward == 10.0 else self.steps_since_food + 1
        if not ended and self.steps_since_food >= 100 * len(self.env.snake):
            self.env.done = True
        if self.env.done:
            self.death_time = now

    def draw(self, now):
        screen = self.screen
        if self.exploring:
            self.explorer.draw(screen, self.inputs, self.hidden_1,
                               self.hidden_2, self.q_values)
            pygame.display.flip()
            return
        screen.fill(BG)
        write(screen, "SNAKE", 37, 24, 30, WHITE, True)
        write(screen, " / AI", 146, 24, 30, GREEN, True)
        write(screen, "Watch a trained neural network play Snake.", 39, 68, 15, SOFT)
        right(screen, "DEEP Q-LEARNING", 1185, 36, 13, SOFT, True)

        card(screen, (34, 131, 500, 536))
        write(screen, "LIVE GAME", 54, 152, 12, SOFT, True)
        right(screen, f"SCORE  {self.env.score:02d}     BEST  {self.best:02d}",
              515, 150, 15, GREEN, True)
        board = pygame.Surface((460, 460))
        fraction = max(0.0, min(1.0, (now - self.last_move) / self.interval))
        draw_board(board, self.env, self.previous, fraction,
                   cell_size=BOARD_CELLS, show_hud=False, mode="NEURAL AI",
                   paused=self.paused)
        screen.blit(board, (54, 183))

        card(screen, (554, 131, 632, 536))
        write(screen, "LAST AI DECISION", 576, 151, 12, SOFT, True)
        if self.model is None:
            write(screen, "Could not load model" if self.model_error else
                  "No trained model found", 576, 207, 29, YELLOW, True)
            write(screen, "Check the terminal for details" if self.model_error else
                  "Place your weights at models/snake_dqn.pt", 576, 253, 16, SOFT)
        elif self.action is None:
            write(screen, "Thinking...", 576, 205, 33, WHITE, True)
        else:
            write(screen, f"{ACTIONS[self.action]}", 576, 202, 38, GREEN, True)
            write(screen, "Most recent move · highest Q-value", 577, 257, 13, SOFT)

        if self.model is not None:
            for index, title in enumerate(ACTIONS):
                x = 839 + index * 108
                active = self.action == index
                pygame.draw.rect(screen, (24, 74, 64) if active else RAISED,
                                 (x, 199, 100, 77), border_radius=12)
                pygame.draw.rect(screen, (35, 155, 112) if active else STROKE,
                                 (x, 199, 100, 77), 1, border_radius=12)
                write(screen, title.upper(), x + 11, 212, 11,
                      GREEN if active else SOFT, True)
                if self.q_values is not None:
                    write(screen, f"{self.q_values[index]:+.2f}", x + 11, 234,
                          19, WHITE, True)
                else:
                    write(screen, "—", x + 11, 234, 19, DIM)

        pygame.draw.line(screen, STROKE, (576, 309), (1164, 309), 1)
        write(screen, "REAL-TIME FORWARD PASS", 578, 329, 12, SOFT, True)
        explore_button = button(screen, (970, 320, 190, 33),
                                "E · Explore network")
        draw_neural_activity(screen, self.model, self.inputs,
                             self.hidden_1, self.hidden_2, self.q_values)

        card(screen, (34, 683, 1152, 49), color=SURFACE, radius=12)
        pygame.draw.circle(screen, GREEN if not self.paused and self.model else YELLOW,
                           (56, 707), 5)
        mode = "PLAYING" if not self.paused and self.model else "PAUSED" if self.paused else "NO MODEL"
        write(screen, mode, 71, 698, 13, WHITE, True)
        write(screen, f"GAME {self.games}", 161, 698, 13, DIM)
        write(screen, "Space · pause      R · restart      E · network", 310, 699, 12, DIM)
        write(screen, f"{1000 / self.interval:.1f} moves/s", 738, 698, 13, SOFT)
        self.buttons = {
            "explore": explore_button,
            "pause": button(screen, (854, 689, 104, 37),
                            "Resume" if self.paused else "Pause", active=self.paused),
            "restart": button(screen, (968, 689, 104, 37), "Restart"),
            "slower": button(screen, (1082, 689, 45, 37), "−"),
            "faster": button(screen, (1137, 689, 44, 37), "+"),
        }
        pygame.display.flip()

    def run(self):
        running = True
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if self.exploring:
                            if self.explorer.click(event.pos) == "back":
                                self.exploring = False
                        else:
                            for action, rect in self.buttons.items():
                                if rect.collidepoint(event.pos):
                                    self.activate(action)
                                    break
                    elif event.type == pygame.KEYDOWN:
                        if self.exploring and event.key in (pygame.K_ESCAPE, pygame.K_e):
                            self.exploring = False
                        elif event.key == pygame.K_e:
                            self.activate("explore")
                        elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                            running = False
                        elif event.key == pygame.K_SPACE and not self.exploring:
                            self.activate("pause")
                        elif event.key == pygame.K_r and not self.exploring:
                            self.activate("restart")
                        elif not self.exploring and event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                            self.activate("faster")
                        elif not self.exploring and event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                            self.activate("slower")
                now = pygame.time.get_ticks()
                self.update(now)
                self.draw(now)
                self.clock.tick(30)
        finally:
            pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch the Snake DQN play")
    parser.add_argument("--model", type=str,
                        help="Optional checkpoint path (default: bundled model)")
    args = parser.parse_args()
    SnakeShowcase(model_path=args.model).run()
