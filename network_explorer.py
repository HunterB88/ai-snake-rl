"""Full, interactive visualization of the actual 11→128→128→3 Snake DQN.

Every neuron is drawn. The brightest animated edges are the strongest real
weighted contributions in the current forward pass. Drawing all 18,176
weights is available as an optional, deliberately dense overlay.
"""

import heapq
import math

import pygame


SIZE = (1220, 748)
BG = (8, 14, 23)
PANEL = (15, 27, 40)
RAISED = (23, 42, 57)
STROKE = (40, 64, 79)
WHITE = (238, 248, 251)
SOFT = (161, 185, 199)
DIM = (106, 133, 152)
GREEN = (8, 239, 162)
CYAN = (53, 184, 218)
PINK = (236, 100, 179)
GOLD = (255, 213, 119)
GRAPH = pygame.Rect(35, 142, 1150, 485)

INPUT_NAMES = (
    "Danger ahead", "Danger right", "Danger left", "Moving up",
    "Moving down", "Moving left", "Moving right", "Food above",
    "Food below", "Food left", "Food right",
)

ACTION_NAMES = ("STRAIGHT", "RIGHT", "LEFT")
LAYER_NAMES = ("Input", "Hidden 1", "Hidden 2", "Output")
COUNTS = (11, 128, 128, 3)


def _font(size, bold=False):
    cache = getattr(_font, "cache", {})
    key = (size, bold)

    if key not in cache:
        cache[key] = pygame.font.SysFont("Arial", size, bold=bold)
        _font.cache = cache

    return cache[key]


def _text(screen, value, pos, size=15, color=WHITE, bold=False):
    image = _font(size, bold).render(str(value), True, color)
    screen.blit(image, pos)
    return image


def _button(screen, rect, label, *, active=False):
    hover = (
        pygame.display.get_init()
        and rect.collidepoint(pygame.mouse.get_pos())
    )

    fill = (
        (26, 85, 69) if active
        else (34, 58, 73) if hover
        else RAISED
    )

    pygame.draw.rect(screen, fill, rect, border_radius=11)
    pygame.draw.rect(
        screen,
        GREEN if active else STROKE,
        rect,
        1,
        border_radius=11,
    )

    image = _font(14, True).render(
        label,
        True,
        GREEN if active else WHITE,
    )

    screen.blit(image, image.get_rect(center=rect.center))


def _hidden_points(cx, cy):
    """Four elliptical rings: 16 + 24 + 40 + 48 = 128 visible units."""

    points = []

    rings = (
        (16, 35, 53),
        (24, 58, 88),
        (40, 84, 127),
        (48, 110, 170),
    )

    for ring, (count, radius_x, radius_y) in enumerate(rings):
        for i in range(count):
            angle = 2 * math.pi * (
                i / count + (ring % 2) / (2 * count)
            )

            points.append(
                (
                    round(cx + radius_x * math.cos(angle)),
                    round(cy + radius_y * math.sin(angle)),
                )
            )

    return points


class NetworkExplorer:
    """An inspectable neuron constellation powered by real model weights."""

    def __init__(self, model):
        self.weights = None
        self.biases = None

        if model is not None:
            layers = (
                model.input_layer,
                model.hidden_layer,
                model.output_layer,
            )

            self.weights = [
                layer.weight.detach().cpu().tolist()
                for layer in layers
            ]

            self.biases = [
                layer.bias.detach().cpu().tolist()
                for layer in layers
            ]

        self.selected = None
        self.show_all = False
        self.dragging = False
        self.buttons = {}

        self._positions = [
            [(215, 220 + 32 * i) for i in range(11)],
            _hidden_points(457, 381),
            _hidden_points(760, 381),
            [(1046, 273), (1046, 386), (1046, 499)],
        ]

        self._cached_state = None
        self._active_edges = None
        self._all_weights_surface = None

    def positions(self):
        """Return every neuron position in exact model-index order."""
        return self._positions

    def activations(self, inputs, hidden_1, hidden_2, q_values):
        arrays = (inputs, hidden_1, hidden_2, q_values)

        return [
            list(map(float, a))
            if a is not None
            else [0.0] * count
            for a, count in zip(arrays, COUNTS)
        ]

    def _selection(self, values):
        if self.selected is not None:
            return self.selected

        return 3, max(
            range(3),
            key=lambda i: values[3][i],
        )

    def _edge_candidates(self, values):
        """Calculate the strongest real weighted contributions."""

        key = tuple(
            value
            for layer in values
            for value in layer
        )

        if (
            key == self._cached_state
            and self._active_edges is not None
        ):
            return self._active_edges

        picked = []

        for layer, limit in enumerate((36, 120, 36)):
            candidates = []

            for dst_index, row in enumerate(self.weights[layer]):
                for src_index, weight in enumerate(row):
                    signal = values[layer][src_index] * weight

                    if abs(signal) > 1e-9:
                        candidates.append(
                            (
                                abs(signal),
                                signal,
                                layer,
                                src_index,
                                dst_index,
                            )
                        )

            picked.extend(
                heapq.nlargest(
                    limit,
                    candidates,
                    key=lambda item: item[0],
                )
            )

        self._active_edges = picked
        self._cached_state = key

        return picked

    def _all_connections(self):
        if self._all_weights_surface is not None:
            return self._all_weights_surface

        ghost = pygame.Surface(SIZE, pygame.SRCALPHA)
        ghost.set_clip(GRAPH)

        for layer, matrix in enumerate(self.weights):
            for destination, row in enumerate(matrix):
                end = self._positions[layer + 1][destination]

                for source, weight in enumerate(row):
                    color = GREEN if weight >= 0 else PINK

                    pygame.draw.line(
                        ghost,
                        (*color, 7),
                        self._positions[layer][source],
                        end,
                        1,
                    )

        ghost.set_clip(None)
        self._all_weights_surface = ghost

        return ghost

    def _draw_background(self, screen):
        screen.fill(BG)

        pygame.draw.rect(
            screen,
            PANEL,
            GRAPH,
            border_radius=21,
        )

        pygame.draw.rect(
            screen,
            STROKE,
            GRAPH,
            1,
            border_radius=21,
        )

        glow = pygame.Surface(SIZE, pygame.SRCALPHA)

        for center_x, color in (
            (457, GREEN),
            (760, CYAN),
        ):
            pygame.draw.ellipse(
                glow,
                (*color, 9),
                (center_x - 138, 175, 276, 414),
            )

            for radius_x, radius_y in (
                (38, 57),
                (61, 92),
                (87, 131),
                (116, 177),
            ):
                pygame.draw.ellipse(
                    glow,
                    (*color, 25),
                    (
                        center_x - radius_x,
                        381 - radius_y,
                        2 * radius_x,
                        2 * radius_y,
                    ),
                    1,
                )

        screen.blit(glow, (0, 0))

    def _draw_activity(self, screen, values):
        if self.show_all:
            screen.blit(self._all_connections(), (0, 0))

        edges = self._edge_candidates(values)

        overlay = pygame.Surface(SIZE, pygame.SRCALPHA)
        overlay.set_clip(GRAPH)

        greatest = max(
            (edge[0] for edge in edges),
            default=1.0,
        )

        for magnitude, signal, layer, src, dst in edges:
            base = GREEN if signal >= 0 else PINK
            alpha = round(
                48 + 123 * magnitude / greatest
            )

            pygame.draw.line(
                overlay,
                (*base, alpha),
                self._positions[layer][src],
                self._positions[layer + 1][dst],
                1,
            )

        # Highlight real connections into and out of selected neuron.
        layer, index = self._selection(values)
        selected_edges = []

        if layer > 0:
            for src, weight in enumerate(
                self.weights[layer - 1][index]
            ):
                signal = weight * values[layer - 1][src]

                selected_edges.append(
                    (
                        abs(signal),
                        signal,
                        self._positions[layer - 1][src],
                        self._positions[layer][index],
                    )
                )

        if layer < 3:
            for dst, row in enumerate(self.weights[layer]):
                signal = row[index] * values[layer][index]

                selected_edges.append(
                    (
                        abs(signal),
                        signal,
                        self._positions[layer][index],
                        self._positions[layer + 1][dst],
                    )
                )

        strongest = heapq.nlargest(
            18,
            selected_edges,
            key=lambda edge: edge[0],
        )

        for magnitude, signal, start, end in strongest:
            if magnitude > 1e-9:
                rgb = GREEN if signal >= 0 else PINK

                pygame.draw.line(
                    overlay,
                    (*rgb, 245),
                    start,
                    end,
                    2,
                )

        overlay.set_clip(None)
        screen.blit(overlay, (0, 0))

        # Animate signal propagation along actual model connections.
        time_s = pygame.time.get_ticks() / 1000.0

        strongest_active = sorted(
            edges,
            reverse=True,
            key=lambda item: item[0],
        )[:24]

        for order, edge in enumerate(strongest_active):
            _, signal, layer, src, dst = edge

            start = self._positions[layer][src]
            end = self._positions[layer + 1][dst]

            phase = (
                time_s * 0.42 + order * 0.067
            ) % 1.0

            point = (
                round(
                    start[0] + (end[0] - start[0]) * phase
                ),
                round(
                    start[1] + (end[1] - start[1]) * phase
                ),
            )

            pygame.draw.circle(
                screen,
                GREEN if signal >= 0 else PINK,
                point,
                2,
            )

    def _draw_nodes(self, screen, values):
        selected = self._selection(values)

        action = max(
            range(3),
            key=lambda i: values[3][i],
        )

        for layer, column in enumerate(self._positions):
            peak = max(
                (abs(v) for v in values[layer]),
                default=0.0,
            )

            for index, center in enumerate(column):
                val = values[layer][index]

                intensity = (
                    abs(val) / peak if peak else 0.0
                )

                base = GREEN if val >= 0 else PINK

                shade = tuple(
                    round(a + (b - a) * intensity)
                    for a, b in zip(
                        (55, 86, 100),
                        base,
                    )
                )

                radius = 5 if layer in (1, 2) else 7

                if selected == (layer, index):
                    pygame.draw.circle(
                        screen,
                        GOLD,
                        center,
                        radius + 8,
                        2,
                    )

                    pygame.draw.circle(
                        screen,
                        GOLD,
                        center,
                        radius + 3,
                        1,
                    )

                elif layer == 3 and index == action:
                    pygame.draw.circle(
                        screen,
                        GREEN,
                        center,
                        radius + 7,
                        2,
                    )

                pygame.draw.circle(
                    screen,
                    (9, 25, 34),
                    center,
                    radius + 1,
                )

                pygame.draw.circle(
                    screen,
                    shade,
                    center,
                    radius,
                )

    def _draw_labels(self, screen, values):
        headings = (
            (215, "INPUT · 11"),
            (457, "HIDDEN 1 · 128"),
            (760, "HIDDEN 2 · 128"),
            (1046, "ACTIONS · 3"),
        )

        for x, label in headings:
            rendered = _font(14, True).render(
                label,
                True,
                SOFT,
            )

            screen.blit(
                rendered,
                (
                    x - rendered.get_width() // 2,
                    169,
                ),
            )

        for layer, cx in (
            (1, 457),
            (2, 760),
        ):
            count = sum(
                value > 0
                for value in values[layer]
            )

            active = f"{count} ACTIVE / 128"

            rendered = _font(12, True).render(
                active,
                True,
                GREEN if layer == 1 else CYAN,
            )

            screen.blit(
                rendered,
                (
                    cx - rendered.get_width() // 2,
                    190,
                ),
            )

        for i, name in enumerate(INPUT_NAMES):
            x, y = self._positions[0][i]

            label = _font(12).render(
                name,
                True,
                WHITE if values[0][i] else DIM,
            )

            screen.blit(
                label,
                (
                    x - 22 - label.get_width(),
                    y - 9,
                ),
            )

        greedy = max(
            range(3),
            key=lambda i: values[3][i],
        )

        for i, name in enumerate(ACTION_NAMES):
            x, y = self._positions[3][i]

            _text(
                screen,
                name,
                (x + 22, y - 17),
                13,
                GREEN if i == greedy else SOFT,
                True,
            )

            _text(
                screen,
                f"{values[3][i]:+.2f}",
                (x + 22, y + 2),
                15,
                WHITE,
                True,
            )

    def _draw_inspector(self, screen, values):
        layer, index = self._selection(values)

        pygame.draw.rect(
            screen,
            PANEL,
            (35, 640, 1150, 64),
            border_radius=14,
        )

        pygame.draw.rect(
            screen,
            STROKE,
            (35, 640, 1150, 64),
            1,
            border_radius=14,
        )

        _text(
            screen,
            f"{LAYER_NAMES[layer].upper()} / NEURON {index + 1}",
            (53, 650),
            15,
            GOLD,
            True,
        )

        _text(
            screen,
            f"Activation  {values[layer][index]:+.4f}",
            (339, 650),
            15,
            WHITE,
        )

        if layer:
            _text(
                screen,
                f"Bias  {self.biases[layer - 1][index]:+.4f}",
                (601, 650),
                15,
                SOFT,
            )

        _text(
            screen,
            "Click any neuron to reveal its strongest connections",
            (53, 679),
            13,
            DIM,
        )

        _text(
            screen,
            "GREEN  positive",
            (865, 678),
            12,
            GREEN,
        )

        _text(
            screen,
            "PINK  negative",
            (1030, 678),
            12,
            PINK,
        )

    def draw(self, screen, inputs, hidden_1, hidden_2, q_values):
        self._draw_background(screen)

        _text(
            screen,
            "FULL NEURAL NETWORK",
            (36, 24),
            29,
            GREEN,
            True,
        )

        _text(
            screen,
            "270 real neurons · 18,176 learned connections · live DQN activity",
            (38, 67),
            15,
            SOFT,
        )

        self.buttons = {
            "all": pygame.Rect(826, 27, 178, 39),
            "back": pygame.Rect(1019, 27, 166, 39),
        }

        _button(
            screen,
            self.buttons["all"],
            "Hide all weights" if self.show_all else "Show all weights",
            active=self.show_all,
        )

        _button(
            screen,
            self.buttons["back"],
            "← Back to demo",
        )

        _text(
            screen,
            "ACTIVE SIGNALS",
            (57, 112),
            12,
            GREEN,
            True,
        )

        _text(
            screen,
            "Brighter links = larger weighted contributions",
            (202, 112),
            12,
            DIM,
        )

        if self.weights is None:
            _text(
                screen,
                "Load a trained checkpoint to inspect the network.",
                (330, 370),
                21,
                SOFT,
            )
            return

        values = self.activations(
            inputs,
            hidden_1,
            hidden_2,
            q_values,
        )

        self._draw_activity(screen, values)
        self._draw_nodes(screen, values)
        self._draw_labels(screen, values)
        self._draw_inspector(screen, values)

        _text(
            screen,
            "E or Esc · return to demo     Click · inspect any neuron",
            (42, 720),
            13,
            DIM,
        )

        _text(
            screen,
            "Live model weights and activations — no simulated connections",
            (796, 720),
            12,
            DIM,
        )

    def click(self, point):
        for key, button in self.buttons.items():
            if button.collidepoint(point):
                if key == "all":
                    self.show_all = not self.show_all

                return key

        nearby = min(
            (
                (point[0] - x) ** 2 + (point[1] - y) ** 2,
                layer,
                index,
            )
            for layer, column in enumerate(self._positions)
            for index, (x, y) in enumerate(column)
        )

        if nearby[0] <= 12 ** 2:
            self.selected = (
                nearby[1],
                nearby[2],
            )

        return None

    def drag(self, _position):
        """Compatibility with older demo mouse handlers."""
        pass

    def wheel(self, _delta):
        """All 270 neurons already fit on the screen."""
        pass