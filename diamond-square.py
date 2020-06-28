import argparse
from enum import IntEnum
import random
from typing import Callable, Dict, Tuple
from blessed import Terminal
import numpy as np
import math
import signal


class SquareNodeGraph:
    def __init__(self, size: int):

        if size < 3:
            raise ValueError('Must be of size 3 or more!')

        if size % 2 == 0:
            raise ValueError('Must have odd-numbered size!')

        # Initialise some data structures
        self.size = size
        self.nodes: Dict[str, float]  = {}  # Coordinates are screen-space. Top left is (0,0)

        # Populate node graph with zeros
        for x in range(self.size):
            for y in range(self.size):
                self.set_node(x, y, 0.0)
    
    def get_node(self, x: int, y: int) -> float:
        coordinate_hash = SquareNodeGraph.get_coordinate_hash(x, y)
        return self.nodes[coordinate_hash]
    
    def set_node(self, x: int, y: int, value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError('Node value must be within 0 and 1, inclusive!')

        coordinate_hash = SquareNodeGraph.get_coordinate_hash(x, y)
        self.nodes[coordinate_hash] = value

    @staticmethod
    def get_coordinate_hash(x: int, y: int) -> str:
        return f'{x},{y}'

    def __str__(self):
        total_output = ''

        for y in range(self.size):

            output_line = ''
            for x in range(self.size):
                output_line += str(self.get_node(x, y))[:3]
                output_line += '--' if x < self.size-1 else ''
            total_output += output_line

            if y < self.size-1:
                vertical_joins = ' '
                vertical_joins += '|    ' * self.size
                total_output += ('\n' + vertical_joins) * 2 + '\n'

        return total_output

    def as_nparray(self):
        arr = np.zeros((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                value = self.get_node(x, y)
                arr[y, x] = math.floor(value * 10)/10.0
        return arr


class Edge(IntEnum):
    Top = 1
    Left = 2
    Bottom = 4
    Right = 8


def clamp(a: float, n: float, b: float) -> float:
    return max(a, min(b, n))


def get_midpoint_from_tuples(first: [int, int], second: [int, int]) -> [int, int]:
    return [(first[0] + second[0]) >> 1, (first[1] + second[1]) >> 1]


def get_edge_coordinates_from_corners(corner_coordinates: Dict[Edge, Tuple[int, int]]) -> Dict[Edge, Tuple[int, int]]:
    return {
        Edge.Top: get_midpoint_from_tuples(
            corner_coordinates[Edge.Top | Edge.Left],
            corner_coordinates[Edge.Top | Edge.Right],
        ),
        Edge.Left: get_midpoint_from_tuples(
            corner_coordinates[Edge.Top | Edge.Left],
            corner_coordinates[Edge.Bottom | Edge.Left],
        ),
        Edge.Bottom: get_midpoint_from_tuples(
            corner_coordinates[Edge.Bottom | Edge.Left],
            corner_coordinates[Edge.Bottom | Edge.Right],
        ),
        Edge.Right: get_midpoint_from_tuples(
            corner_coordinates[Edge.Top | Edge.Right],
            corner_coordinates[Edge.Bottom | Edge.Right],
        ),
    }


def diamond_square(
    graph: SquareNodeGraph,
    corners: Dict[Edge, Tuple[int, int]],
    noise_scaling_factor: float = 1.0,
    noise_scaling_function: Callable = lambda f: f * 0.5
):

    # Diamond step
    midpoint = get_midpoint_from_tuples(
        corners[Edge.Top | Edge.Left],
        corners[Edge.Bottom | Edge.Right],
    )

    corner_values = [graph.get_node(c[0], c[1]) for c in corners.values()]
    mean_corner_value = sum(corner_values) / len(corner_values)
    midpoint_value = mean_corner_value + random.uniform(-1, 1) * noise_scaling_factor
    midpoint_value = float(clamp(0, midpoint_value, 1))

    graph.set_node(
        midpoint[0],
        midpoint[1],
        midpoint_value
    )

    # Square step
    edges = get_edge_coordinates_from_corners(corners)
    for edge, edge_coords in edges.items():

        x, y = edge_coords

        # Skip node if already set (e.g.: from an adjacent quad)
        if graph.get_node(x, y) != 0:
            continue

        adjacent_values: List[float] = [ midpoint_value ]
        for index, corner in enumerate(corners.keys()):
            is_adjacent_corner: bool = (edge & corner) != 0
            if is_adjacent_corner:
                adjacent_values.append(corner_values[index])

        mean_adjacent_value = sum(adjacent_values) / len(adjacent_values) 
        edge_value = mean_adjacent_value + random.uniform(-1, 1) * noise_scaling_factor
        edge_value = float(clamp(0, edge_value, 1))
        graph.set_node(x, y, edge_value)

    # Recursion break: do not recurse if we're at the most granular scale possible
    side_length = corners[Edge.Top | Edge.Right][0] - corners[Edge.Top | Edge.Left][0]
    if side_length <= 1:
        return

    # Define sub-quads for recursion in terms of their corner vertices
    subquads = [
        {
            Edge.Top | Edge.Left: corners[Edge.Top | Edge.Left],
            Edge.Top | Edge.Right: edges[Edge.Top],
            Edge.Bottom | Edge.Left: edges[Edge.Left],
            Edge.Bottom | Edge.Right: midpoint,
        },
        {
            Edge.Top | Edge.Left: edges[Edge.Top],
            Edge.Top | Edge.Right: corners[Edge.Top | Edge.Right],
            Edge.Bottom | Edge.Left: midpoint,
            Edge.Bottom | Edge.Right: edges[Edge.Right],
        },
        {
            Edge.Top | Edge.Left: edges[Edge.Left],
            Edge.Top | Edge.Right: midpoint,
            Edge.Bottom | Edge.Left: corners[Edge.Bottom | Edge.Left],
            Edge.Bottom | Edge.Right: edges[Edge.Bottom],
        },
        {
            Edge.Top | Edge.Left: midpoint,
            Edge.Top | Edge.Right: edges[Edge.Right],
            Edge.Bottom | Edge.Left: edges[Edge.Bottom],
            Edge.Bottom | Edge.Right: corners[Edge.Bottom | Edge.Right],
        },
    ]

    # Apply the diamond-square algorithm recursively to those sub-quads
    for subquad in subquads:
        diamond_square(
            graph,
            subquad,
            noise_scaling_function(noise_scaling_factor),
            noise_scaling_function
        )


def diamond_square_iterative(
    graph: SquareNodeGraph,
    noise_scaling_factor: float = 1.0,
    noise_scaling_function: Callable = lambda f: f * 0.5
):

    # Set up preliminary info
    quads = []
    quads.append({
        Edge.Top | Edge.Left: [0, 0],
        Edge.Top | Edge.Right: [graph.size-1, 0],
        Edge.Bottom | Edge.Left: [0, graph.size-1],
        Edge.Bottom | Edge.Right: [graph.size-1, graph.size-1],
    })
    latest_min_sidelength = graph.size

    while len(quads):
        corners = quads.pop()

        # Diamond step
        midpoint = get_midpoint_from_tuples(
            corners[Edge.Top | Edge.Left],
            corners[Edge.Bottom | Edge.Right],
        )

        corner_values = [graph.get_node(c[0], c[1]) for c in corners.values()]
        mean_corner_value = sum(corner_values) / len(corner_values)
        midpoint_value = mean_corner_value + random.uniform(-1, 1) * noise_scaling_factor
        midpoint_value = float(clamp(0, midpoint_value, 1))

        graph.set_node(
            midpoint[0],
            midpoint[1],
            midpoint_value
        )

        # Square step
        edges = get_edge_coordinates_from_corners(corners)
        for edge, edge_coords in edges.items():
            x, y = edge_coords
            if graph.get_node(x, y) != 0:
                continue
            adjacent_values: List[float] = [ midpoint_value ]
            for index, corner in enumerate(corners):
                is_relevant_corner: bool = (edge & corner) != 0
                if is_relevant_corner:
                    adjacent_values.append(corner_values[index])
            mean_adjacent_value = sum(adjacent_values) / len(adjacent_values) 
            edge_value = mean_adjacent_value + random.uniform(-1, 1) * noise_scaling_factor
            edge_value = float(clamp(0, edge_value, 1))
            graph.set_node(x, y, edge_value)

        # Recursion
        side_length = corners[Edge.Top | Edge.Right][0] - corners[Edge.Top | Edge.Left][0]
        if side_length < latest_min_sidelength:
            noise_scaling_factor = noise_scaling_function(noise_scaling_factor)
            latest_min_sidelength = side_length
        if side_length > 1:
            subquads = [
                {
                    Edge.Top | Edge.Left: corners[Edge.Top | Edge.Left],
                    Edge.Top | Edge.Right: edges[Edge.Top],
                    Edge.Bottom | Edge.Left: edges[Edge.Left],
                    Edge.Bottom | Edge.Right: midpoint,
                },
                {
                    Edge.Top | Edge.Left: edges[Edge.Top],
                    Edge.Top | Edge.Right: corners[Edge.Top | Edge.Right],
                    Edge.Bottom | Edge.Left: midpoint,
                    Edge.Bottom | Edge.Right: edges[Edge.Right],
                },
                {
                    Edge.Top | Edge.Left: edges[Edge.Left],
                    Edge.Top | Edge.Right: midpoint,
                    Edge.Bottom | Edge.Left: corners[Edge.Bottom | Edge.Left],
                    Edge.Bottom | Edge.Right: edges[Edge.Bottom],
                },
                {
                    Edge.Top | Edge.Left: midpoint,
                    Edge.Top | Edge.Right: edges[Edge.Right],
                    Edge.Bottom | Edge.Left: edges[Edge.Bottom],
                    Edge.Bottom | Edge.Right: corners[Edge.Bottom | Edge.Right],
                },
            ]
            quads.extend(subquads)


def run():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'size',
        type=int,
        help='The side length of the square node graph to generate',
        default=3,
    )
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['recursive', 'iterative'],
        default='recursive',
    )
    parser.add_argument(
        '--print', '-p',
        action='store_true',
    )

    args = parser.parse_args()
    size = int(args.size)

    # Set up the graph we'll be working on
    graph = SquareNodeGraph(size)

    # Initialise the corners
    graph.set_node(0, 0, random.uniform(0, 1))
    graph.set_node(0, size-1, random.uniform(0, 1))
    graph.set_node(size-1, 0, random.uniform(0, 1))
    graph.set_node(size-1, size-1, random.uniform(0, 1))

    # Conduct the actual algorithm
    if args.mode == 'recursive':

        diamond_square(
            graph=graph,
            corners={
                Edge.Top | Edge.Left: [0, 0],
                Edge.Top | Edge.Right: [size-1, 0],
                Edge.Bottom | Edge.Left: [0, size-1],
                Edge.Bottom | Edge.Right: [size-1, size-1],
            },
            noise_scaling_factor=1,
            noise_scaling_function=lambda f: f / 3
        )

    else:

        diamond_square_iterative(
            graph=graph,
            noise_scaling_factor=0.2,
        )

    # Output as NumPy array
    # array = grid.as_nparray()
    # ... do some plotting

    if args.print:
        # print(graph.as_nparray())
        term = Terminal()
        data = graph.as_nparray()

        with term.fullscreen(), term.cbreak():
            x = 0
            y = 0
            grid = Grid(x, y, data, term)

            grid.draw()

            signal.signal(signal.SIGWINCH, grid.on_resize)

            keypress = None
    
            while keypress != 'q':
                keypress = term.inkey()

                # print(keypress.code)

                if keypress.code == 258:
                    y -= 1
                    grid.set_pos(x, y)
                    grid.draw()
                if keypress.code == 259:
                    y += 1
                    grid.set_pos(x, y)
                    grid.draw()

                if keypress.code == 260:
                    x -= 1
                    grid.set_pos(x, y)
                    grid.draw()
                if keypress.code == 261:
                    x += 1
                    grid.set_pos(x, y)
                    grid.draw()

class Grid:
    def __init__(self, x_pos: int, y_pos: int, data, term: Terminal):
        self.x = term.width
        self.x_offset = 3
        self.y = term.height
        self.y_offset = 1
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.data = data
        self.term = term
        self.color_treshholds = {
            0: term.on_black,
            0.1: term.on_gray10,
            0.2: term.on_gray20,
            0.3: term.on_gray30,
            0.4: term.on_gray40,
            0.5: term.on_gray50,
            0.6: term.on_gray60,
            0.7: term.on_gray70,
            0.8: term.on_gray80,
            0.9: term.on_gray90,
            1: term.on_white
        }

    def on_resize(self, sig, action):
        self.draw()

    def set_pos(self, x_pos: int, y_pos: int) -> None:
        self.x_pos = x_pos
        self.y_pos = y_pos

    def draw_cell(self, start_x: int, start_y: int, value: float) -> None:
        color = self.color_treshholds[value]
        for y in range(start_y, end_y := start_y + self.y_offset + 1):
            for x in range(start_x, end_x := start_x + self.x_offset + 1):
                print(self.term.move(y, x) + color(' '))

    def draw(self) -> None:
        for y, row in enumerate(self.data):
            for x, value in enumerate(row):
                if x == 0:
                    start_x = x
                else:
                    start_x = x + x * self.x_offset

                if y == 0:
                    start_y = y
                else:
                    start_y = y + y * self.y_offset

                end_x = start_x + self.x_offset + 1
                end_y = start_y + self.y_offset + 1

                if (not end_x >= self.term.width) and (not end_y >=
                self.term.height):
                    self.draw_cell(start_x, start_y, value)

if __name__ == '__main__':

    run()
