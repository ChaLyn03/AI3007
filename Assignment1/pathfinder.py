import numpy as np
import math
from queue import Queue, PriorityQueue
import sys

def parse_map(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        rows, cols = map(int, lines[0].split())
        start = tuple(map(int, lines[1].split()))
        goal = tuple(map(int, lines[2].split()))
        # Adjusted to handle spaces correctly by splitting each line on spaces after stripping.
        map_data = np.array([line.strip().split() for line in lines[3:]])
    return map_data, start, goal

def to_xy(pos):
    return pos[1] - 1, pos[0] - 1

def from_xy(pos):
    return pos[1] + 1, pos[0] + 1

def is_valid(state, map_data):
    x, y = state
    return 0 <= x < map_data.shape[1] and 0 <= y < map_data.shape[0] and map_data[y, x] != 'X'

def calculate_cost(map_data, from_pos, to_pos):
    elevation_from = int(map_data[from_pos[1], from_pos[0]])
    elevation_to = int(map_data[to_pos[1], to_pos[0]])
    if elevation_to > elevation_from:
        return 1 + (elevation_to - elevation_from)
    else:
        return 1

def bfs(map_data, start, goal):
    start = to_xy(start)
    goal = to_xy(goal)
    queue = Queue()
    queue.put((start, [start]))
    visited = set()

    while not queue.empty():
        current, path = queue.get()
        if current == goal:
            return path

        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_state = (current[0] + dx, current[1] + dy)
            if is_valid(next_state, map_data):
                queue.put((next_state, path + [next_state]))

    return None

def ucs(map_data, start, goal):
    start = to_xy(start)
    goal = to_xy(goal)
    frontier = PriorityQueue()
    frontier.put((0, start, [start]))
    visited = set()

    while not frontier.empty():
        current_cost, current, path = frontier.get()

        if current == goal:
            return path

        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_state = (current[0] + dx, current[1] + dy)
            if is_valid(next_state, map_data):
                new_cost = current_cost + calculate_cost(map_data, current, next_state)
                frontier.put((new_cost, next_state, path + [next_state]))

    return None

def manhattan_distance(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def euclidean_distance(pos, goal):
    return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

def astar(map_data, start, goal, heuristic=manhattan_distance):
    start = to_xy(start)
    goal = to_xy(goal)
    frontier = PriorityQueue()
    frontier.put((0, start, [start]))
    costs = {start: 0}

    while not frontier.empty():
        current_cost, current, path = frontier.get()

        if current == goal:
            return path

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_state = (current[0] + dx, current[1] + dy)
            if is_valid(next_state, map_data):
                new_cost = costs[current] + calculate_cost(map_data, current, next_state)
                if next_state not in costs or new_cost < costs[next_state]:
                    costs[next_state] = new_cost
                    priority = new_cost + heuristic(next_state, goal)
                    frontier.put((priority, next_state, path + [next_state]))

    return None

def print_path_on_map(map_data, path):
    path_set = set(path)
    for y in range(map_data.shape[0]):
        for x in range(map_data.shape[1]):
            if (x, y) in path_set:
                print('*', end=' ')
            else:
                print(map_data[y, x], end=' ')
        print()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python pathfinder.py [map] [algorithm] [heuristic]")
        sys.exit(1)

    map_file_path = sys.argv[1]
    algorithm = sys.argv[2].lower()
    heuristic_input = sys.argv[3].lower()

    map_data, start, goal = parse_map(map_file_path)
    path = None

    if algorithm == "bfs":
        path = bfs(map_data, start, goal)
    elif algorithm == "ucs":
        path = ucs(map_data, start, goal)
    elif algorithm == "astar":
        if heuristic_input == "euclidean":
            path = astar(map_data, start, goal, heuristic=euclidean_distance)
        elif heuristic_input == "manhattan":
            path = astar(map_data, start, goal)
        else:
            print("Invalid heuristic. Use 'euclidean' or 'manhattan'.")
            sys.exit(1)
    else:
        print("Invalid algorithm. Use 'bfs', 'ucs', or 'astar'.")
        sys.exit(1)

    if path:
        print_path_on_map(map_data, path)
    else:
        print("null")