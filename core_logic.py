import heapq

# Note: Pygame specific constants (WINDOW_WIDTH, colors, etc.) are NOT here.
# CELL_WIDTH/HEIGHT are passed to Node but not strictly needed for core logic if x,y are for GUI.

class Node:
    def __init__(self, row, col, total_rows, total_cols, width=0, height=0): # width/height optional for core
        self.row = row
        self.col = col
        # x, y, width, height are primarily for GUI, but pathfinding might store them.
        # For core logic, they are not strictly necessary for algorithm operation.
        self.x = col * width
        self.y = row * height
        self.width = width
        self.height = height

        self.is_obstacle = False
        self.g_score = float('inf')
        self.h_score = float('inf')
        self.f_score = float('inf')
        self.previous_node = None
        self.neighbors = []
        self.total_rows = total_rows
        self.total_cols = total_cols

        # Visualization attributes - these might be better handled by a wrapper class in GUI
        # but for now, keeping them here to minimize changes to algorithm returns.
        self.is_visited_by_algorithm = False
        self.is_in_open_set_for_algorithm = False
        self.is_part_of_path = False

    def __lt__(self, other):
        return self.f_score < other.f_score

    def get_pos(self):
        return self.row, self.col

    def reset_algorithm_attributes(self):
        self.g_score = float('inf')
        self.h_score = float('inf')
        self.f_score = float('inf')
        self.previous_node = None

        self.is_visited_by_algorithm = False
        self.is_in_open_set_for_algorithm = False
        self.is_part_of_path = False

    def add_neighbors(self, grid_nodes_matrix): # Renamed from grid_nodes for clarity
        self.neighbors = []
        # Down
        if self.row < self.total_rows - 1 and not grid_nodes_matrix[self.row + 1][self.col].is_obstacle:
            self.neighbors.append(grid_nodes_matrix[self.row + 1][self.col])
        # Up
        if self.row > 0 and not grid_nodes_matrix[self.row - 1][self.col].is_obstacle:
            self.neighbors.append(grid_nodes_matrix[self.row - 1][self.col])
        # Right
        if self.col < self.total_cols - 1 and not grid_nodes_matrix[self.row][self.col + 1].is_obstacle:
            self.neighbors.append(grid_nodes_matrix[self.row][self.col + 1])
        # Left
        if self.col > 0 and not grid_nodes_matrix[self.row][self.col - 1].is_obstacle:
            self.neighbors.append(grid_nodes_matrix[self.row][self.col - 1])


class Grid:
    def __init__(self, rows, cols, cell_width_for_nodes=0, cell_height_for_nodes=0): # cell_width/height optional
        self.rows = rows
        self.cols = cols
        # Store cell_width/height if nodes need them, but Grid itself doesn't use them for logic
        self.nodes = [[Node(r, c, rows, cols, cell_width_for_nodes, cell_height_for_nodes) for c in range(cols)] for r in range(rows)]
        self.start_node = None
        self.end_node = None
        self.update_all_node_neighbors()

    def update_all_node_neighbors(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.nodes[r][c].add_neighbors(self.nodes) # Pass the matrix of nodes

    def reset_algorithm_states(self): # Resets states on all nodes
        for r in range(self.rows):
            for c in range(self.cols):
                self.nodes[r][c].reset_algorithm_attributes()

    def get_node(self, row, col): # Helper to get a node
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.nodes[row][col]
        return None

# --- Pathfinding Algorithms ---

def heuristic(node_a, node_b):
    return abs(node_a.row - node_b.row) + abs(node_a.col - node_b.col)

def dijkstra(grid, start_node, end_node):
    if not start_node or not end_node or start_node.is_obstacle or end_node.is_obstacle:
        # print("Start or end node is missing or is an obstacle.") # Silenced for tests
        return [], [], []

    visited_nodes_in_order = []
    start_node.g_score = 0
    start_node.h_score = 0
    start_node.f_score = start_node.g_score + start_node.h_score

    pq = [(start_node.f_score, start_node)]
    open_set_tracker = {start_node}

    while pq:
        current_f_score, current_node = heapq.heappop(pq)

        if current_node not in open_set_tracker: # Node already processed or removed
             continue
        open_set_tracker.remove(current_node)

        visited_nodes_in_order.append(current_node)

        if current_node == end_node:
            path = []
            temp = end_node
            while temp:
                path.append(temp)
                temp = temp.previous_node
            final_open_set = list(open_set_tracker)
            return path[::-1], visited_nodes_in_order, final_open_set

        if current_f_score > current_node.f_score :
            continue

        for neighbor in current_node.neighbors:
            temp_g_score = current_node.g_score + 1

            if temp_g_score < neighbor.g_score:
                neighbor.previous_node = current_node
                neighbor.g_score = temp_g_score
                neighbor.h_score = 0
                neighbor.f_score = neighbor.g_score + neighbor.h_score

                if neighbor not in open_set_tracker: # Add to PQ only if not processed from open set
                    heapq.heappush(pq, (neighbor.f_score, neighbor))
                    open_set_tracker.add(neighbor)
                # If it was in open_set_tracker but already popped, this new path is longer or equal.
                # If it is still in open_set_tracker (i.e. in pq), heapq handles pushing this better path.
                # The check `if current_node not in open_set_tracker` handles already processed nodes.

    # print("No path found by Dijkstra.") # Silenced for tests
    final_open_set = list(open_set_tracker)
    return [], visited_nodes_in_order, final_open_set


def a_star(grid, start_node, end_node):
    if not start_node or not end_node or start_node.is_obstacle or end_node.is_obstacle:
        # print("Start or end node is missing or is an obstacle for A*.") # Silenced for tests
        return [], [], []

    visited_nodes_in_order = []
    start_node.g_score = 0
    start_node.h_score = heuristic(start_node, end_node)
    start_node.f_score = start_node.g_score + start_node.h_score

    pq = [(start_node.f_score, start_node)]
    open_set_tracker = {start_node}

    while pq:
        current_f_score, current_node = heapq.heappop(pq)

        if current_node not in open_set_tracker:
            continue
        open_set_tracker.remove(current_node)

        visited_nodes_in_order.append(current_node)

        if current_node == end_node:
            path = []
            temp = end_node
            while temp:
                path.append(temp)
                temp = temp.previous_node
            final_open_set = list(open_set_tracker)
            return path[::-1], visited_nodes_in_order, final_open_set

        if current_f_score > current_node.f_score: # Optimization
             continue

        for neighbor in current_node.neighbors:
            temp_g_score = current_node.g_score + 1

            if temp_g_score < neighbor.g_score:
                neighbor.previous_node = current_node
                neighbor.g_score = temp_g_score
                neighbor.h_score = heuristic(neighbor, end_node)
                neighbor.f_score = neighbor.g_score + neighbor.h_score

                # Add to PQ even if already in open_set_tracker (i.e. in pq).
                # The check `if current_node not in open_set_tracker` at loop start
                # handles cases where a node is pulled from PQ after a shorter path to it was already found and processed.
                heapq.heappush(pq, (neighbor.f_score, neighbor))
                open_set_tracker.add(neighbor) # Ensure it's marked as "conceptually" open

    # print("No path found by A*.") # Silenced for tests
    final_open_set = list(open_set_tracker)
    return [], visited_nodes_in_order, final_open_set

```
