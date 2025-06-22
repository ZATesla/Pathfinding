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
        self.g = float('inf')  # Renamed from g_score
        self.rhs = float('inf') # Added for D* Lite
        self.h_score = float('inf') # Still used by A*, can be reused by D* heuristic
        self.f_score = float('inf') # Still used by A*, D* Lite uses a different key structure
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
        self.g = float('inf')
        self.rhs = float('inf')
        self.h_score = float('inf') # Reset for A* if used
        self.f_score = float('inf') # Reset for A* if used
        self.previous_node = None

        self.is_visited_by_algorithm = False
        self.is_in_open_set_for_algorithm = False
        self.is_part_of_path = False

    def add_neighbors(self, grid_nodes_matrix, allow_diagonal=True):
        self.neighbors = []
        # Cardinal directions
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

        if allow_diagonal:
            # Diagonal directions
            # Down-Right
            if self.row < self.total_rows - 1 and self.col < self.total_cols - 1 and \
               not grid_nodes_matrix[self.row + 1][self.col + 1].is_obstacle:
                # Check for corner cutting: make sure adjacent cardinal cells are not obstacles
                if not grid_nodes_matrix[self.row + 1][self.col].is_obstacle or \
                   not grid_nodes_matrix[self.row][self.col + 1].is_obstacle:
                    self.neighbors.append(grid_nodes_matrix[self.row + 1][self.col + 1])
            # Down-Left
            if self.row < self.total_rows - 1 and self.col > 0 and \
               not grid_nodes_matrix[self.row + 1][self.col - 1].is_obstacle:
                if not grid_nodes_matrix[self.row + 1][self.col].is_obstacle or \
                   not grid_nodes_matrix[self.row][self.col - 1].is_obstacle:
                    self.neighbors.append(grid_nodes_matrix[self.row + 1][self.col - 1])
            # Up-Right
            if self.row > 0 and self.col < self.total_cols - 1 and \
               not grid_nodes_matrix[self.row - 1][self.col + 1].is_obstacle:
                if not grid_nodes_matrix[self.row - 1][self.col].is_obstacle or \
                   not grid_nodes_matrix[self.row][self.col + 1].is_obstacle:
                    self.neighbors.append(grid_nodes_matrix[self.row - 1][self.col + 1])
            # Up-Left
            if self.row > 0 and self.col > 0 and \
               not grid_nodes_matrix[self.row - 1][self.col - 1].is_obstacle:
                if not grid_nodes_matrix[self.row - 1][self.col].is_obstacle or \
                   not grid_nodes_matrix[self.row][self.col - 1].is_obstacle:
                    self.neighbors.append(grid_nodes_matrix[self.row - 1][self.col - 1])


class Grid:
    def __init__(self, rows, cols, cell_width_for_nodes=0, cell_height_for_nodes=0): # cell_width/height optional
        self.rows = rows
        self.cols = cols
        self.allow_diagonal_movement = True # Default to True, can be changed by GUI
        # Store cell_width/height if nodes need them, but Grid itself doesn't use them for logic
        self.nodes = [[Node(r, c, rows, cols, cell_width_for_nodes, cell_height_for_nodes) for c in range(cols)] for r in range(rows)]
        self.start_node = None
        self.end_node = None
        self.update_all_node_neighbors() # Initial update

    def update_all_node_neighbors(self): # Pass the diagonal movement setting
        for r in range(self.rows):
            for c in range(self.cols):
                self.nodes[r][c].add_neighbors(self.nodes, self.allow_diagonal_movement)

    def set_allow_diagonal_movement(self, allow: bool):
        self.allow_diagonal_movement = allow
        self.update_all_node_neighbors() # Re-calculate neighbors when this setting changes

    def reset_algorithm_states(self): # Resets states on all nodes
        for r in range(self.rows):
            for c in range(self.cols):
                self.nodes[r][c].reset_algorithm_attributes()

    def get_node(self, row, col): # Helper to get a node
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.nodes[row][col]
        return None

    def set_target_node(self, row, col):
        """Sets a new target node (end_node) for the grid."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            new_target_node = self.nodes[row][col]
            if not new_target_node.is_obstacle:
                # Unset previous end_node's specific visual role if needed by GUI (not strictly here)
                # (The GUI checks 'node == grid.end_node', so direct flags aren't essential on Node itself for this)
                self.end_node = new_target_node
                print(f"Target node set to ({row}, {col})") # For feedback
                return True
            else:
                print(f"Cannot set target at ({row}, {col}) as it's an obstacle.")
                return False
        else:
            print(f"Target coordinates ({row}, {col}) are out of bounds.")
            return False


# --- Pathfinding Algorithms ---

# Cost function for moving between adjacent nodes
COST_CARDINAL = 1.0
COST_DIAGONAL = 1.41421356 # sqrt(2) - can be adjusted, e.g. to 1.0 for Chebyshev distance like behavior

def get_move_cost(node1, node2):
    """Returns the cost of moving from node1 to node2."""
    is_diagonal = abs(node1.row - node2.row) == 1 and abs(node1.col - node2.col) == 1
    if is_diagonal:
        return COST_DIAGONAL
    return COST_CARDINAL

def heuristic(node_a, node_b, allow_diagonal=True):
    """
    Calculates heuristic distance.
    Uses Manhattan distance if diagonal movement is not allowed.
    Uses Octile distance (approximation) if diagonal movement is allowed.
    """
    dx = abs(node_a.col - node_b.col)
    dy = abs(node_a.row - node_b.row)

    if allow_diagonal:
        # Octile distance (D_diag * min(dx, dy) + D_cardinal * (max(dx, dy) - min(dx, dy)))
        # Assumes COST_CARDINAL = 1, COST_DIAGONAL = sqrt(2) or similar
        return COST_DIAGONAL * min(dx, dy) + COST_CARDINAL * (max(dx, dy) - min(dx, dy))
    else:
        # Manhattan distance
        return COST_CARDINAL * (dx + dy)


def dijkstra(grid, start_node, end_node):
    if not start_node or not end_node or start_node.is_obstacle or end_node.is_obstacle:
        return [], [], []

    visited_nodes_in_order = []
    start_node.g = 0
    start_node.h_score = 0 # Dijkstra doesn't use heuristic for path decision but can store it
    start_node.f_score = start_node.g

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
            return path[::-1], visited_nodes_in_order, list(open_set_tracker)

        if current_f_score > current_node.f_score: # Already found a shorter path to this node
            continue

        for neighbor in current_node.neighbors:
            cost = get_move_cost(current_node, neighbor)
            temp_g = current_node.g + cost

            if temp_g < neighbor.g:
                neighbor.previous_node = current_node
                neighbor.g = temp_g
                neighbor.f_score = neighbor.g # For Dijkstra, f_score is g_score

                if neighbor not in open_set_tracker:
                    heapq.heappush(pq, (neighbor.f_score, neighbor))
                    open_set_tracker.add(neighbor)
                # If it was in open_set_tracker but already popped, this new path is longer or equal.
                # If it is still in open_set_tracker (i.e. in pq), heapq handles pushing this better path.
                # The check `if current_node not in open_set_tracker` handles already processed nodes.

    # print("No path found by Dijkstra.") # Silenced for tests
    final_open_set = list(open_set_tracker)
    return [], visited_nodes_in_order, final_open_set


# --- D* Lite Implementation ---

# D* Lite Key: (min(g, rhs) + h(start, node), min(g, rhs))
# The heuristic is from the current node to the start node.
# Note: In some D* Lite versions, km (key modifier for moved start) is added to h.
# For now, assuming start node is fixed for a given planning episode.
def calculate_d_star_key(node, start_node, goal_node, heuristic_func, allow_diagonal): # Added allow_diagonal
    h_val = heuristic_func(node, start_node, allow_diagonal) # Pass allow_diagonal to heuristic
    min_g_rhs = min(node.g, node.rhs)
    return (min_g_rhs + h_val, min_g_rhs)

_d_star_pq = [] # Global-like for the module, or pass around
_d_star_open_set_tracker = set() # Tracks nodes conceptually in PQ

# Need to pass allow_diagonal down to where heuristic is called.
# The grid object (which knows about allow_diagonal) is available in run_d_star_lite.
# We can pass it from there.

def d_star_lite_initialize(grid, start_node, goal_node, heuristic_func): # allow_diagonal implicitly from grid
    global _d_star_pq, _d_star_open_set_tracker
    _d_star_pq = []
    _d_star_open_set_tracker = set()

    grid.reset_algorithm_states() # Resets g, rhs for all nodes to inf

    goal_node.rhs = 0
    key = calculate_d_star_key(goal_node, start_node, goal_node, heuristic_func, grid.allow_diagonal_movement)
    heapq.heappush(_d_star_pq, (key, goal_node))
    _d_star_open_set_tracker.add(goal_node)


def d_star_lite_update_node(node, grid, start_node, goal_node, heuristic_func): # allow_diagonal implicitly from grid
    global _d_star_pq, _d_star_open_set_tracker

    if node != goal_node:
        min_rhs = float('inf')
        for successor in node.neighbors: # Successors are just neighbors in grid
            cost = get_move_cost(node, successor) # Use get_move_cost
            min_rhs = min(min_rhs, successor.g + cost)
        node.rhs = min_rhs

    # If node is in PQ, conceptually remove it (actual removal is tricky with heapq)
    # We handle this by checking when popping, or by only adding if not inconsistent.
    # For D*, it's common to allow duplicates and let the processing logic sort it out,
    # or use a PQ that supports decrease_key.
    # Here, we will rely on the main loop's logic to handle nodes correctly even if they
    # have multiple entries in PQ due to updates.
    # A simple way to handle "if node in PQ, remove it":
    # For this version, we'll assume that if a node is re-evaluated and needs to be in PQ,
    # it will be re-added. The main loop (compute_shortest_path) handles outdated entries.
    # If we want to be more explicit about removing from _d_star_open_set_tracker if g == rhs:
    if node in _d_star_open_set_tracker and node.g == node.rhs:
         pass # Rely on "add if inconsistent" logic


    if node.g != node.rhs:
        key = calculate_d_star_key(node, start_node, goal_node, heuristic_func, grid.allow_diagonal_movement)
        heapq.heappush(_d_star_pq, (key, node))
        _d_star_open_set_tracker.add(node)
    elif node in _d_star_open_set_tracker: # g == rhs, but it's still in open set tracker
        _d_star_open_set_tracker.remove(node) # Became consistent, remove from conceptual open set


def d_star_lite_compute_shortest_path(grid, start_node, goal_node, heuristic_func): # allow_diagonal from grid
    global _d_star_pq, _d_star_open_set_tracker

    processed_nodes_for_viz = []

    # Loop while top_key < key(start) OR start.rhs != start.g
    while (_d_star_pq and calculate_d_star_key(start_node, start_node, goal_node, heuristic_func, grid.allow_diagonal_movement) > _d_star_pq[0][0]) \
          or start_node.g != start_node.rhs:

        if not _d_star_pq: break

        k_old_popped, u_node = heapq.heappop(_d_star_pq)

        if u_node not in _d_star_open_set_tracker and u_node.g == u_node.rhs :
            continue

        if u_node in _d_star_open_set_tracker:
             _d_star_open_set_tracker.remove(u_node)

        processed_nodes_for_viz.append(u_node)

        k_new_recalc = calculate_d_star_key(u_node, start_node, goal_node, heuristic_func, grid.allow_diagonal_movement)

        if k_old_popped < k_new_recalc:
            heapq.heappush(_d_star_pq, (k_new_recalc, u_node))
            _d_star_open_set_tracker.add(u_node)
        elif u_node.g > u_node.rhs:
            u_node.g = u_node.rhs
            for pred_node in u_node.neighbors:
                d_star_lite_update_node(pred_node, grid, start_node, goal_node, heuristic_func)
        else: # u_node.g < u_node.rhs (locally underconsistent)
            g_old = u_node.g
            u_node.g = float('inf')
            # Update predecessors of u_node
            for pred_node in u_node.neighbors:
                # If pred_node's rhs was calculated through u_node, it might need update
                # This is complex: check if pred_node.rhs == g_old + cost(pred_node, u_node)
                # Simpler: just call update_node, it will re-calculate rhs if necessary.
                d_star_lite_update_node(pred_node, grid, start_node, goal_node, heuristic_func)

            # Update u_node itself (its g changed to inf, its rhs might need re-evaluation)
            d_star_lite_update_node(u_node, grid, start_node, goal_node, heuristic_func)


    # Path Reconstruction
    path = []
    if start_node.g == float('inf'):
        print("No path found by D* Lite.")
        return [], processed_nodes_for_viz, list(_d_star_open_set_tracker)

    curr = start_node
    path.append(curr)
    while curr != goal_node:
        min_total_cost_to_goal = float('inf') # This should be g(neighbor) + cost(curr, neighbor) effectively
        next_node = None
        if not curr.neighbors:
            print("Error in path reconstruction: current node has no neighbors.")
            return [], processed_nodes_for_viz, list(_d_star_open_set_tracker)

        for neighbor in curr.neighbors:
            cost_to_neighbor = get_move_cost(curr, neighbor)
            # Path is built by moving to successor s' that has lowest g(s') + cost(curr, s')
            # For D* Lite, we move towards the neighbor that minimizes cost_to_neighbor + g(neighbor)
            # This is essentially finding the neighbor with the smallest g-value if costs are uniform (1).
            # With variable costs, we need to check cost_to_neighbor + neighbor.g
            current_path_cost_via_neighbor = cost_to_neighbor + neighbor.g
            if current_path_cost_via_neighbor < min_total_cost_to_goal:
                min_total_cost_to_goal = current_path_cost_via_neighbor
                next_node = neighbor
            elif current_path_cost_via_neighbor == min_total_cost_to_goal: # Tie-breaking: prefer diagonal if possible or stick to one
                 if next_node is None: # First one found
                     next_node = neighbor
                 # Could add more sophisticated tie-breaking here if needed (e.g. prefer diagonal)

        if next_node is None :
             # This can happen if all neighbors have g = inf, or start_node.g itself is inf
             print(f"Path reconstruction failed: No valid next node from {curr.get_pos()} with g={curr.g}")
             # This indicates no path or an issue in g value propagation.
             return [], processed_nodes_for_viz, list(_d_star_open_set_tracker) # Return empty path

        curr = next_node
        path.append(curr)
        if len(path) > grid.rows * grid.cols: # Safety break for very long paths / loops
            print("Path reconstruction exceeded max length.")
            return [], processed_nodes_for_viz, list(_d_star_open_set_tracker)


    return path, processed_nodes_for_viz, list(_d_star_open_set_tracker)


def run_d_star_lite(grid, start_node, goal_node, heuristic_func):
    """
    Main orchestration function for D* Lite.
    Manages initialization and calls compute_shortest_path.
    This is the primary function to be called from the GUI or test environment.
    """
    if not start_node or not goal_node or start_node.is_obstacle or goal_node.is_obstacle:
        print("D* Lite: Start or Goal is invalid or an obstacle.")
        return [], [], []

    # Set the grid's official start and end nodes
    grid.start_node = start_node
    grid.end_node = goal_node # The 'goal_node' param is the initial goal

    d_star_lite_initialize(grid, start_node, goal_node, heuristic_func)

    # For dynamic changes (e.g., obstacles appear, goal moves):
    # 1. Update affected edge costs / node obstacle status.
    # 2. For each node u whose edge cost changed, update u.rhs and call d_star_lite_update_node(u, ...).
    #    (Also for its affected neighbor v, update v.rhs and call d_star_lite_update_node(v, ...)).
    # 3. If goal moves: update heuristic values for affected nodes (may need to re-add to PQ),
    #    update old_goal.rhs, new_goal.rhs = 0, and call d_star_lite_update_node for relevant nodes.
    # 4. Then, call d_star_lite_compute_shortest_path again.
    # This subtask is initial computation, not dynamic updates yet.

    path, visited_nodes, open_set = d_star_lite_compute_shortest_path(grid, start_node, goal_node, heuristic_func)

    # Convert node objects in open_set_tracker (which are actual Node objects) back to (key, node) for consistency if needed
    # For now, open_set is just list of Node objects from _d_star_open_set_tracker

    return path, visited_nodes, open_set

def reset_d_star_lite_internals():
    """Resets the internal global-like D* Lite priority queue and tracker."""
    global _d_star_pq, _d_star_open_set_tracker
    _d_star_pq = []
    _d_star_open_set_tracker = set()


def a_star(grid, start_node, end_node):
    if not start_node or not end_node or start_node.is_obstacle or end_node.is_obstacle:
        # print("Start or end node is missing or is an obstacle for A*.") # Silenced for tests
        return [], [], []

    visited_nodes_in_order = []
    start_node.g = 0
    # Pass grid.allow_diagonal_movement to heuristic
    start_node.h_score = heuristic(start_node, end_node, grid.allow_diagonal_movement)
    start_node.f_score = start_node.g + start_node.h_score

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
            return path[::-1], visited_nodes_in_order, list(open_set_tracker)

        if current_f_score > current_node.f_score: # Optimization
            continue

        for neighbor in current_node.neighbors:
            cost = get_move_cost(current_node, neighbor)
            temp_g = current_node.g + cost

            if temp_g < neighbor.g:
                neighbor.previous_node = current_node
                neighbor.g = temp_g
                neighbor.h_score = heuristic(neighbor, end_node, grid.allow_diagonal_movement)
                neighbor.f_score = neighbor.g + neighbor.h_score

                # Add to PQ even if already in open_set_tracker.
                # The check at the start of the loop handles cases where a node is pulled
                # handles cases where a node is pulled from PQ after a shorter path to it was already found and processed.
                heapq.heappush(pq, (neighbor.f_score, neighbor))
                open_set_tracker.add(neighbor) # Ensure it's marked as "conceptually" open

    # print("No path found by A*.") # Silenced for tests
    final_open_set = list(open_set_tracker)
    return [], visited_nodes_in_order, final_open_set
