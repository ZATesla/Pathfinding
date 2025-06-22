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
        self.terrain_cost = 1.0 # Default terrain cost
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
    """
    Returns the cost of moving from node1 to node2.
    Factors in the terrain_cost of the destination node (node2).
    """
    base_cost = COST_CARDINAL
    if abs(node1.row - node2.row) == 1 and abs(node1.col - node2.col) == 1: # is_diagonal
        base_cost = COST_DIAGONAL

    # Cost to enter node2 is base_cost multiplied by node2's terrain difficulty
    return base_cost * node2.terrain_cost

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


# --- Bidirectional Search Implementation ---
def bidirectional_search(grid, start_node, end_node):
    if not start_node or not end_node or start_node.is_obstacle or end_node.is_obstacle:
        return [], [], [], [], [] # path, visited_fwd, visited_bwd, open_fwd, open_bwd

    # Attributes for forward search
    # We'll use existing g, f_score, previous_node for forward.
    # Need separate tracking for backward search if node attributes are shared.
    # For simplicity, let's use dictionaries to store g_scores and previous nodes for backward search.

    g_fwd = {node: float('inf') for row in grid.nodes for node in row}
    g_bwd = {node: float('inf') for row in grid.nodes for node in row}

    prev_fwd = {node: None for row in grid.nodes for node in row}
    prev_bwd = {node: None for row in grid.nodes for node in row}

    g_fwd[start_node] = 0
    start_node.f_score = heuristic(start_node, end_node, grid.allow_diagonal_movement) # A* like f_score for forward

    g_bwd[end_node] = 0
    # For backward search, f_score can be g_bwd + heuristic to start_node
    # However, basic bidirectional Dijkstra/BFS doesn't strictly need f_scores in open set if just using g.
    # Let's use g-scores for priority in PQs for simplicity (like Dijkstra).

    pq_fwd = [(0, start_node)] # (g_score, node)
    pq_bwd = [(0, end_node)]   # (g_score, node)

    # Using sets to track nodes in open_set for quick lookups (complementing heapq)
    open_set_tracker_fwd = {start_node}
    open_set_tracker_bwd = {end_node}

    # Using sets to track visited nodes (closed set) for each direction
    closed_set_fwd = set()
    closed_set_bwd = set()

    # For visualization
    visited_nodes_in_order_fwd = []
    visited_nodes_in_order_bwd = []

    meeting_node = None
    path_cost = float('inf')

    while pq_fwd and pq_bwd:
        # Forward search step
        if pq_fwd:
            current_g_fwd, current_node_fwd = heapq.heappop(pq_fwd)

            if current_node_fwd not in open_set_tracker_fwd: # Already processed or removed with better path
                continue
            open_set_tracker_fwd.remove(current_node_fwd)

            closed_set_fwd.add(current_node_fwd)
            visited_nodes_in_order_fwd.append(current_node_fwd)

            # Check for meeting point
            if current_node_fwd in closed_set_bwd:
                current_path_cost = g_fwd[current_node_fwd] + g_bwd[current_node_fwd]
                if current_path_cost < path_cost:
                    path_cost = current_path_cost
                    meeting_node = current_node_fwd
                # Termination condition: if current_g_fwd + (g_bwd value from top of pq_bwd if available, else heuristic) > path_cost, can potentially stop
                # For basic Dijkstra-like expansion, we ensure optimal meeting after both frontiers meet.
                # A stricter termination uses sum of top keys of PQs. If sum_of_top_keys >= path_cost, stop.
                # For now, we continue until one PQ is empty or a more robust condition is met.
                # If the sum of costs of nodes at the top of both queues is >= path_cost, we can stop.
                if pq_bwd and (current_g_fwd + pq_bwd[0][0] >= path_cost): # Heuristic for pq_bwd[0][0] if it's f-score
                     break # Found potential shortest path


            for neighbor in current_node_fwd.neighbors:
                if neighbor in closed_set_fwd:
                    continue

                cost = get_move_cost(current_node_fwd, neighbor)
                temp_g_fwd = g_fwd[current_node_fwd] + cost

                if temp_g_fwd < g_fwd[neighbor]:
                    g_fwd[neighbor] = temp_g_fwd
                    prev_fwd[neighbor] = current_node_fwd
                    # For Dijkstra-like expansion, f_score in PQ is just g_score
                    heapq.heappush(pq_fwd, (temp_g_fwd, neighbor))
                    open_set_tracker_fwd.add(neighbor)

        # Backward search step
        if pq_bwd:
            current_g_bwd, current_node_bwd = heapq.heappop(pq_bwd)

            if current_node_bwd not in open_set_tracker_bwd:
                continue
            open_set_tracker_bwd.remove(current_node_bwd)

            closed_set_bwd.add(current_node_bwd)
            visited_nodes_in_order_bwd.append(current_node_bwd)

            if current_node_bwd in closed_set_fwd:
                current_path_cost = g_fwd[current_node_bwd] + g_bwd[current_node_bwd]
                if current_path_cost < path_cost:
                    path_cost = current_path_cost
                    meeting_node = current_node_bwd
                if pq_fwd and (current_g_bwd + pq_fwd[0][0] >= path_cost):
                    break

            for neighbor in current_node_bwd.neighbors:
                if neighbor in closed_set_bwd:
                    continue

                # Cost is from neighbor to current_node_bwd (reverse of edge)
                cost = get_move_cost(neighbor, current_node_bwd)
                temp_g_bwd = g_bwd[current_node_bwd] + cost

                if temp_g_bwd < g_bwd[neighbor]:
                    g_bwd[neighbor] = temp_g_bwd
                    prev_bwd[neighbor] = current_node_bwd
                    heapq.heappush(pq_bwd, (temp_g_bwd, neighbor))
                    open_set_tracker_bwd.add(neighbor)

        if not pq_fwd or not pq_bwd: # One queue is empty, no path possible through further expansion
            break


    # Path reconstruction
    final_path = []
    if meeting_node:
        # Reconstruct forward path
        temp = meeting_node
        while temp:
            final_path.append(temp)
            temp = prev_fwd[temp]
        final_path.reverse() # Path from start to meeting_node

        # Reconstruct backward path and append (excluding meeting_node itself)
        temp = prev_bwd[meeting_node] # Start from node before meeting_node in backward path
        while temp:
            final_path.append(temp)
            temp = prev_bwd[temp]
        # The backward path segment is naturally from meeting_node's predecessor towards end_node's predecessor.
        # It needs to be appended correctly.
        # Example: S->A->M and E->B->M. Path S->A->M->B->E
        # prev_fwd: M->A, A->S. Reversed: S->A->M
        # prev_bwd: M->B, B->E. We need M, then B, then E.
        # My reconstruction of backward part is adding E...B to S...M. It should be M-B-E.
        # Let's re-verify path reconstruction for backward part.
        # Path: S... -> prev_fwd[M] -> M <- prev_bwd[M] <- ...E
        # Forward part: S -> ... -> prev_fwd[prev_fwd[meeting_node]] -> prev_fwd[meeting_node] -> meeting_node
        # Backward part: meeting_node <- prev_bwd[meeting_node] <- prev_bwd[prev_bwd[meeting_node]] <- ... <- E

        # Corrected backward path reconstruction:
        path_segment_bwd = []
        temp = prev_bwd[meeting_node]
        while temp: # Iterate from node after meeting point towards end_node
            path_segment_bwd.append(temp)
            temp = prev_bwd[temp]
        # path_segment_bwd is currently [prev_to_meeting_on_bwd_path, ..., end_node_predecessor]
        # It needs to be reversed to get [end_node_predecessor, ..., prev_to_meeting_on_bwd_path]
        # No, this is wrong. prev_bwd[node] points to the node *from which we came* to reach 'node' in backward search.
        # So, if M is meeting node, prev_bwd[M] is the node that expanded M from backward search.
        # Path from E to M: E ... -> prev_bwd[prev_bwd[M]] -> prev_bwd[M] -> M.
        # We want M -> prev_bwd[M] -> prev_bwd[prev_bwd[M]] ... -> E.
        # The list `final_path` already has S...M. We need to append nodes from M towards E.
        # The current `final_path.append(temp)` for backward path is adding nodes in reverse order of path.
        # Let's reconstruct backward path separately then reverse and append.

        # path_fwd = final_path.copy() # S...M
        # path_bwd_segment_reversed = []
        # temp = meeting_node
        # # We need the path from M to E. prev_bwd stores "parent" towards E.
        # # So if M is current, prev_bwd[M] is parent of M in path from E.
        # # This means if we iterate from M using prev_bwd, we go E...prev(M)...M
        # # No, prev_bwd[X] = Y means edge Y->X was traversed in backward search (cost from Y to X)
        # # So g_bwd[X] = g_bwd[Y] + cost(Y,X). Y is closer to E.
        # # Path: E ... Y -> X. So Y = prev_bwd[X].
        # # To reconstruct E...M: M, prev_bwd[M], prev_bwd[prev_bwd[M]]... E. Then reverse. This is path_to_meeting_from_end
        # # To reconstruct M...E: M, (node whose prev_bwd is M), ... E
        # This is tricky. Let's use the standard way: reconstruct S->M and E->M, then combine.

        path_s_to_m = []
        curr = meeting_node
        while curr:
            path_s_to_m.append(curr)
            curr = prev_fwd[curr]
        path_s_to_m.reverse()

        path_e_to_m = []
        curr = meeting_node
        while curr: # This will include meeting_node
            path_e_to_m.append(curr)
            curr = prev_bwd[curr]
        path_e_to_m.reverse() # Now E ... -> meeting_node

        # Combine: path_s_to_m (S...M) + path_e_to_m reversed (M_prev ... E) skipping M from second part
        final_path = path_s_to_m
        if len(path_e_to_m) > 1: # if meeting_node is not end_node
            # path_e_to_m is E...prev_of_M...M. We want M...E. So reverse it: M...prev_of_M...E
            # then take from index 1 (skip M)
            final_path.extend(path_e_to_m[-2::-1]) # Reversed path_e_to_m, from element before last (M) down to E


    # For visualization, we can combine visited nodes.
    # Open sets are also tricky for direct visualization on Node unless we add more bool flags.
    # For now, just return the lists of nodes.
    # The GUI will have to decide how to color these (e.g. forward visited, backward visited, overlap)
    # For open sets, it's even harder to show simultaneously on node itself.
    # We return the trackers.

    # Consolidate visited nodes for return (can be used by GUI for coloring)
    # Order might be interleaved if we want to show step-by-step expansion.
    # For now, just concatenate. GUI can process visited_nodes_in_order_fwd and _bwd separately if needed.
    all_visited_for_return = visited_nodes_in_order_fwd + visited_nodes_in_order_bwd # Simplistic merge

    # Similarly for open sets (nodes that were in PQ at the end)
    final_open_set_fwd = list(open_set_tracker_fwd)
    final_open_set_bwd = list(open_set_tracker_bwd)

    return final_path, visited_nodes_in_order_fwd, visited_nodes_in_order_bwd, final_open_set_fwd, final_open_set_bwd


# --- Jump Point Search (JPS) Implementation ---

def _jps_is_walkable(grid, r, c):
    """ Helper to check if a node is within bounds and not an obstacle. """
    if not (0 <= r < grid.rows and 0 <= c < grid.cols):
        return False
    return not grid.nodes[r][c].is_obstacle

def _jps_jump(grid, current_r, current_c, dr, dc, start_node, end_node):
    """
    Recursively searches for a jump point.
    dr, dc define the direction from parent to current.
    """
    next_r, next_c = current_r + dr, current_c + dc

    if not _jps_is_walkable(grid, next_r, next_c):
        return None

    next_node = grid.nodes[next_r][next_c]
    if next_node == end_node:
        return next_node

    # Check for forced neighbors
    if dr != 0 and dc != 0: # Diagonal move
        # Check horizontally
        if (_jps_is_walkable(grid, next_r - dr, next_c + dc) and not _jps_is_walkable(grid, next_r - dr, next_c)) or \
           (_jps_is_walkable(grid, next_r + dr, next_c + dc) and not _jps_is_walkable(grid, next_r + dr, next_c)): # This rule seems off for diagonal forced.
             # Simplified: if horizontal or vertical component could lead to a forced neighbor
            pass # Revisit forced neighbor logic for diagonal

        # Check for forced neighbors specific to diagonal moves
        # If moving NE (dr=-1, dc=1):
        # Forced if obstacle at (r, c-1) [W] AND (r-1, c-1) [SW] is walkable -> (r-1,c+1) [current node] becomes jump point
        # Forced if obstacle at (r+1, c) [S] AND (r+1, c+1) [SE] is walkable -> (r-1,c+1) [current node] becomes jump point
        # This needs to be from the perspective of next_node, checking for forced moves from it.
        # (next_r, next_c) is the node being evaluated. (current_r, current_c) is its parent.

        # Pruning rule for diagonal: check if moving straight horizontally or vertically from (next_r, next_c)
        # in the components of the diagonal direction leads to a jump point.
        if _jps_jump(grid, next_r, next_c, dr, 0, start_node, end_node) or \
           _jps_jump(grid, next_r, next_c, 0, dc, start_node, end_node):
            return next_node
    else: # Straight move (horizontal or vertical)
        if dr != 0: # Vertical move (dr is -1 or 1, dc is 0)
            # Forced neighbor check: if (r, c+1) is obstacle and (r-dr, c+1) is not obstacle -> (r,c+1) is jump point from (r,c)
            if (_jps_is_walkable(grid, next_r, next_c + 1) and not _jps_is_walkable(grid, next_r - dr, next_c + 1)) or \
               (_jps_is_walkable(grid, next_r, next_c - 1) and not _jps_is_walkable(grid, next_r - dr, next_c - 1)):
                return next_node
        else: # Horizontal move (dc is -1 or 1, dr is 0)
            # Forced neighbor check: if (r+1, c) is obstacle and (r+1, c-dc) is not obstacle -> (r+1,c) is jump point
            if (_jps_is_walkable(grid, next_r + 1, next_c) and not _jps_is_walkable(grid, next_r + 1, next_c - dc)) or \
               (_jps_is_walkable(grid, next_r - 1, next_c) and not _jps_is_walkable(grid, next_r - 1, next_c - dc)):
                return next_node

    # If diagonal movement is not allowed, JPS simplifies (but this implementation assumes it can be based on grid setting)
    if not grid.allow_diagonal_movement and dr != 0 and dc != 0:
        return None # Cannot make this jump

    #return _jps_jump(grid, next_r, next_c, dr, dc, start_node, end_node) # Continue jump
    return None # MODIFIED FOR NOW: Prevent recursion error in placeholder

def _jps_identify_successors(grid, current_node, end_node):
    successors = []
    # Get pruned neighbors based on direction from parent (not explicitly stored on node for JPS, need to infer)
    # For now, let's consider all 8 directions if no parent, or pruned if parent exists.
    # This part is complex and needs parent direction. For a simpler start, assume current_node is from open list.
    # The parent is current_node.previous_node.

    parent = current_node.previous_node # This is the previous JUMP POINT
    px, py = (parent.col, parent.row) if parent else (-1,-1) # Use col,row for consistency
    cx, cy = current_node.col, current_node.row

    # Normalized direction from parent to current
    dx = (cx - px) // max(1, abs(cx - px)) if px != -1 else 0
    dy = (cy - py) // max(1, abs(cy - py)) if py != -1 else 0

    # Possible directions to explore (pruned set)
    # This logic is from standard JPS neighbor pruning rules.
    # TODO: Refine this pruning based on Harabor's papers, especially for no corner cutting.
    # For now, this is a placeholder for a more accurate pruning.

    possible_directions = []
    if dx != 0 and dy != 0: # Came from diagonal
        # Natural neighbors: (dx,0), (0,dy), (dx,dy)
        if _jps_is_walkable(grid, cy + dy, cx): possible_directions.append((0, dy))
        if _jps_is_walkable(grid, cy, cx + dx): possible_directions.append((dx, 0))
        if _jps_is_walkable(grid, cy + dy, cx + dx) and (_jps_is_walkable(grid, cy + dy, cx) or _jps_is_walkable(grid, cy, cx + dx)):
            possible_directions.append((dx, dy))
        # Forced neighbors (example for NE move: dx=1, dy=-1)
        # Check W for obstacle, if so, NW is forced. Check S for obstacle, if so, SE is forced.
        if not _jps_is_walkable(grid, cy, cx - dx) and _jps_is_walkable(grid, cy + dy, cx - dx): # Forced NW for NE
             possible_directions.append((-dx, dy))
        if not _jps_is_walkable(grid, cy - dy, cx) and _jps_is_walkable(grid, cy - dy, cx + dx): # Forced SE for NE
             possible_directions.append((dx, -dy))

    else: # Came from straight or is start_node
        if dx == 0: # Vertical move (or start node exploring vertically)
            if _jps_is_walkable(grid, cy + dy, cx):
                possible_directions.append((0, dy)) # Straight
            # Forced neighbors
            if not _jps_is_walkable(grid, cy, cx + 1) and _jps_is_walkable(grid, cy + dy, cx + 1): # Obstacle right
                possible_directions.append((1, dy))
            if not _jps_is_walkable(grid, cy, cx - 1) and _jps_is_walkable(grid, cy + dy, cx - 1): # Obstacle left
                possible_directions.append((-1, dy))
        elif dy == 0: # Horizontal move (or start node exploring horizontally)
            if _jps_is_walkable(grid, cy, cx + dx):
                possible_directions.append((dx, 0)) # Straight
            # Forced neighbors
            if not _jps_is_walkable(grid, cy + 1, cx) and _jps_is_walkable(grid, cy + 1, cx + dx): # Obstacle below
                possible_directions.append((dx, 1))
            if not _jps_is_walkable(grid, cy - 1, cx) and _jps_is_walkable(grid, cy - 1, cx + dx): # Obstacle above
                possible_directions.append((dx, -1))
        else: # Start node, explore all valid directions
            for dr_new in [-1, 0, 1]:
                for dc_new in [-1, 0, 1]:
                    if dr_new == 0 and dc_new == 0: continue
                    if not grid.allow_diagonal_movement and dr_new != 0 and dc_new != 0: continue
                    if _jps_is_walkable(grid, cy + dr_new, cx + dc_new):
                         possible_directions.append((dc_new, dr_new)) # dc, dr for consistency with jump func


    for dc_new, dr_new in possible_directions: # Note: jump uses (r,c,dr,dc) but directions here are (dc,dr)
        jump_point = _jps_jump(grid, cy, cx, dr_new, dc_new, current_node, end_node) # Pass current_node as start for this jump
        if jump_point:
            successors.append(jump_point)
    return successors


def jps_search(grid, start_node, end_node):
    # This is a placeholder and needs full A* main loop with JPS specific successor generation.
    # The _jps_jump and _jps_identify_successors are very rough first drafts and need significant refinement
    # based on the detailed JPS rules (especially forced neighbor detection and proper pruning).

    if not start_node or not end_node or start_node.is_obstacle or end_node.is_obstacle:
        return [], [], [] # path, visited_nodes, open_set_nodes

    open_set = [] # Priority queue (f_score, h_score, node)
    heapq.heappush(open_set, (start_node.f_score, heuristic(start_node, end_node, grid.allow_diagonal_movement), start_node))

    open_set_tracker = {start_node} # For quick lookups
    # previous_node handled by node.previous_node
    # g_score handled by node.g

    start_node.g = 0
    start_node.h_score = heuristic(start_node, end_node, grid.allow_diagonal_movement)
    start_node.f_score = start_node.g + start_node.h_score

    visited_nodes_in_order = [] # For visualization

    while open_set:
        _, _, current_jump_point = heapq.heappop(open_set)

        if current_jump_point not in open_set_tracker: # Already processed
            continue
        open_set_tracker.remove(current_jump_point)

        visited_nodes_in_order.append(current_jump_point) # Add jump point to visited

        if current_jump_point == end_node:
            path = []
            temp = end_node
            while temp:
                path.append(temp)
                temp = temp.previous_node
            return path[::-1], visited_nodes_in_order, list(open_set_tracker)

        successors = _jps_identify_successors(grid, current_jump_point, end_node)

        for successor_jp_node in successors:
            # Cost from current_jump_point to successor_jp_node
            # This needs to be the actual path cost, not just direct get_move_cost if JPS jumps over many cells
            # For now, using heuristic as placeholder for actual cost calculation during jump
            # This is a major simplification and likely incorrect for proper g-score accumulation in JPS

            # Correct g-score calculation: distance between current_jump_point and successor_jp_node
            # This means the jump function should also return the cost of the jump.
            # For now, let's use Manhattan/Octile distance as g cost between jump points
            # This is NOT the true path cost if terrain varies. JPS is for uniform cost grids.
            # If we use terrain_cost, the jump function must accumulate it.

            # Assuming jump function ensures optimal path to jump point, so g is direct distance cost
            # This is a simplification; true JPS calculates g-cost by summing costs along jumped path.
            # For initial structure:
            move_dx = abs(current_jump_point.col - successor_jp_node.col)
            move_dy = abs(current_jump_point.row - successor_jp_node.row)
            cost_to_successor = 0
            if grid.allow_diagonal_movement:
                diag_steps = min(move_dx, move_dy)
                card_steps = max(move_dx, move_dy) - diag_steps
                cost_to_successor = diag_steps * COST_DIAGONAL * successor_jp_node.terrain_cost + \
                                    card_steps * COST_CARDINAL * successor_jp_node.terrain_cost
            else:
                cost_to_successor = (move_dx + move_dy) * COST_CARDINAL * successor_jp_node.terrain_cost


            temp_g_score = current_jump_point.g + cost_to_successor # Simplified cost

            if temp_g_score < successor_jp_node.g:
                successor_jp_node.previous_node = current_jump_point
                successor_jp_node.g = temp_g_score
                successor_jp_node.h_score = heuristic(successor_jp_node, end_node, grid.allow_diagonal_movement)
                successor_jp_node.f_score = successor_jp_node.g + successor_jp_node.h_score
                if successor_jp_node not in open_set_tracker:
                    heapq.heappush(open_set, (successor_jp_node.f_score, successor_jp_node.h_score, successor_jp_node))
                    open_set_tracker.add(successor_jp_node)

    return [], visited_nodes_in_order, list(open_set_tracker) # No path found
