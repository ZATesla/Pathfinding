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

def heuristic(node_a, node_b):
    return abs(node_a.row - node_b.row) + abs(node_a.col - node_b.col)

def dijkstra(grid, start_node, end_node):
    if not start_node or not end_node or start_node.is_obstacle or end_node.is_obstacle:
        # print("Start or end node is missing or is an obstacle.") # Silenced for tests
        return [], [], []

    visited_nodes_in_order = []
    start_node.g = 0 # Use .g
    start_node.h_score = 0
    start_node.f_score = start_node.g + start_node.h_score # Use .g

    # For Dijkstra, f_score is effectively g_score as h_score is 0.
    # The __lt__ method on Node uses f_score, so this is fine.
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
            temp_g = current_node.g + 1 # Use .g

            if temp_g < neighbor.g: # Use .g
                neighbor.previous_node = current_node
                neighbor.g = temp_g # Use .g
                neighbor.h_score = 0
                neighbor.f_score = neighbor.g + neighbor.h_score # Use .g

                if neighbor not in open_set_tracker: # Add to PQ only if not processed from open set
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
def calculate_d_star_key(node, start_node, goal_node, heuristic_func): # goal_node is not used for key's h here
    h_val = heuristic_func(node, start_node)
    min_g_rhs = min(node.g, node.rhs)
    return (min_g_rhs + h_val, min_g_rhs)

_d_star_pq = [] # Global-like for the module, or pass around
_d_star_open_set_tracker = set() # Tracks nodes conceptually in PQ

def d_star_lite_initialize(grid, start_node, goal_node, heuristic_func):
    global _d_star_pq, _d_star_open_set_tracker
    _d_star_pq = []
    _d_star_open_set_tracker = set()

    grid.reset_algorithm_states() # Resets g, rhs for all nodes to inf

    goal_node.rhs = 0
    key = calculate_d_star_key(goal_node, start_node, goal_node, heuristic_func) # Pass goal_node as current_node_for_h_calc_placeholder
    heapq.heappush(_d_star_pq, (key, goal_node))
    _d_star_open_set_tracker.add(goal_node)

    # grid.start_node = start_node # Ensure grid object knows its start/goal
    # grid.end_node = goal_node   # These should be set before calling run_d_star_lite

def d_star_lite_update_node(node, grid, start_node, goal_node, heuristic_func):
    global _d_star_pq, _d_star_open_set_tracker

    if node != goal_node:
        min_rhs = float('inf')
        for successor in node.neighbors: # Successors are just neighbors in grid
            cost = 1 # Assuming uniform cost
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
         # Conceptually, it might be removed if consistent and in PQ, but D* adds if g != rhs
         # Forcing removal here might be complex if not using a PQ supporting direct removal.
         # Let's rely on the "add if inconsistent" logic below.
         pass


    if node.g != node.rhs:
        key = calculate_d_star_key(node, start_node, goal_node, heuristic_func)
        heapq.heappush(_d_star_pq, (key, node))
        _d_star_open_set_tracker.add(node)
    elif node in _d_star_open_set_tracker: # g == rhs, but it's still in open set tracker
        # This means it became consistent. Conceptually remove from PQ.
        # Actual removal from heapq is hard. We mark it in tracker.
        # The main loop, when popping, should check if node is still valid to process.
        _d_star_open_set_tracker.remove(node)


def d_star_lite_compute_shortest_path(grid, start_node, goal_node, heuristic_func):
    global _d_star_pq, _d_star_open_set_tracker

    processed_nodes_for_viz = [] # For visualization

    # Loop while top_key < key(start) OR start.rhs != start.g
    # heapq.nsmallest(1, _d_star_pq)[0][0] gives top_key without popping
    while (_d_star_pq and calculate_d_star_key(start_node, start_node, goal_node, heuristic_func) > _d_star_pq[0][0]) \
          or start_node.g != start_node.rhs:

        if not _d_star_pq: break # Should not happen if start is reachable and inconsistent

        k_old_popped, u_node = heapq.heappop(_d_star_pq)

        if u_node not in _d_star_open_set_tracker and u_node.g == u_node.rhs : # Already processed and consistent or explicitly removed
            # This check helps ignore outdated entries if node was re-added with better key
            # or became consistent and was conceptually removed from open set.
            continue

        # If it was in open_set_tracker, it's now being processed from PQ, so remove from conceptual open set.
        if u_node in _d_star_open_set_tracker:
             _d_star_open_set_tracker.remove(u_node)

        processed_nodes_for_viz.append(u_node)

        k_new_recalc = calculate_d_star_key(u_node, start_node, goal_node, heuristic_func)

        if k_old_popped < k_new_recalc: # Condition 1: Node got worse or heuristic changed
            heapq.heappush(_d_star_pq, (k_new_recalc, u_node))
            _d_star_open_set_tracker.add(u_node)
        elif u_node.g > u_node.rhs: # Condition 2: Locally overconsistent (found a better way to u_node)
            u_node.g = u_node.rhs
            for pred_node in u_node.neighbors: # Predecessors are neighbors
                d_star_lite_update_node(pred_node, grid, start_node, goal_node, heuristic_func)
        else: # Condition 3: Locally underconsistent (path through u_node got worse)
            g_old = u_node.g
            u_node.g = float('inf')
            # Update current node u_node itself and its predecessors
            # Order might matter: update predecessors first, then current node
            for pred_node in u_node.neighbors:
                if pred_node != goal_node: # Goal's rhs is 0 by definition
                     # If pred_node.rhs was based on g_old of u_node, it needs update
                     # This implies re-evaluating pred_node.rhs, which update_node does.
                     if pred_node.rhs == (g_old + 1): # Heuristic check if it was indeed through u
                          if pred_node != goal_node:
                            min_rhs = float('inf')
                            for succ_of_pred in pred_node.neighbors:
                                min_rhs = min(min_rhs, succ_of_pred.g + 1)
                            pred_node.rhs = min_rhs
                # And then, if inconsistent, it will be added to PQ by update_node
                if pred_node.g != pred_node.rhs and pred_node not in _d_star_open_set_tracker:
                     key_pred = calculate_d_star_key(pred_node, start_node, goal_node, heuristic_func)
                     heapq.heappush(_d_star_pq, (key_pred, pred_node))
                     _d_star_open_set_tracker.add(pred_node)
                elif pred_node.g == pred_node.rhs and pred_node in _d_star_open_set_tracker:
                    _d_star_open_set_tracker.remove(pred_node)


            # Update current node u_node itself (its g changed to inf)
            # This will re-evaluate its rhs and add to PQ if inconsistent
            if u_node.g != u_node.rhs and u_node not in _d_star_open_set_tracker:
                 key_u = calculate_d_star_key(u_node, start_node, goal_node, heuristic_func)
                 heapq.heappush(_d_star_pq, (key_u, u_node))
                 _d_star_open_set_tracker.add(u_node)
            elif u_node.g == u_node.rhs and u_node in _d_star_open_set_tracker:
                _d_star_open_set_tracker.remove(u_node)


    # Path Reconstruction
    path = []
    if start_node.g == float('inf'):
        print("No path found by D* Lite.")
        return [], processed_nodes_for_viz, list(_d_star_open_set_tracker)

    curr = start_node
    path.append(curr)
    while curr != goal_node:
        min_cost = float('inf')
        next_node = None
        if not curr.neighbors: # Should not happen if goal is reachable
            print("Error in path reconstruction: current node has no neighbors.")
            return [], processed_nodes_for_viz, list(_d_star_open_set_tracker)

        for neighbor in curr.neighbors:
            cost = 1 # Cost to move to neighbor
            # Path is built by moving to successor s' that has lowest g(s') + cost(curr, s')
            # This is actually just finding the neighbor with the smallest g value.
            if neighbor.g < min_cost: # This should be g_val + cost, but cost is 1 and we need lowest g
                min_cost = neighbor.g
                next_node = neighbor

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
    start_node.g = 0 # Use .g
    start_node.h_score = heuristic(start_node, end_node)
    start_node.f_score = start_node.g + start_node.h_score # Use .g

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
            temp_g = current_node.g + 1 # Use .g

            if temp_g < neighbor.g: # Use .g
                neighbor.previous_node = current_node
                neighbor.g = temp_g # Use .g
                neighbor.h_score = heuristic(neighbor, end_node)
                neighbor.f_score = neighbor.g + neighbor.h_score # Use .g

                # Add to PQ even if already in open_set_tracker (i.e. in pq).
                # The check `if current_node not in open_set_tracker` at loop start
                # handles cases where a node is pulled from PQ after a shorter path to it was already found and processed.
                heapq.heappush(pq, (neighbor.f_score, neighbor))
                open_set_tracker.add(neighbor) # Ensure it's marked as "conceptually" open

    # print("No path found by A*.") # Silenced for tests
    final_open_set = list(open_set_tracker)
    return [], visited_nodes_in_order, final_open_set

```
