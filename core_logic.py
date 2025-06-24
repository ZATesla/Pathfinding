import heapq

# Note: Pygame specific constants (WINDOW_WIDTH, colors, etc.) are NOT here.
# CELL_WIDTH/HEIGHT are passed to Node but not strictly needed for core logic if x,y are for GUI.

class Node:
    """
    Represents a single cell/node in the grid used for pathfinding.

    Each node stores its position, attributes relevant to pathfinding algorithms
    (like g, h, f scores, terrain cost), and flags for GUI visualization.
    It also maintains a list of its walkable neighbors.
    """
    def __init__(self, row: int, col: int, total_rows: int, total_cols: int,
                 width: int = 0, height: int = 0):
        """
        Initializes a Node instance.

        Args:
            row: The row index of the node in the grid.
            col: The column index of the node in the grid.
            total_rows: The total number of rows in the grid, used for boundary checks.
            total_cols: The total number of columns in the grid, used for boundary checks.
            width: The width of the node in pixels (for GUI). Defaults to 0.
                   If not for GUI, x, y, width, height might not be strictly needed by core logic.
            height: The height of the node in pixels (for GUI). Defaults to 0.
        """
        self.row = row
        self.col = col
        self.x = col * width  # Pixel x-coordinate (top-left)
        self.y = row * height # Pixel y-coordinate (top-left)
        self.width = width    # Width for GUI representation
        self.height = height  # Height for GUI representation

        self.is_obstacle = False    # True if this node is an impassable wall
        self.terrain_cost = 1.0     # Multiplier for movement cost onto this node

        # Pathfinding algorithm scores
        self.g = float('inf')       # Cost from start node to this node
        self.rhs = float('inf')     # D* Lite: right-hand side value (an estimate of g-score)
        self.h_score = float('inf') # Heuristic: estimated cost from this node to the end node
        self.f_score = float('inf') # Total estimated cost (e.g., g + h_score for A*)

        self.previous_node: Node | None = None # Node from which this node was reached in the path
        self.neighbors: list[Node] = []        # List of adjacent walkable nodes

        self.total_rows = total_rows # Total rows in the grid
        self.total_cols = total_cols # Total columns in the grid

        # Flags for GUI visualization
        self.is_visited_by_algorithm = False # True if in the "closed set" (generic)
        self.is_in_open_set_for_algorithm = False # True if in the "open set" (generic)
        self.is_part_of_path = False         # True if part of the final reconstructed path

        # Visualization flags specific to Bidirectional Search
        self.is_in_open_set_fwd = False      # Bidirectional: in forward search's open set
        self.is_in_closed_set_fwd = False    # Bidirectional: in forward search's closed set
        self.is_in_open_set_bwd = False      # Bidirectional: in backward search's open set
        self.is_in_closed_set_bwd = False    # Bidirectional: in backward search's closed set

    def __lt__(self, other: 'Node') -> bool:
        """
        Compares this node to another node for priority queue ordering.
        Primarily based on f_score. Used by heapq.
        """
        return self.f_score < other.f_score

    def get_pos(self) -> tuple[int, int]:
        """Returns the (row, col) grid position of the node."""
        return self.row, self.col

    def reset_algorithm_attributes(self):
        """
        Resets attributes of the node that are specific to a single run of a
        pathfinding algorithm. This prepares the node for a new search.
        """
        self.g = float('inf')
        self.rhs = float('inf')
        self.h_score = float('inf')
        self.f_score = float('inf')
        self.previous_node = None
        self.is_visited_by_algorithm = False
        self.is_in_open_set_for_algorithm = False
        self.is_part_of_path = False

        # Reset bidirectional flags as well
        self.is_in_open_set_fwd = False
        self.is_in_closed_set_fwd = False
        self.is_in_open_set_bwd = False
        self.is_in_closed_set_bwd = False

    def add_neighbors(self, grid_nodes_matrix: list[list['Node']], allow_diagonal: bool = True):
        """
        Populates the `self.neighbors` list with valid, walkable adjacent nodes.

        This method checks cardinal (Up, Down, Left, Right) and optionally
        diagonal neighbors. For diagonal movement, it includes a check to
        prevent "cutting corners" between two diagonally adjacent obstacles.
        A diagonal neighbor is only added if the diagonal cell itself is walkable
        AND at least one of the two cardinal cells that form the corner with the
        current node and the diagonal node is also walkable.

        Args:
            grid_nodes_matrix: The 2D list representing all nodes in the grid.
            allow_diagonal: Boolean indicating if diagonal neighbors should be considered.
        """
        self.neighbors = []
        # Cardinal directions (Up, Down, Left, Right)
        if self.row < self.total_rows - 1 and not grid_nodes_matrix[self.row + 1][self.col].is_obstacle: # Move DOWN
            self.neighbors.append(grid_nodes_matrix[self.row + 1][self.col])
        if self.row > 0 and not grid_nodes_matrix[self.row - 1][self.col].is_obstacle: # Move UP
            self.neighbors.append(grid_nodes_matrix[self.row - 1][self.col])
        if self.col < self.total_cols - 1 and not grid_nodes_matrix[self.row][self.col + 1].is_obstacle: # Move RIGHT
            self.neighbors.append(grid_nodes_matrix[self.row][self.col + 1])
        if self.col > 0 and not grid_nodes_matrix[self.row][self.col - 1].is_obstacle: # Move LEFT
            self.neighbors.append(grid_nodes_matrix[self.row][self.col - 1])

        if allow_diagonal:
            # Diagonal directions
            # Check if diagonal cell itself is not an obstacle AND
            # at least one of the two adjacent cardinal cells (that form the corner) is also not an obstacle.
            # This prevents "cutting corners" through two diagonally adjacent obstacles.

            # Down-Right (DR)
            if self.row < self.total_rows - 1 and self.col < self.total_cols - 1 and \
               not grid_nodes_matrix[self.row + 1][self.col + 1].is_obstacle: # Target DR cell
                # Check if Down or Right is clear to avoid cutting a corner
                if not grid_nodes_matrix[self.row + 1][self.col].is_obstacle or \
                   not grid_nodes_matrix[self.row][self.col + 1].is_obstacle:
                    self.neighbors.append(grid_nodes_matrix[self.row + 1][self.col + 1])

            # Down-Left (DL)
            if self.row < self.total_rows - 1 and self.col > 0 and \
               not grid_nodes_matrix[self.row + 1][self.col - 1].is_obstacle: # Target DL cell
                # Check if Down or Left is clear
                if not grid_nodes_matrix[self.row + 1][self.col].is_obstacle or \
                   not grid_nodes_matrix[self.row][self.col - 1].is_obstacle:
                    self.neighbors.append(grid_nodes_matrix[self.row + 1][self.col - 1])

            # Up-Right (UR)
            if self.row > 0 and self.col < self.total_cols - 1 and \
               not grid_nodes_matrix[self.row - 1][self.col + 1].is_obstacle: # Target UR cell
                # Check if Up or Right is clear
                if not grid_nodes_matrix[self.row - 1][self.col].is_obstacle or \
                   not grid_nodes_matrix[self.row][self.col + 1].is_obstacle:
                    self.neighbors.append(grid_nodes_matrix[self.row - 1][self.col + 1])

            # Up-Left (UL)
            if self.row > 0 and self.col > 0 and \
               not grid_nodes_matrix[self.row - 1][self.col - 1].is_obstacle: # Target UL cell
                # Check if Up or Left is clear
                if not grid_nodes_matrix[self.row - 1][self.col].is_obstacle or \
                   not grid_nodes_matrix[self.row][self.col - 1].is_obstacle:
                    self.neighbors.append(grid_nodes_matrix[self.row - 1][self.col - 1])

class Grid:
    """
    Manages the 2D grid of Node objects and global grid properties such as
    start/end nodes and whether diagonal movement is permitted. It provides
    methods to update and interact with the grid and its nodes.
    """
    def __init__(self, rows: int, cols: int, cell_width_for_nodes: int = 0, cell_height_for_nodes: int = 0):
        """
        Initializes the grid, creating all Node objects and setting up their initial neighbor relationships.

        Args:
            rows: Number of rows in the grid.
            cols: Number of columns in the grid.
            cell_width_for_nodes: Width to assign to each Node for GUI purposes.
            cell_height_for_nodes: Height to assign to each Node for GUI purposes.
        """
        self.rows = rows
        self.cols = cols
        self.allow_diagonal_movement = True # Diagonal movement is allowed by default
        self.nodes = [[Node(r, c, rows, cols, cell_width_for_nodes, cell_height_for_nodes) for c in range(cols)] for r in range(rows)]
        self.start_node: Node | None = None
        self.end_node: Node | None = None
        self.update_all_node_neighbors() # Initialize neighbor lists for all nodes

    def update_all_node_neighbors(self):
        """Recalculates and updates the neighbors list for every node in the grid."""
        for r in range(self.rows):
            for c in range(self.cols):
                self.nodes[r][c].add_neighbors(self.nodes, self.allow_diagonal_movement)

    def set_allow_diagonal_movement(self, allow: bool):
        """
        Sets the flag for allowing diagonal movement and updates all node neighbors
        to reflect this change.
        """
        self.allow_diagonal_movement = allow
        self.update_all_node_neighbors()

    def reset_algorithm_states(self):
        """Resets algorithm-specific attributes (like g, h, f scores, previous_node) for all nodes."""
        for r in range(self.rows):
            for c in range(self.cols):
                self.nodes[r][c].reset_algorithm_attributes()

    def clear_visualizations(self):
        """Resets only the visualization-related attributes of all nodes (e.g., visited, open set, path flags)."""
        for r in range(self.rows):
            for c in range(self.cols):
                node = self.nodes[r][c]
                node.is_visited_by_algorithm = False
                node.is_in_open_set_for_algorithm = False
                node.is_part_of_path = False
                # Does not reset g, h, f, rhs, previous_node, is_obstacle, terrain_cost

    def get_node(self, row: int, col: int) -> Node | None:
        """
        Retrieves a node from the grid at the specified row and column.

        Args:
            row: The row index.
            col: The column index.

        Returns:
            The Node object if coordinates are valid, otherwise None.
        """
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.nodes[row][col]
        return None

    def set_target_node(self, row: int, col: int) -> bool:
        """
        Sets the grid's end/target node to the specified row and column.
        Used primarily by D* Lite for dynamic target changes.

        Args:
            row: The row index for the new target.
            col: The column index for the new target.

        Returns:
            True if the target was successfully set, False otherwise (e.g., out of bounds or obstacle).
        """
        if 0 <= row < self.rows and 0 <= col < self.cols:
            new_target_node = self.nodes[row][col]
            if not new_target_node.is_obstacle:
                self.end_node = new_target_node
                print(f"Target node set to ({row}, {col})")
                return True
            else:
                print(f"Cannot set target at ({row}, {col}) as it's an obstacle.")
                return False
        else:
            print(f"Target coordinates ({row}, {col}) are out of bounds.")
            return False

# Cost constants for movement
COST_CARDINAL = 1.0  # Cost for moving to adjacent cardinal cells (Up, Down, Left, Right)
COST_DIAGONAL = 1.41421356  # Approximate sqrt(2), cost for moving to adjacent diagonal cells

def get_move_cost(node1: Node, node2: Node) -> float:
    """
    Calculates the cost of moving from node1 to an adjacent node2.
    Considers if the movement is cardinal or diagonal and applies the terrain cost of the destination node (node2).

    Args:
        node1: The starting node.
        node2: The destination (neighboring) node.

    Returns:
        The calculated movement cost.
    """
    base_cost = COST_CARDINAL
    # Check for diagonal movement (difference in both row and col is 1)
    if abs(node1.row - node2.row) == 1 and abs(node1.col - node2.col) == 1:
        base_cost = COST_DIAGONAL
    return base_cost * node2.terrain_cost

def heuristic(node_a: Node, node_b: Node, allow_diagonal: bool = True) -> float:
    """
    Calculates the heuristic (estimated cost) between two nodes.
    Uses Octile distance if diagonal movement is allowed, otherwise Manhattan distance.

    Args:
        node_a: The first node.
        node_b: The second node (typically the goal node).
        allow_diagonal: If True, uses Octile distance. Otherwise, uses Manhattan distance.

    Returns:
        The estimated heuristic cost.
    """
    dx = abs(node_a.col - node_b.col)  # Change in x (columns)
    dy = abs(node_a.row - node_b.row)  # Change in y (rows)

    if allow_diagonal:
        # Octile distance: D2 * min(dx, dy) + D1 * (max(dx, dy) - min(dx, dy))
        return COST_DIAGONAL * min(dx, dy) + COST_CARDINAL * (max(dx, dy) - min(dx, dy))
    else:
        # Manhattan distance: D1 * (dx + dy)
        return COST_CARDINAL * (dx + dy)

def dijkstra(grid: Grid, start_node: Node, end_node: Node) -> tuple[list[Node], list[Node], list[Node]]:
    """
    Performs Dijkstra's algorithm to find the shortest path from start_node to end_node.
    It explores paths based on accumulated cost (g-score) without using a heuristic.

    Args:
        grid: The Grid object containing all nodes.
        start_node: The starting node for the path.
        end_node: The target/goal node.

    Returns:
        A tuple (path, visited_nodes_in_order, open_set_final_nodes):
            - path: A list of Node objects representing the shortest path, or an empty list if no path found.
            - visited_nodes_in_order: A list of nodes in the order they were visited (expanded).
            - open_set_final_nodes: A list of nodes remaining in the open set when the algorithm terminated.
    """
    if not start_node or not end_node or start_node.is_obstacle or end_node.is_obstacle:
        return [], [], []
    grid.reset_algorithm_states()

    visited_nodes_in_order = []
    start_node.g = 0
    start_node.h_score = 0 # Not used by Dijkstra, but reset for consistency
    start_node.f_score = start_node.g # f_score is effectively g_score

    pq = [(start_node.f_score, start_node)]
    open_set_tracker = {start_node}

    while pq:
        _, current_node = heapq.heappop(pq)

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

        for neighbor in current_node.neighbors:
            cost = get_move_cost(current_node, neighbor)
            temp_g = current_node.g + cost

            if temp_g < neighbor.g:
                neighbor.previous_node = current_node
                neighbor.g = temp_g
                neighbor.f_score = neighbor.g
                if neighbor not in open_set_tracker:
                    heapq.heappush(pq, (neighbor.f_score, neighbor))
                    open_set_tracker.add(neighbor)

    return [], visited_nodes_in_order, list(open_set_tracker)

def calculate_d_star_key(node: Node, start_node_for_h: Node, goal_node_for_search: Node,
                         heuristic_func, allow_diagonal: bool) -> tuple[float, float]:
    """
    Calculates the D* Lite key for a node, used for priority queue ordering.
    The key is (min(g, rhs) + h(node, start_node_for_h), min(g, rhs)).
    The heuristic h is calculated from the current node to the actual start of the path.

    Args:
        node: The node for which to calculate the key.
        start_node_for_h: The actual start node of the pathfinding problem (used for h-value).
        goal_node_for_search: The goal node of the pathfinding problem (D* Lite's search starts here,
                              but this param is mostly for context in this specific key calculation).
        heuristic_func: The heuristic function (e.g., heuristic).
        allow_diagonal: Boolean for heuristic calculation.

    Returns:
        A tuple representing the key (k1, k2).
    """
    # h_val is heuristic from 'node' to the path's actual start node 'start_node_for_h'
    h_val = heuristic_func(node, start_node_for_h, allow_diagonal)
    min_g_rhs = min(node.g, node.rhs)
    return (min_g_rhs + h_val, min_g_rhs)

def d_star_lite_initialize(grid: Grid, path_start_node: Node, path_goal_node: Node,
                           heuristic_func, pq: list, open_set_tracker: set):
    """
    Initializes D* Lite algorithm. Resets grid states, sets the path_goal_node's rhs to 0,
    and adds it to the priority queue. The D* Lite search effectively works backward
    from the path_goal_node.

    Args:
        grid: The main grid object.
        path_start_node: The start node of the overall pathfinding problem.
        path_goal_node: The goal node of the overall pathfinding problem (where D* Lite starts its calculations).
        heuristic_func: The heuristic function.
        pq: The priority queue (list to be used with heapq).
        open_set_tracker: A set to track nodes in the priority queue.
    """
    grid.reset_algorithm_states()
    path_goal_node.rhs = 0 # Cost to reach goal from goal is 0
    # Key calculation uses path_start_node for the heuristic component.
    key = calculate_d_star_key(path_goal_node, path_start_node, path_goal_node, heuristic_func, grid.allow_diagonal_movement)
    heapq.heappush(pq, (key, path_goal_node))
    open_set_tracker.add(path_goal_node)

def d_star_lite_update_node(node: Node, grid: Grid, path_start_node: Node, path_goal_node: Node,
                            heuristic_func, pq: list, open_set_tracker: set):
    """
    Updates a node's rhs value and its status in the priority queue for D* Lite.
    This is called when costs change or to propagate updates.

    Args:
        node: The node to update.
        grid: The main grid object.
        path_start_node: The start node of the overall pathfinding problem.
        path_goal_node: The goal node of the overall pathfinding problem.
        heuristic_func: The heuristic function.
        pq: The priority queue.
        open_set_tracker: Set tracking nodes in the priority queue.
    """
    if node != path_goal_node:
        min_rhs = float('inf')
        # rhs(u) = min_{s' in Succ(u)} (c(u,s') + g(s'))
        # Here, Succ(u) are the standard neighbors of u.
        for successor in node.neighbors:
            cost = get_move_cost(node, successor)
            min_rhs = min(min_rhs, successor.g + cost)
        node.rhs = min_rhs

    # If node was in open_set_tracker (PQ) and is consistent, it might be removed.
    # However, D* Lite often relies on checks during pop rather than explicit removal.
    # if node in open_set_tracker and node.g == node.rhs:
    #    pass # Potentially remove from PQ if an efficient way exists.

    if node.g != node.rhs: # Node is inconsistent, add/update in PQ
        key = calculate_d_star_key(node, path_start_node, path_goal_node, heuristic_func, grid.allow_diagonal_movement)
        heapq.heappush(pq, (key, node))
        open_set_tracker.add(node)
    elif node in open_set_tracker: # Node is consistent (g == rhs) but was in PQ
        # This implies it was already processed or its key is up-to-date.
        # Standard D* Lite might remove it here if it's truly consistent and doesn't need re-expansion.
        # For simplicity with heapq, we rely on the check when popping from PQ.
        # To explicitly remove, one would mark it or use a PQ supporting removal.
        # For now, if it's consistent and in open_set_tracker, we assume it's there with an old, higher key.
        # If it's popped and still consistent, it will be skipped.
        # If its key needs recalculation and it's consistent, it shouldn't be in PQ unless g or rhs changed to make it inconsistent.
        # This logic implies that if a node is consistent, it should not be in the open_set_tracker unless it's the goal node or start node being evaluated.
        # A common approach: if node.g == node.rhs and node in open_set_tracker: open_set_tracker.remove(node) (if PQ supports remove by value)
        pass


def d_star_lite_compute_shortest_path(grid: Grid, path_start_node: Node, path_goal_node: Node,
                                      heuristic_func, pq: list, open_set_tracker: set
                                      ) -> tuple[list[Node], list[Node], list[Node]]:
    """
    Computes the shortest path in D* Lite by processing nodes from the priority queue
    until the path_start_node is consistent (g == rhs) and its key is not less than
    the minimum key in the priority queue.

    Args:
        grid: The main grid object.
        path_start_node: The start node of the pathfinding problem.
        path_goal_node: The goal node of the pathfinding problem.
        heuristic_func: The heuristic function.
        pq: The priority queue.
        open_set_tracker: Set tracking nodes in the priority queue.

    Returns:
        A tuple (path, visited_nodes_in_order, open_set_final_nodes):
            - path: List of nodes forming the path from path_start_node to path_goal_node.
            - visited_nodes_in_order: List of nodes expanded during the search.
            - open_set_final_nodes: List of nodes remaining in the open set.
    """
    processed_nodes_for_viz = []
    # Loop while path_start_node's key is greater than the top key in PQ, OR path_start_node is inconsistent.
    while (pq and calculate_d_star_key(path_start_node, path_start_node, path_goal_node, heuristic_func, grid.allow_diagonal_movement) > pq[0][0]) \
          or path_start_node.g != path_start_node.rhs:
        if not pq: break

        k_old_popped, u_node = heapq.heappop(pq)

        # If u_node was removed from open_set_tracker (e.g. became consistent and removed)
        # or if its g=rhs (consistent) and it's popped, but a better entry for it exists or it was already processed.
        # This check helps handle duplicate entries in PQ if a node's key was updated.
        if u_node not in open_set_tracker: # Node was already processed and removed from open_set_tracker
            continue

        k_new_recalc = calculate_d_star_key(u_node, path_start_node, path_goal_node, heuristic_func, grid.allow_diagonal_movement)

        if k_old_popped < k_new_recalc: # Key increased (e.g. obstacle added, path became costlier)
            heapq.heappush(pq, (k_new_recalc, u_node)) # Re-add with new, higher key
            # open_set_tracker already contains u_node if it was just popped with an old key
        elif u_node.g > u_node.rhs: # Overconsistent node (g > rhs), an improved path to goal found via this node
            u_node.g = u_node.rhs # Make it consistent; this is the new best g-score (cost to goal)
            open_set_tracker.remove(u_node) # Now processed and consistent
            processed_nodes_for_viz.append(u_node)
            # Update predecessors of u_node (nodes u' for which u is a successor s')
            # In grid terms, these are standard neighbors of u_node.
            for pred_node_candidate in u_node.neighbors:
                d_star_lite_update_node(pred_node_candidate, grid, path_start_node, path_goal_node, heuristic_func, pq, open_set_tracker)
        else: # Underconsistent (g < rhs), path through this node became more expensive or non-existent
            g_old = u_node.g
            u_node.g = float('inf') # Set g to infinity, path through here is currently invalid/unknown
            open_set_tracker.remove(u_node) # Processed with this g=inf state
            processed_nodes_for_viz.append(u_node)
            # Update predecessors of u_node and u_node itself (to re-evaluate its rhs)
            for pred_node_candidate in u_node.neighbors:
                d_star_lite_update_node(pred_node_candidate, grid, path_start_node, path_goal_node, heuristic_func, pq, open_set_tracker)
            d_star_lite_update_node(u_node, grid, path_start_node, path_goal_node, heuristic_func, pq, open_set_tracker) # Update u_node itself

    # Path Reconstruction for D* Lite
    path = []
    if path_start_node.g == float('inf'):
        print("No path found by D* Lite.")
        return [], processed_nodes_for_viz, list(open_set_tracker)

    curr = path_start_node
    path.append(curr)
    while curr != path_goal_node:
        min_cost_plus_g = float('inf')
        next_node_on_path = None
        if not curr.neighbors:
            print(f"Error in D* Lite path reconstruction: node {curr.get_pos()} has no neighbors.")
            return [], processed_nodes_for_viz, list(open_set_tracker)

        for neighbor_succ in curr.neighbors: # Successors of current node on the path
            cost_curr_to_succ = get_move_cost(curr, neighbor_succ)
            # Choose successor s' that minimizes c(curr, s') + g(s')
            if cost_curr_to_succ + neighbor_succ.g < min_cost_plus_g:
                min_cost_plus_g = cost_curr_to_succ + neighbor_succ.g
                next_node_on_path = neighbor_succ
            # Tie-breaking: if costs are equal, could add logic (e.g., prefer straight)
            # For now, first one found with min cost is taken.

        if next_node_on_path is None :
             print(f"D* Lite path reconstruction failed: No valid next node from {curr.get_pos()} (g={curr.g}, rhs={curr.rhs})")
             return [], processed_nodes_for_viz, list(open_set_tracker)

        curr = next_node_on_path
        path.append(curr)
        if len(path) > grid.rows * grid.cols:
            print("D* Lite path reconstruction exceeded max length (possible loop).")
            return [], processed_nodes_for_viz, list(open_set_tracker)

    return path, processed_nodes_for_viz, list(open_set_tracker)

def run_d_star_lite(grid: Grid, start_node: Node, goal_node: Node, heuristic_func) -> tuple[list[Node], list[Node], list[Node]]:
    """
    Main function to run the D* Lite algorithm for an initial path computation.
    It initializes the algorithm and then computes the shortest path.
    This function manages the persistent priority queue (pq) and open_set_tracker
    if D* Lite were to be used for subsequent replanning without full re-initialization.
    However, for this project, main_gui.py handles the persistence if needed.

    Args:
        grid: The grid object.
        start_node: The starting node for the path.
        goal_node: The target node for the path.
        heuristic_func: The heuristic function to be used (e.g., core_logic.heuristic).

    Returns:
        A tuple (path, visited_nodes, open_set_nodes) as from d_star_lite_compute_shortest_path.
    """
    if not start_node or not goal_node or start_node.is_obstacle or goal_node.is_obstacle:
        print("D* Lite: Start or Goal is invalid or an obstacle.")
        return [], [], []

    pq_instance: list = []
    open_set_tracker_instance: set = set()

    # Ensure the grid object has the correct overall start and end nodes set.
    # These are used by calculate_d_star_key for consistent heuristic calculations.
    grid.start_node = start_node
    grid.end_node = goal_node

    d_star_lite_initialize(grid, start_node, goal_node, heuristic_func, pq_instance, open_set_tracker_instance)
    path, visited_nodes, open_set_final_nodes = d_star_lite_compute_shortest_path(
        grid, start_node, goal_node, heuristic_func, pq_instance, open_set_tracker_instance
    )
    return path, visited_nodes, open_set_final_nodes

def d_star_lite_obstacle_change_update(grid: Grid, r_changed: int, c_changed: int,
                                       pq: list, open_set_tracker: set,
                                       path_start_node: Node, path_goal_node: Node, heuristic_func):
    """
    Handles updates in D* Lite when an obstacle's status changes in the grid.
    This function updates the affected node and its neighbors, potentially adding them
    to the priority queue for replanning via `d_star_lite_compute_shortest_path`.

    Args:
        grid: The grid object.
        r_changed: Row of the node whose obstacle status changed.
        c_changed: Column of the node whose obstacle status changed.
        pq: The persistent priority queue from the D* Lite instance (managed in main_gui).
        open_set_tracker: The persistent open set tracker (managed in main_gui).
        path_start_node: The overall start node of the path.
        path_goal_node: The overall goal node of the path.
        heuristic_func: The heuristic function.
    """
    changed_node = grid.get_node(r_changed, c_changed)
    if not changed_node:
        print(f"Error in d_star_lite_obstacle_change_update: Node ({r_changed},{c_changed}) not found.")
        return

    # Obstacle status has changed, so neighbor relationships for all nodes might change.
    grid.update_all_node_neighbors()

    # Update the node that changed and its direct neighbors.
    # The change in obstacle status directly affects the changed_node's traversability
    # and the costs of edges to/from it.
    # D* Lite updates propagate from these initial changes.

    # Update the node whose obstacle status changed.
    # This will re-evaluate its rhs if it's not the goal and add it to PQ if inconsistent.
    d_star_lite_update_node(changed_node, grid, path_start_node, path_goal_node, heuristic_func, pq, open_set_tracker)

    # Also update all immediate neighbors of the changed node, as their rhs values
    # might change if their path to the goal was through changed_node, or if changed_node
    # becoming an obstacle/clear affects their edge costs to it.
    for neighbor_of_changed in changed_node.neighbors: # Use current neighbors after update_all_node_neighbors
         d_star_lite_update_node(neighbor_of_changed, grid, path_start_node, path_goal_node, heuristic_func, pq, open_set_tracker)

    print(f"D* Lite: Processed obstacle change at ({r_changed},{c_changed}). Call compute_shortest_path to replan.")

def d_star_lite_target_move_update(grid: Grid, new_path_goal_node: Node, old_path_goal_node: Node | None,
                                   pq: list, open_set_tracker: set,
                                   path_start_node: Node, heuristic_func):
    """
    Handles updates in D* Lite when the target/goal node moves.
    The old goal node is updated (its rhs might change as it's no longer the goal),
    and the new goal node is initialized (rhs = 0).

    Args:
        grid: The grid object.
        new_path_goal_node: The new goal node for the path.
        old_path_goal_node: The previous goal node (can be None if this is the first goal).
        pq: The persistent priority queue (managed in main_gui).
        open_set_tracker: The persistent open set tracker (managed in main_gui).
        path_start_node: The overall start node of the path.
        heuristic_func: The heuristic function.
    """
    # Update the old goal node: it's no longer the goal, so its rhs might change.
    # Setting rhs to infinity and updating it effectively removes its special status.
    if old_path_goal_node and old_path_goal_node != new_path_goal_node:
        old_path_goal_node.rhs = float('inf')
        d_star_lite_update_node(old_path_goal_node, grid, path_start_node, new_path_goal_node, heuristic_func, pq, open_set_tracker)

    grid.end_node = new_path_goal_node # Update the grid's reference to the current goal
    new_path_goal_node.rhs = 0 # The new goal's rhs (cost to itself) is 0
    d_star_lite_update_node(new_path_goal_node, grid, path_start_node, new_path_goal_node, heuristic_func, pq, open_set_tracker)

    print(f"D* Lite: Processed target move to ({new_path_goal_node.row},{new_path_goal_node.col}). Call compute_shortest_path to replan.")

def a_star(grid: Grid, start_node: Node, end_node: Node) -> tuple[list[Node], list[Node], list[Node]]:
    """
    Performs A* algorithm to find the shortest path from start_node to end_node.
    A* uses a heuristic to guide its search towards the goal.

    Args:
        grid: The Grid object.
        start_node: The starting Node.
        end_node: The target/goal Node.

    Returns:
        A tuple (path, visited_nodes_in_order, open_set_final_nodes):
            - path: List of Node objects representing the path, empty if no path.
            - visited_nodes_in_order: List of nodes in the order they were expanded.
            - open_set_final_nodes: List of nodes in the open set upon termination.
    """
    if not start_node or not end_node or start_node.is_obstacle or end_node.is_obstacle:
        return [], [], []
    grid.reset_algorithm_states()

    visited_nodes_in_order = []
    start_node.g = 0
    start_node.h_score = heuristic(start_node, end_node, grid.allow_diagonal_movement)
    start_node.f_score = start_node.g + start_node.h_score

    pq = [(start_node.f_score, start_node.h_score, start_node)] # Use h_score as tie-breaker
    open_set_tracker = {start_node: start_node.g} # Store g_scores for efficient update checks

    while pq:
        _, _, current_node = heapq.heappop(pq) # f_score and h_score are implicitly used by heapq

        # If this entry in PQ is for an outdated path to current_node (g_score is higher than already found)
        if current_node not in open_set_tracker or open_set_tracker[current_node] < current_node.g:
            continue

        # If we are processing this node, remove it from conceptual open set for this path.
        # (A node could be re-added if a shorter path is found later, but this specific expansion is done)
        # For this simple A*, once popped and g_score matches tracker, it's "closed".
        if current_node in open_set_tracker: # Should always be true if not skipped above
            del open_set_tracker[current_node]

        visited_nodes_in_order.append(current_node)

        if current_node == end_node:
            path = []
            temp = end_node
            while temp:
                path.append(temp)
                temp = temp.previous_node
            return path[::-1], visited_nodes_in_order, [n for n in open_set_tracker.keys()] # Nodes remaining in open set

        for neighbor in current_node.neighbors:
            cost = get_move_cost(current_node, neighbor)
            temp_g = current_node.g + cost

            if temp_g < neighbor.g:
                neighbor.previous_node = current_node
                neighbor.g = temp_g
                neighbor.h_score = heuristic(neighbor, end_node, grid.allow_diagonal_movement)
                neighbor.f_score = neighbor.g + neighbor.h_score
                # Add to open set if not already there with a better or equal G score
                if neighbor not in open_set_tracker or open_set_tracker[neighbor] > neighbor.g:
                    heapq.heappush(pq, (neighbor.f_score, neighbor.h_score, neighbor))
                    open_set_tracker[neighbor] = neighbor.g

    return [], visited_nodes_in_order, [n for n in open_set_tracker.keys()]

def bidirectional_search(grid: Grid, start_node: Node, end_node: Node
                         ) -> tuple[list[Node], list[Node], list[Node], list[Node], list[Node]]:
    """
    Performs a Bidirectional Search (Dijkstra-based from both ends) to find the shortest path.
    It searches simultaneously from the start and end nodes until the two search frontiers meet.

    Args:
        grid: The Grid object.
        start_node: The starting Node.
        end_node: The target Node.

    Returns:
        A tuple (final_path, visited_fwd, visited_bwd, open_fwd, open_bwd):
            - final_path: List of Node objects for the path. Empty if no path.
            - visited_fwd: Nodes visited by the forward search.
            - visited_bwd: Nodes visited by the backward search.
            - open_fwd: Nodes in the open set of the forward search at termination.
            - open_bwd: Nodes in the open set of the backward search at termination.
    """
    if not start_node or not end_node or start_node.is_obstacle or end_node.is_obstacle:
        return [], [], [], [], []
    grid.reset_algorithm_states()

    g_fwd = {node: float('inf') for row_nodes in grid.nodes for node in row_nodes}
    g_bwd = {node: float('inf') for row_nodes in grid.nodes for node in row_nodes}
    prev_fwd = {node: None for row_nodes in grid.nodes for node in row_nodes}
    prev_bwd = {node: None for row_nodes in grid.nodes for node in row_nodes}

    g_fwd[start_node] = 0
    g_bwd[end_node] = 0

    pq_fwd = [(0, start_node)]
    pq_bwd = [(0, end_node)]

    open_set_tracker_fwd = {start_node}
    open_set_tracker_bwd = {end_node}

    closed_set_fwd = set()
    closed_set_bwd = set()

    visited_nodes_in_order_fwd = []
    visited_nodes_in_order_bwd = []

    meeting_node: Node | None = None
    path_cost = float('inf')

    while pq_fwd and pq_bwd:
        if pq_fwd:
            _, current_node_fwd = heapq.heappop(pq_fwd)
            if current_node_fwd not in open_set_tracker_fwd: continue
            open_set_tracker_fwd.remove(current_node_fwd)
            closed_set_fwd.add(current_node_fwd)
            visited_nodes_in_order_fwd.append(current_node_fwd)

            if current_node_fwd in closed_set_bwd:
                current_path_cost_through_node = g_fwd[current_node_fwd] + g_bwd[current_node_fwd]
                if current_path_cost_through_node < path_cost:
                    path_cost = current_path_cost_through_node
                    meeting_node = current_node_fwd
                if pq_fwd and pq_bwd and (pq_fwd[0][0] + pq_bwd[0][0] >= path_cost):
                     break

            for neighbor in current_node_fwd.neighbors:
                if neighbor in closed_set_fwd: continue
                cost = get_move_cost(current_node_fwd, neighbor)
                temp_g_fwd = g_fwd[current_node_fwd] + cost
                if temp_g_fwd < g_fwd[neighbor]:
                    g_fwd[neighbor] = temp_g_fwd
                    prev_fwd[neighbor] = current_node_fwd
                    heapq.heappush(pq_fwd, (temp_g_fwd, neighbor))
                    open_set_tracker_fwd.add(neighbor)

        if pq_bwd:
            _, current_node_bwd = heapq.heappop(pq_bwd)
            if current_node_bwd not in open_set_tracker_bwd: continue
            open_set_tracker_bwd.remove(current_node_bwd)
            closed_set_bwd.add(current_node_bwd)
            visited_nodes_in_order_bwd.append(current_node_bwd)

            if current_node_bwd in closed_set_fwd:
                current_path_cost_through_node = g_fwd[current_node_bwd] + g_bwd[current_node_bwd]
                if current_path_cost_through_node < path_cost:
                    path_cost = current_path_cost_through_node
                    meeting_node = current_node_bwd
                if pq_fwd and pq_bwd and (pq_fwd[0][0] + pq_bwd[0][0] >= path_cost):
                    break

            for neighbor in current_node_bwd.neighbors:
                if neighbor in closed_set_bwd: continue
                cost = get_move_cost(neighbor, current_node_bwd)
                temp_g_bwd = g_bwd[current_node_bwd] + cost
                if temp_g_bwd < g_bwd[neighbor]:
                    g_bwd[neighbor] = temp_g_bwd
                    prev_bwd[neighbor] = current_node_bwd
                    heapq.heappush(pq_bwd, (temp_g_bwd, neighbor))
                    open_set_tracker_bwd.add(neighbor)

        if not pq_fwd or not pq_bwd:
            break

    final_path = []
    if meeting_node:
        # Path from Start to Meeting Node
        path_s_to_m = []
        curr = meeting_node
        while curr:
            path_s_to_m.append(curr)
            curr = prev_fwd.get(curr) # Use .get for safety, though curr should be in prev_fwd if path exists
        path_s_to_m.reverse()
        final_path.extend(path_s_to_m)

        # Path from Meeting Node to End Node (excluding meeting_node itself if already added)
        if meeting_node != end_node:
            path_m_to_e_segment = []
            curr = end_node
            # Trace from end_node back towards meeting_node, using prev_bwd
            # prev_bwd[X] = Y means Y is the parent of X in the backward search tree (Y is closer to End)
            while curr and curr != meeting_node:
                path_m_to_e_segment.append(curr)
                parent_in_bwd_search = prev_bwd.get(curr)
                if parent_in_bwd_search is None and curr != end_node : # Should not happen if path to meeting_node exists from end
                    print(f"Bidirectional path reconstruction error: Disconnected path segment from end node {end_node.get_pos()} to meeting node {meeting_node.get_pos()} at {curr.get_pos()}.")
                    return [], visited_nodes_in_order_fwd, visited_nodes_in_order_bwd, \
                           list(open_set_tracker_fwd), list(open_set_tracker_bwd) # Return empty path on error
                curr = parent_in_bwd_search

            # path_m_to_e_segment is now [End, node_before_End, ..., node_after_MeetingNode]
            # Reverse it to get [node_after_MeetingNode, ..., End]
            final_path.extend(path_m_to_e_segment[::-1])

    return final_path, visited_nodes_in_order_fwd, visited_nodes_in_order_bwd, \
           list(open_set_tracker_fwd), list(open_set_tracker_bwd)


def _jps_is_walkable(grid: Grid, r: int, c: int) -> bool:
    """
    Helper for JPS: Checks if the given (r, c) coordinates are within grid bounds
    and the corresponding node is not an obstacle.
    """
    if not (0 <= r < grid.rows and 0 <= c < grid.cols):
        return False # Out of bounds
    return not grid.nodes[r][c].is_obstacle # Check obstacle status

def _jps_jump(grid: Grid, current_r: int, current_c: int, dr: int, dc: int,
              _start_node_alg: Node, end_node_alg: Node) -> Node | None:
    """
    Recursively searches for a "jump point" in a given direction (dr, dc) from (current_r, current_c).
    A jump point is a node that warrants further examination in JPS. It's either:
    1. The goal node.
    2. A node with "forced neighbors" (neighbors that must be considered due to obstacles).
    3. For diagonal moves, if a horizontal or vertical jump from the current step yields a jump point.

    Args:
        grid: The grid instance.
        current_r: Row of the node from which to jump.
        current_c: Column of the node from which to jump.
        dr: Row direction of the jump (-1, 0, or 1).
        dc: Column direction of the jump (-1, 0, or 1).
        _start_node_alg: The overall start node of the JPS algorithm (passed for context, largely unused here).
        end_node_alg: The overall end/goal node of the JPS algorithm.

    Returns:
        The jump point Node if one is found along this path, otherwise None.
    """
    next_r, next_c = current_r + dr, current_c + dc

    if not _jps_is_walkable(grid, next_r, next_c):
        return None

    next_node = grid.nodes[next_r][next_c]

    if next_node == end_node_alg:
        return next_node

    if dr != 0 and dc != 0:  # Moving diagonally
        if not grid.allow_diagonal_movement: return None

        # Check for forced neighbors (based on Harabor's paper conditions for diagonal moves)
        # Condition 1: (next_r-dr, next_c) is blocked AND (next_r-dr, next_c+dc) is open
        if not _jps_is_walkable(grid, next_r - dr, next_c) and _jps_is_walkable(grid, next_r - dr, next_c + dc):
            return next_node
        # Condition 2: (next_r, next_c-dc) is blocked AND (next_r+dr, next_c-dc) is open
        if not _jps_is_walkable(grid, next_r, next_c - dc) and _jps_is_walkable(grid, next_r + dr, next_c - dc):
            return next_node

        # Recursively jump horizontally and vertically from the current diagonal step.
        if _jps_jump(grid, next_r, next_c, dr, 0, _start_node_alg, end_node_alg):
            return next_node
        if _jps_jump(grid, next_r, next_c, 0, dc, _start_node_alg, end_node_alg):
            return next_node

        return _jps_jump(grid, next_r, next_c, dr, dc, _start_node_alg, end_node_alg)

    else:  # Moving cardinally
        if dr != 0:  # Moving vertically
            # Forced neighbor if (next_r, next_c+1) (right of next) is blocked AND (current_r, next_c+1) (right of current) is open
            if not _jps_is_walkable(grid, next_r, next_c + 1) and _jps_is_walkable(grid, current_r, next_c + 1):
                return next_node
            # Forced neighbor if (next_r, next_c-1) (left of next) is blocked AND (current_r, next_c-1) (left of current) is open
            if not _jps_is_walkable(grid, next_r, next_c - 1) and _jps_is_walkable(grid, current_r, next_c - 1):
                return next_node
        else:  # Moving horizontally
            # Forced neighbor if (next_r+1, next_c) (below next) is blocked AND (next_r+1, current_c) (below current) is open
            if not _jps_is_walkable(grid, next_r + 1, next_c) and _jps_is_walkable(grid, next_r + 1, current_c):
                return next_node
            # Forced neighbor if (next_r-1, next_c) (above next) is blocked AND (next_r-1, current_c) (above current) is open
            if not _jps_is_walkable(grid, next_r - 1, next_c) and _jps_is_walkable(grid, next_r - 1, current_c):
                return next_node

        return _jps_jump(grid, next_r, next_c, dr, dc, _start_node_alg, end_node_alg)


def _jps_identify_successors(grid: Grid, current_node: Node, end_node_alg: Node) -> list[Node]:
    """
    Identifies all valid jump point successors for the `current_node` based on JPS pruning rules.
    The direction of movement from the parent to `current_node` (if any) influences which
    directions are explored for potential jump points.

    Args:
        grid: The grid instance.
        current_node: The node whose successors are to be identified. This is a jump point.
        end_node_alg: The overall goal node of the JPS algorithm.

    Returns:
        A list of unique jump point successor Nodes found by jumping from `current_node`.
    """
    successors: list[Node] = []
    parent = current_node.previous_node
    r, c = current_node.row, current_node.col

    dr_parent, dc_parent = 0, 0
    if parent:
        pr, pc = parent.row, parent.col
        if r - pr != 0: dr_parent = (r - pr) // abs(r - pr)
        if c - pc != 0: dc_parent = (c - pc) // abs(c - pc)

    possible_directions_to_check: list[tuple[int,int]] = []

    if dr_parent == 0 and dc_parent == 0:
        for dr_new in [-1, 0, 1]:
            for dc_new in [-1, 0, 1]:
                if dr_new == 0 and dc_new == 0: continue
                if not grid.allow_diagonal_movement and dr_new != 0 and dc_new != 0: continue
                possible_directions_to_check.append((dr_new, dc_new))
    elif dr_parent != 0 and dc_parent != 0:
        possible_directions_to_check.append((dr_parent, dc_parent))
        possible_directions_to_check.append((dr_parent, 0))
        possible_directions_to_check.append((0, dc_parent))
        if grid.allow_diagonal_movement:
            if not _jps_is_walkable(grid, r, c - dc_parent) and _jps_is_walkable(grid, r + dr_parent, c - dc_parent):
                possible_directions_to_check.append((dr_parent, -dc_parent))
            if not _jps_is_walkable(grid, r - dr_parent, c) and _jps_is_walkable(grid, r - dr_parent, c + dc_parent):
                possible_directions_to_check.append((-dr_parent, dc_parent))
    else:
        possible_directions_to_check.append((dr_parent, dc_parent))
        if grid.allow_diagonal_movement:
            if dr_parent != 0:
                if not _jps_is_walkable(grid, r, c + 1) and _jps_is_walkable(grid, r + dr_parent, c + 1):
                    possible_directions_to_check.append((dr_parent, 1))
                if not _jps_is_walkable(grid, r, c - 1) and _jps_is_walkable(grid, r + dr_parent, c - 1):
                    possible_directions_to_check.append((dr_parent, -1))
            else:
                if not _jps_is_walkable(grid, r + 1, c) and _jps_is_walkable(grid, r + 1, c + dc_parent):
                    possible_directions_to_check.append((1, dc_parent))
                if not _jps_is_walkable(grid, r - 1, c) and _jps_is_walkable(grid, r - 1, c + dc_parent):
                    possible_directions_to_check.append((-1, dc_parent))

    actual_directions_to_jump = []
    seen_directions = set()
    for dr_jump, dc_jump in possible_directions_to_check:
        if dr_jump == 0 and dc_jump == 0: continue
        if not grid.allow_diagonal_movement and dr_jump != 0 and dc_jump != 0: continue
        direction_tuple = (dr_jump, dc_jump)
        if direction_tuple not in seen_directions:
            actual_directions_to_jump.append(direction_tuple)
            seen_directions.add(direction_tuple)

    for dr_jump, dc_jump in actual_directions_to_jump:
        jump_point_node = _jps_jump(grid, r, c, dr_jump, dc_jump, grid.start_node, end_node_alg)
        if jump_point_node:
            successors.append(jump_point_node)

    return list(dict.fromkeys(successors))

def jps_search(grid: Grid, start_node: Node, end_node: Node) -> tuple[list[Node], list[Node], list[Node]]:
    """
    Performs Jump Point Search (JPS) algorithm, an optimization of A*,
    to find the shortest path from start_node to end_node on a uniform-cost grid.
    JPS prunes the search space by identifying "jump points" rather than expanding all neighbors.

    Args:
        grid: The Grid object.
        start_node: The starting Node.
        end_node: The target/goal Node.

    Returns:
        A tuple (path, visited_jump_points_in_order, open_set_final_nodes):
            - path: List of Node objects representing the full path (all cells interpolated between jump points),
                    empty if no path found.
            - visited_jump_points_in_order: List of jump point Nodes in the order they were expanded from the open set.
            - open_set_final_nodes: List of jump point Nodes remaining in the open set upon termination.
    """
    if not start_node or not end_node or start_node.is_obstacle or end_node.is_obstacle:
        return [], [], []

    grid.reset_algorithm_states()

    open_set_pq = []
    open_set_tracker = {}

    start_node.g = 0
    start_node.h_score = heuristic(start_node, end_node, grid.allow_diagonal_movement)
    start_node.f_score = start_node.g + start_node.h_score
    start_node.previous_node = None

    heapq.heappush(open_set_pq, (start_node.f_score, start_node.h_score, start_node))
    open_set_tracker[start_node] = start_node.g

    visited_jump_points_in_order = []

    while open_set_pq:
        _, _, current_jp_node = heapq.heappop(open_set_pq)

        if current_jp_node not in open_set_tracker or open_set_tracker[current_jp_node] < current_jp_node.g :
             if open_set_tracker.get(current_jp_node, float('inf')) < current_jp_node.g:
                continue

        del open_set_tracker[current_jp_node]
        visited_jump_points_in_order.append(current_jp_node)
        current_jp_node.is_visited_by_algorithm = True

        if current_jp_node == end_node:
            jp_path_reversed = []
            temp_jp = end_node
            while temp_jp:
                jp_path_reversed.append(temp_jp)
                temp_jp = temp_jp.previous_node

            if not jp_path_reversed:
                if end_node == start_node:
                    return [start_node], visited_jump_points_in_order, [node for _, _, node in open_set_pq if node in open_set_tracker]
                return [], visited_jump_points_in_order, [node for _, _, node in open_set_pq if node in open_set_tracker]

            jp_path = jp_path_reversed[::-1]

            final_path = []
            if not jp_path:
                return [], visited_jump_points_in_order, [node for _, _, node in open_set_pq if node in open_set_tracker]

            final_path.append(jp_path[0])

            for i in range(len(jp_path) - 1):
                p_start_segment = jp_path[i]
                p_end_segment = jp_path[i+1]

                r_curr, c_curr = p_start_segment.row, p_start_segment.col
                r_target, c_target = p_end_segment.row, p_end_segment.col

                while (r_curr, c_curr) != (r_target, c_target):
                    dr_step = (r_target - r_curr) // max(1, abs(r_target - r_curr)) if r_target != r_curr else 0
                    dc_step = (c_target - c_curr) // max(1, abs(c_target - c_curr)) if c_target != c_curr else 0

                    r_curr += dr_step
                    c_curr += dc_step

                    node_to_add = grid.get_node(r_curr, c_curr)
                    if node_to_add and (final_path[-1] != node_to_add if final_path else True):
                        final_path.append(node_to_add)

            return final_path, visited_jump_points_in_order, [node for _, _, node in open_set_pq if node in open_set_tracker]

        successors = _jps_identify_successors(grid, current_jp_node, end_node)

        for successor_jp_node in successors:
            dr_move = abs(current_jp_node.row - successor_jp_node.row)
            dc_move = abs(current_jp_node.col - successor_jp_node.col)

            cost_to_successor = 0.0
            if grid.allow_diagonal_movement:
                diag_steps = min(dr_move, dc_move)
                card_steps = max(dr_move, dc_move) - diag_steps
                cost_to_successor = (diag_steps * COST_DIAGONAL + card_steps * COST_CARDINAL) * successor_jp_node.terrain_cost
            else:
                cost_to_successor = (dr_move + dc_move) * COST_CARDINAL * successor_jp_node.terrain_cost

            temp_g_score = current_jp_node.g + cost_to_successor

            if temp_g_score < successor_jp_node.g:
                successor_jp_node.previous_node = current_jp_node
                successor_jp_node.g = temp_g_score
                successor_jp_node.h_score = heuristic(successor_jp_node, end_node, grid.allow_diagonal_movement)
                successor_jp_node.f_score = successor_jp_node.g + successor_jp_node.h_score

                if successor_jp_node not in open_set_tracker or open_set_tracker[successor_jp_node] > successor_jp_node.g:
                    heapq.heappush(open_set_pq, (successor_jp_node.f_score, successor_jp_node.h_score, successor_jp_node))
                    open_set_tracker[successor_jp_node] = successor_jp_node.g
                    successor_jp_node.is_in_open_set_for_algorithm = True

    return [], visited_jump_points_in_order, [node for _, _, node in open_set_pq if node in open_set_tracker]
