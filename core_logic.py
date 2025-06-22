import heapq

# Note: Pygame specific constants (WINDOW_WIDTH, colors, etc.) are NOT here.
# CELL_WIDTH/HEIGHT are passed to Node but not strictly needed for core logic if x,y are for GUI.

class Node:
    def __init__(self, row, col, total_rows, total_cols, width=0, height=0): # width/height optional for core
        self.row = row
        self.col = col
        self.x = col * width
        self.y = row * height
        self.width = width
        self.height = height

        self.is_obstacle = False
        self.terrain_cost = 1.0 # Default terrain cost
        self.g = float('inf')
        self.rhs = float('inf')
        self.h_score = float('inf')
        self.f_score = float('inf')
        self.previous_node = None
        self.neighbors = []
        self.total_rows = total_rows
        self.total_cols = total_cols

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
        self.h_score = float('inf')
        self.f_score = float('inf')
        self.previous_node = None
        self.is_visited_by_algorithm = False
        self.is_in_open_set_for_algorithm = False
        self.is_part_of_path = False

    def add_neighbors(self, grid_nodes_matrix, allow_diagonal=True):
        self.neighbors = []
        # Cardinal directions
        if self.row < self.total_rows - 1 and not grid_nodes_matrix[self.row + 1][self.col].is_obstacle:
            self.neighbors.append(grid_nodes_matrix[self.row + 1][self.col])
        if self.row > 0 and not grid_nodes_matrix[self.row - 1][self.col].is_obstacle:
            self.neighbors.append(grid_nodes_matrix[self.row - 1][self.col])
        if self.col < self.total_cols - 1 and not grid_nodes_matrix[self.row][self.col + 1].is_obstacle:
            self.neighbors.append(grid_nodes_matrix[self.row][self.col + 1])
        if self.col > 0 and not grid_nodes_matrix[self.row][self.col - 1].is_obstacle:
            self.neighbors.append(grid_nodes_matrix[self.row][self.col - 1])

        if allow_diagonal:
            # Diagonal directions
            if self.row < self.total_rows - 1 and self.col < self.total_cols - 1 and \
               not grid_nodes_matrix[self.row + 1][self.col + 1].is_obstacle:
                if not grid_nodes_matrix[self.row + 1][self.col].is_obstacle or \
                   not grid_nodes_matrix[self.row][self.col + 1].is_obstacle:
                    self.neighbors.append(grid_nodes_matrix[self.row + 1][self.col + 1])
            if self.row < self.total_rows - 1 and self.col > 0 and \
               not grid_nodes_matrix[self.row + 1][self.col - 1].is_obstacle:
                if not grid_nodes_matrix[self.row + 1][self.col].is_obstacle or \
                   not grid_nodes_matrix[self.row][self.col - 1].is_obstacle:
                    self.neighbors.append(grid_nodes_matrix[self.row + 1][self.col - 1])
            if self.row > 0 and self.col < self.total_cols - 1 and \
               not grid_nodes_matrix[self.row - 1][self.col + 1].is_obstacle:
                if not grid_nodes_matrix[self.row - 1][self.col].is_obstacle or \
                   not grid_nodes_matrix[self.row][self.col + 1].is_obstacle:
                    self.neighbors.append(grid_nodes_matrix[self.row - 1][self.col + 1])
            if self.row > 0 and self.col > 0 and \
               not grid_nodes_matrix[self.row - 1][self.col - 1].is_obstacle:
                if not grid_nodes_matrix[self.row - 1][self.col].is_obstacle or \
                   not grid_nodes_matrix[self.row][self.col - 1].is_obstacle:
                    self.neighbors.append(grid_nodes_matrix[self.row - 1][self.col - 1])

class Grid:
    def __init__(self, rows, cols, cell_width_for_nodes=0, cell_height_for_nodes=0):
        self.rows = rows
        self.cols = cols
        self.allow_diagonal_movement = True
        self.nodes = [[Node(r, c, rows, cols, cell_width_for_nodes, cell_height_for_nodes) for c in range(cols)] for r in range(rows)]
        self.start_node = None
        self.end_node = None
        self.update_all_node_neighbors()

    def update_all_node_neighbors(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.nodes[r][c].add_neighbors(self.nodes, self.allow_diagonal_movement)

    def set_allow_diagonal_movement(self, allow: bool):
        self.allow_diagonal_movement = allow
        self.update_all_node_neighbors()

    def reset_algorithm_states(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.nodes[r][c].reset_algorithm_attributes()

    def get_node(self, row, col):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.nodes[row][col]
        return None

    def set_target_node(self, row, col):
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

COST_CARDINAL = 1.0
COST_DIAGONAL = 1.41421356

def get_move_cost(node1, node2):
    base_cost = COST_CARDINAL
    if abs(node1.row - node2.row) == 1 and abs(node1.col - node2.col) == 1:
        base_cost = COST_DIAGONAL
    return base_cost * node2.terrain_cost

def heuristic(node_a, node_b, allow_diagonal=True):
    dx = abs(node_a.col - node_b.col)
    dy = abs(node_a.row - node_b.row)
    if allow_diagonal:
        return COST_DIAGONAL * min(dx, dy) + COST_CARDINAL * (max(dx, dy) - min(dx, dy))
    else:
        return COST_CARDINAL * (dx + dy)

def dijkstra(grid, start_node, end_node):
    if not start_node or not end_node or start_node.is_obstacle or end_node.is_obstacle:
        return [], [], []
    visited_nodes_in_order = []
    start_node.g = 0
    start_node.h_score = 0
    start_node.f_score = start_node.g
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

def calculate_d_star_key(node, start_node, goal_node, heuristic_func, allow_diagonal):
    h_val = heuristic_func(node, start_node, allow_diagonal)
    min_g_rhs = min(node.g, node.rhs)
    return (min_g_rhs + h_val, min_g_rhs)

def d_star_lite_initialize(grid, start_node, goal_node, heuristic_func, pq, open_set_tracker):
    grid.reset_algorithm_states()
    goal_node.rhs = 0
    key = calculate_d_star_key(goal_node, start_node, goal_node, heuristic_func, grid.allow_diagonal_movement)
    heapq.heappush(pq, (key, goal_node))
    open_set_tracker.add(goal_node)

def d_star_lite_update_node(node, grid, start_node, goal_node, heuristic_func, pq, open_set_tracker):
    if node != goal_node:
        min_rhs = float('inf')
        for successor in node.neighbors:
            cost = get_move_cost(node, successor)
            min_rhs = min(min_rhs, successor.g + cost)
        node.rhs = min_rhs
    if node in open_set_tracker and node.g == node.rhs:
         pass
    if node.g != node.rhs:
        key = calculate_d_star_key(node, start_node, goal_node, heuristic_func, grid.allow_diagonal_movement)
        heapq.heappush(pq, (key, node))
        open_set_tracker.add(node)
    elif node in open_set_tracker:
        open_set_tracker.remove(node)

def d_star_lite_compute_shortest_path(grid, start_node, goal_node, heuristic_func, pq, open_set_tracker):
    processed_nodes_for_viz = []
    while (pq and calculate_d_star_key(start_node, start_node, goal_node, heuristic_func, grid.allow_diagonal_movement) > pq[0][0]) \
          or start_node.g != start_node.rhs:
        if not pq: break
        k_old_popped, u_node = heapq.heappop(pq)
        if u_node not in open_set_tracker and u_node.g == u_node.rhs :
            continue
        if u_node in open_set_tracker:
             open_set_tracker.remove(u_node)
        processed_nodes_for_viz.append(u_node)
        k_new_recalc = calculate_d_star_key(u_node, start_node, goal_node, heuristic_func, grid.allow_diagonal_movement)
        if k_old_popped < k_new_recalc:
            heapq.heappush(pq, (k_new_recalc, u_node))
            open_set_tracker.add(u_node)
        elif u_node.g > u_node.rhs:
            u_node.g = u_node.rhs
            for pred_node in u_node.neighbors:
                d_star_lite_update_node(pred_node, grid, start_node, goal_node, heuristic_func, pq, open_set_tracker)
        else:
            g_old = u_node.g
            u_node.g = float('inf')
            for pred_node in u_node.neighbors:
                d_star_lite_update_node(pred_node, grid, start_node, goal_node, heuristic_func, pq, open_set_tracker)
            d_star_lite_update_node(u_node, grid, start_node, goal_node, heuristic_func, pq, open_set_tracker)
    path = []
    if start_node.g == float('inf'):
        print("No path found by D* Lite.")
        return [], processed_nodes_for_viz, list(open_set_tracker) # Corrected
    curr = start_node
    path.append(curr)
    while curr != goal_node:
        min_total_cost_to_goal = float('inf')
        next_node = None
        if not curr.neighbors:
            print("Error in path reconstruction: current node has no neighbors.")
            return [], processed_nodes_for_viz, list(open_set_tracker) # Corrected
        for neighbor in curr.neighbors:
            cost_to_neighbor = get_move_cost(curr, neighbor)
            current_path_cost_via_neighbor = cost_to_neighbor + neighbor.g
            if current_path_cost_via_neighbor < min_total_cost_to_goal:
                min_total_cost_to_goal = current_path_cost_via_neighbor
                next_node = neighbor
            elif current_path_cost_via_neighbor == min_total_cost_to_goal:
                 if next_node is None:
                     next_node = neighbor
        if next_node is None :
             print(f"Path reconstruction failed: No valid next node from {curr.get_pos()} with g={curr.g}")
             return [], processed_nodes_for_viz, list(open_set_tracker) # Corrected
        curr = next_node
        path.append(curr)
        if len(path) > grid.rows * grid.cols:
            print("Path reconstruction exceeded max length.")
            return [], processed_nodes_for_viz, list(open_set_tracker) # Corrected
    return path, processed_nodes_for_viz, list(open_set_tracker) # Corrected

def run_d_star_lite(grid, start_node, goal_node, heuristic_func):
    if not start_node or not goal_node or start_node.is_obstacle or goal_node.is_obstacle:
        print("D* Lite: Start or Goal is invalid or an obstacle.")
        return [], [], []
    pq = []
    open_set_tracker = set()
    grid.start_node = start_node
    grid.end_node = goal_node
    d_star_lite_initialize(grid, start_node, goal_node, heuristic_func, pq, open_set_tracker)
    path, visited_nodes, open_set_final_nodes = d_star_lite_compute_shortest_path(grid, start_node, goal_node, heuristic_func, pq, open_set_tracker)
    return path, visited_nodes, open_set_final_nodes

def d_star_lite_obstacle_change_update(grid, r_changed, c_changed,
                                       pq, open_set_tracker,
                                       start_node, goal_node, heuristic_func):
    changed_node = grid.get_node(r_changed, c_changed)
    if not changed_node:
        print(f"Error in d_star_lite_obstacle_change_update: Node ({r_changed},{c_changed}) not found.")
        return
    grid.update_all_node_neighbors()
    d_star_lite_update_node(changed_node, grid, start_node, goal_node, heuristic_func, pq, open_set_tracker)
    for dr_offset in [-1, 0, 1]:
        for dc_offset in [-1, 0, 1]:
            if dr_offset == 0 and dc_offset == 0:
                continue
            nr, nc = r_changed + dr_offset, c_changed + dc_offset
            if 0 <= nr < grid.rows and 0 <= nc < grid.cols:
                neighbor_node = grid.nodes[nr][nc]
                d_star_lite_update_node(neighbor_node, grid, start_node, goal_node, heuristic_func, pq, open_set_tracker)
    print(f"D* Lite: Processed obstacle change at ({r_changed},{c_changed}). Call compute_shortest_path to replan.")

def d_star_lite_target_move_update(grid, new_target_node: Node, old_target_node: Node | None,
                                   pq, open_set_tracker,
                                   start_node, heuristic_func):
    if old_target_node and old_target_node != new_target_node:
        old_target_node.rhs = float('inf')
        d_star_lite_update_node(old_target_node, grid, start_node, new_target_node, heuristic_func, pq, open_set_tracker)
    grid.end_node = new_target_node
    new_target_node.rhs = 0
    d_star_lite_update_node(new_target_node, grid, start_node, new_target_node, heuristic_func, pq, open_set_tracker)
    print(f"D* Lite: Processed target move to ({new_target_node.row},{new_target_node.col}). Call compute_shortest_path to replan.")

def a_star(grid, start_node, end_node):
    if not start_node or not end_node or start_node.is_obstacle or end_node.is_obstacle:
        return [], [], []
    visited_nodes_in_order = []
    start_node.g = 0
    start_node.h_score = heuristic(start_node, end_node, grid.allow_diagonal_movement)
    start_node.f_score = start_node.g + start_node.h_score
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
                neighbor.h_score = heuristic(neighbor, end_node, grid.allow_diagonal_movement)
                neighbor.f_score = neighbor.g + neighbor.h_score
                if neighbor not in open_set_tracker:
                    heapq.heappush(pq, (neighbor.f_score, neighbor))
                    open_set_tracker.add(neighbor)
    return [], visited_nodes_in_order, list(open_set_tracker)

def bidirectional_search(grid, start_node, end_node):
    if not start_node or not end_node or start_node.is_obstacle or end_node.is_obstacle:
        return [], [], [], [], []
    g_fwd = {node: float('inf') for row in grid.nodes for node in row}
    g_bwd = {node: float('inf') for row in grid.nodes for node in row}
    prev_fwd = {node: None for row in grid.nodes for node in row}
    prev_bwd = {node: None for row in grid.nodes for node in row}
    g_fwd[start_node] = 0
    start_node.f_score = heuristic(start_node, end_node, grid.allow_diagonal_movement)
    g_bwd[end_node] = 0
    pq_fwd = [(0, start_node)]
    pq_bwd = [(0, end_node)]
    open_set_tracker_fwd = {start_node}
    open_set_tracker_bwd = {end_node}
    closed_set_fwd = set()
    closed_set_bwd = set()
    visited_nodes_in_order_fwd = []
    visited_nodes_in_order_bwd = []
    meeting_node = None
    path_cost = float('inf')
    while pq_fwd and pq_bwd:
        if pq_fwd:
            _, current_node_fwd = heapq.heappop(pq_fwd)
            if current_node_fwd not in open_set_tracker_fwd:
                continue
            open_set_tracker_fwd.remove(current_node_fwd)
            closed_set_fwd.add(current_node_fwd)
            visited_nodes_in_order_fwd.append(current_node_fwd)
            if current_node_fwd in closed_set_bwd:
                current_path_cost = g_fwd[current_node_fwd] + g_bwd[current_node_fwd]
                if current_path_cost < path_cost:
                    path_cost = current_path_cost
                    meeting_node = current_node_fwd
                if pq_bwd and (g_fwd[current_node_fwd] + pq_bwd[0][0] >= path_cost):
                     break
            for neighbor in current_node_fwd.neighbors:
                if neighbor in closed_set_fwd:
                    continue
                cost = get_move_cost(current_node_fwd, neighbor)
                temp_g_fwd = g_fwd[current_node_fwd] + cost
                if temp_g_fwd < g_fwd[neighbor]:
                    g_fwd[neighbor] = temp_g_fwd
                    prev_fwd[neighbor] = current_node_fwd
                    heapq.heappush(pq_fwd, (temp_g_fwd, neighbor))
                    open_set_tracker_fwd.add(neighbor)
        if pq_bwd:
            _, current_node_bwd = heapq.heappop(pq_bwd)
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
                if pq_fwd and (g_bwd[current_node_bwd] + pq_fwd[0][0] >= path_cost):
                    break
            for neighbor in current_node_bwd.neighbors:
                if neighbor in closed_set_bwd:
                    continue
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
        path_s_to_m = []
        curr = meeting_node
        while curr:
            path_s_to_m.append(curr)
            curr = prev_fwd[curr]
        path_s_to_m.reverse()
        path_e_to_m = []
        curr = meeting_node
        while curr:
            path_e_to_m.append(curr)
            curr = prev_bwd[curr]
        path_e_to_m.reverse()
        final_path = path_s_to_m
        if len(path_e_to_m) > 1:
            final_path.extend(path_e_to_m[-2::-1])
    all_visited_for_return = visited_nodes_in_order_fwd + visited_nodes_in_order_bwd
    final_open_set_fwd = list(open_set_tracker_fwd)
    final_open_set_bwd = list(open_set_tracker_bwd)
    return final_path, visited_nodes_in_order_fwd, visited_nodes_in_order_bwd, final_open_set_fwd, final_open_set_bwd

def _jps_is_walkable(grid, r, c):
    if not (0 <= r < grid.rows and 0 <= c < grid.cols):
        return False
    return not grid.nodes[r][c].is_obstacle

def _jps_jump(grid, current_r, current_c, dr, dc, start_node, end_node):
    next_r, next_c = current_r + dr, current_c + dc
    if not _jps_is_walkable(grid, next_r, next_c):
        return None
    next_node = grid.nodes[next_r][next_c]
    if next_node == end_node:
        return next_node
    if dr != 0 and dc != 0:
        if (_jps_is_walkable(grid, next_r - dr, next_c + dc) and not _jps_is_walkable(grid, next_r - dr, next_c)) or \
           (_jps_is_walkable(grid, next_r + dr, next_c + dc) and not _jps_is_walkable(grid, next_r + dr, next_c)):
            pass
        if _jps_jump(grid, next_r, next_c, dr, 0, start_node, end_node) or \
           _jps_jump(grid, next_r, next_c, 0, dc, start_node, end_node):
            return next_node
    else:
        if dr != 0:
            if (_jps_is_walkable(grid, next_r, next_c + 1) and not _jps_is_walkable(grid, next_r - dr, next_c + 1)) or \
               (_jps_is_walkable(grid, next_r, next_c - 1) and not _jps_is_walkable(grid, next_r - dr, next_c - 1)):
                return next_node
        else:
            if (_jps_is_walkable(grid, next_r + 1, next_c) and not _jps_is_walkable(grid, next_r + 1, next_c - dc)) or \
               (_jps_is_walkable(grid, next_r - 1, next_c) and not _jps_is_walkable(grid, next_r - 1, next_c - dc)):
                return next_node
    if not grid.allow_diagonal_movement and dr != 0 and dc != 0:
        return None
    return None

def _jps_identify_successors(grid, current_node, end_node):
    successors = []
    parent = current_node.previous_node
    px, py = (parent.col, parent.row) if parent else (-1,-1)
    cx, cy = current_node.col, current_node.row
    dx = (cx - px) // max(1, abs(cx - px)) if px != -1 else 0
    dy = (cy - py) // max(1, abs(cy - py)) if py != -1 else 0
    possible_directions = []
    if dx != 0 and dy != 0:
        if _jps_is_walkable(grid, cy + dy, cx): possible_directions.append((0, dy))
        if _jps_is_walkable(grid, cy, cx + dx): possible_directions.append((dx, 0))
        if _jps_is_walkable(grid, cy + dy, cx + dx) and (_jps_is_walkable(grid, cy + dy, cx) or _jps_is_walkable(grid, cy, cx + dx)):
            possible_directions.append((dx, dy))
        if not _jps_is_walkable(grid, cy, cx - dx) and _jps_is_walkable(grid, cy + dy, cx - dx):
             possible_directions.append((-dx, dy))
        if not _jps_is_walkable(grid, cy - dy, cx) and _jps_is_walkable(grid, cy - dy, cx + dx):
             possible_directions.append((dx, -dy))
    else:
        if dx == 0:
            if _jps_is_walkable(grid, cy + dy, cx):
                possible_directions.append((0, dy))
            if not _jps_is_walkable(grid, cy, cx + 1) and _jps_is_walkable(grid, cy + dy, cx + 1):
                possible_directions.append((1, dy))
            if not _jps_is_walkable(grid, cy, cx - 1) and _jps_is_walkable(grid, cy + dy, cx - 1):
                possible_directions.append((-1, dy))
        elif dy == 0:
            if _jps_is_walkable(grid, cy, cx + dx):
                possible_directions.append((dx, 0))
            if not _jps_is_walkable(grid, cy + 1, cx) and _jps_is_walkable(grid, cy + 1, cx + dx):
                possible_directions.append((dx, 1))
            if not _jps_is_walkable(grid, cy - 1, cx) and _jps_is_walkable(grid, cy - 1, cx + dx):
                possible_directions.append((dx, -1))
        else:
            for dr_new in [-1, 0, 1]:
                for dc_new in [-1, 0, 1]:
                    if dr_new == 0 and dc_new == 0: continue
                    if not grid.allow_diagonal_movement and dr_new != 0 and dc_new != 0: continue
                    if _jps_is_walkable(grid, cy + dr_new, cx + dc_new):
                         possible_directions.append((dc_new, dr_new))
    for dc_new, dr_new in possible_directions:
        jump_point = _jps_jump(grid, cy, cx, dr_new, dc_new, current_node, end_node)
        if jump_point:
            successors.append(jump_point)
    return successors

def jps_search(grid, start_node, end_node):
    if not start_node or not end_node or start_node.is_obstacle or end_node.is_obstacle:
        return [], [], []
    open_set = []
    heapq.heappush(open_set, (start_node.f_score, heuristic(start_node, end_node, grid.allow_diagonal_movement), start_node))
    open_set_tracker = {start_node}
    start_node.g = 0
    start_node.h_score = heuristic(start_node, end_node, grid.allow_diagonal_movement)
    start_node.f_score = start_node.g + start_node.h_score
    visited_nodes_in_order = []
    while open_set:
        _, _, current_jump_point = heapq.heappop(open_set)
        if current_jump_point not in open_set_tracker:
            continue
        open_set_tracker.remove(current_jump_point)
        visited_nodes_in_order.append(current_jump_point)
        if current_jump_point == end_node:
            path = []
            temp = end_node
            while temp:
                path.append(temp)
                temp = temp.previous_node
            return path[::-1], visited_nodes_in_order, list(open_set_tracker)
        successors = _jps_identify_successors(grid, current_jump_point, end_node)
        for successor_jp_node in successors:
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
            temp_g_score = current_jump_point.g + cost_to_successor
            if temp_g_score < successor_jp_node.g:
                successor_jp_node.previous_node = current_jump_point
                successor_jp_node.g = temp_g_score
                successor_jp_node.h_score = heuristic(successor_jp_node, end_node, grid.allow_diagonal_movement)
                successor_jp_node.f_score = successor_jp_node.g + successor_jp_node.h_score
                if successor_jp_node not in open_set_tracker:
                    heapq.heappush(open_set, (successor_jp_node.f_score, successor_jp_node.h_score, successor_jp_node))
                    open_set_tracker.add(successor_jp_node)
    return [], visited_nodes_in_order, list(open_set_tracker)
