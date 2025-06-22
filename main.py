import pygame

# --- Constants ---
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
GRID_ROWS = 20
GRID_COLS = 20
CELL_WIDTH = WINDOW_WIDTH // GRID_COLS
CELL_HEIGHT = WINDOW_HEIGHT // GRID_ROWS

# Pygame specific imports are already at the top (import pygame)

# Core logic imports
from core_logic import (Node, Grid, dijkstra, a_star, run_d_star_lite, heuristic,
                        bidirectional_search, jps_search,
                        d_star_lite_obstacle_change_update, d_star_lite_target_move_update,
                        d_star_lite_initialize, d_star_lite_compute_shortest_path) # Added D* specific imports

# Colors - remain in GUI module
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200) # For grid lines
RED_OBSTACLE = (255, 0, 0)
GREEN_START = (0, 255, 0)
BLUE_END = (0, 0, 255)
CYAN_VISITED = (0, 180, 180)
ORANGE_OPEN = (255, 165, 0)
MAGENTA_PATH = (255, 0, 255)

# Terrain Colors & Costs
TERRAIN_DEFAULT_COLOR = WHITE # Cost 1.0 (already default for node.terrain_cost)
TERRAIN_MUD_COLOR = (139, 69, 19)  # Brown for "mud" (higher cost)
TERRAIN_WATER_COLOR = (30, 144, 255) # DodgerBlue for "water" (even higher cost)

TERRAIN_COSTS = {
    1: {"cost": 1.0, "name": "Normal"}, # Default, associated with WHITE or no special terrain painting
    2: {"cost": 3.0, "name": "Mud"},    # Key '1' for terrain type 1
    3: {"cost": 5.0, "name": "Water"}   # Key '2' for terrain type 2
}
# Note: The visual association will be handled in draw_all_nodes and interaction logic.
# We'll use number keys 1, 2, 3 to cycle through terrain paint modes.
# Key 1: Paint Normal (effectively remove special terrain)
# Key 2: Paint Mud (cost 3)
# Key 3: Paint Water (cost 5)


# --- GUI specific Helper Functions ---

def get_gui_node_from_mouse_pos(mouse_pos, grid_instance: Grid, cell_width, cell_height):
    """Gets the grid node object from a mouse position using core_logic.Grid."""
    try:
        col = mouse_pos[0] // cell_width
        row = mouse_pos[1] // cell_height
        # Use grid_instance.get_node(row,col) if available, or access grid_instance.nodes directly
        if 0 <= row < grid_instance.rows and 0 <= col < grid_instance.cols:
            return grid_instance.nodes[row][col] # Accessing the .nodes attribute of core_logic.Grid
        return None
    except IndexError:
        return None

def draw_grid_lines(surface, rows, cols, window_width, window_height, cell_width, cell_height):
    """Draws the grid lines on the surface."""
    for r in range(rows + 1): # +1 to draw the bottom and rightmost lines
        pygame.draw.line(surface, GREY, (0, r * cell_height), (window_width, r * cell_height))
    for c in range(cols + 1): # +1
        pygame.draw.line(surface, GREY, (c * cell_width, 0), (c * cell_width, window_height))


def draw_all_nodes(surface, grid_instance: Grid):
    """Draws all nodes on the grid based on their state using core_logic.Grid and core_logic.Node."""
    for r in range(grid_instance.rows):
        for c in range(grid_instance.cols):
            node = grid_instance.nodes[r][c]
            # Node's x, y, width, height are set if cell_width/height were passed to core_logic.Grid
            rect = pygame.Rect(node.x, node.y, node.width, node.height)

            # Determine base color based on terrain cost first
            if node.terrain_cost == TERRAIN_COSTS[2]["cost"]: # Mud
                node_color = TERRAIN_MUD_COLOR
            elif node.terrain_cost == TERRAIN_COSTS[3]["cost"]: # Water
                node_color = TERRAIN_WATER_COLOR
            else: # Default terrain or other states
                node_color = WHITE

            # Algorithm visualization overlays (path, open, visited)
            # These should take precedence over basic terrain color if node is part of visualization
            if node.is_part_of_path:
                node_color = MAGENTA_PATH
            elif node.is_in_open_set_for_algorithm: # Should be drawn before visited
                node_color = ORANGE_OPEN
            elif node.is_visited_by_algorithm:
                node_color = CYAN_VISITED

            # Obstacle, start, and end display properties (highest precedence)
            if node.is_obstacle:
                node_color = RED_OBSTACLE
            elif node == grid_instance.start_node: # Check if it's the start_node instance
                node_color = GREEN_START
            elif node == grid_instance.end_node: # Check if it's the end_node instance
                node_color = BLUE_END

            pygame.draw.rect(surface, node_color, rect)


def main_gui():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    # Caption will be set by update_caption()
    clock = pygame.time.Clock()

    grid_instance = Grid(GRID_ROWS, GRID_COLS, CELL_WIDTH, CELL_HEIGHT)

    setting_start = False
    setting_end = False
    setting_target_d_lite = False # For D* Lite target moving mode
    painting_terrain_type = 0 # 0: Not painting terrain, 1: Normal, 2: Mud, 3: Water

    # D* Lite persistent state for replanning
    d_star_lite_pq = []
    d_star_lite_open_set_tracker = set()
    d_star_lite_initialized_run = False # Flag to check if D* Lite has been run at least once

    current_algorithm = "A_STAR" # Default algorithm
    ALGORITHMS = {
        "A_STAR": a_star,
        "DIJKSTRA": dijkstra,
        "D_STAR_LITE": run_d_star_lite,
        "BIDIRECTIONAL": bidirectional_search,
        "JPS": jps_search # Added JPS
    }
    ALGO_NAMES = { # For display purposes
        "A_STAR": "A*",
        "DIJKSTRA": "Dijkstra",
        "D_STAR_LITE": "D* Lite",
        "BIDIRECTIONAL": "Bidirectional Search",
        "JPS": "Jump Point Search" # Added JPS
    }

    def update_caption():
        nonlocal setting_start, setting_end, setting_target_d_lite, current_algorithm, grid_instance, painting_terrain_type
        mode_text = ""
        if setting_start:
            mode_text = "Set Start"
        elif setting_end:
            mode_text = "Set End"
        elif setting_target_d_lite:
            mode_text = "Set D* Target (LClick)"
        elif painting_terrain_type > 0:
            terrain_info = TERRAIN_COSTS.get(painting_terrain_type)
            if terrain_info:
                mode_text = f"Paint Terrain: {terrain_info['name']} (Cost: {terrain_info['cost']})"
            else: # Should not happen if painting_terrain_type is correctly managed
                mode_text = "Paint Terrain: Unknown"


        diag_status = "ON" if grid_instance.allow_diagonal_movement else "OFF"
        base_caption = f"Algorithm: {ALGO_NAMES[current_algorithm]} | Diagonal: {diag_status}"

        if mode_text:
            caption = f"{base_caption} | {mode_text}"
        else:
            bindings = "S:Start, E:End, D:Diag, R:Reset | Algos:K,A,L,B,J(JPS) | Terrain:1-3 | Enter:Run"
            if current_algorithm == "D_STAR_LITE":
                bindings += ", T:Move Target"
            caption = f"{base_caption} | {bindings}"
        pygame.display.set_caption(caption)

    update_caption() # Initial caption

    animating = False
    animation_data = {
        "visited_nodes_in_order": [],
        "path_nodes": [],
        "open_set_nodes": [], # Nodes that were in open set at the end
        "current_visited_step": 0,
        "current_path_step": 0,
        "animation_phase": "stopped" # "visited", "path", "finished"
    }
    ANIMATION_DELAY_MS = 25


    def start_animation(path, visited_nodes_in_order, open_set_nodes):
        nonlocal animating, animation_data, current_algorithm # Added current_algorithm
        grid_instance.reset_algorithm_states()

        animation_data["visited_nodes_in_order"] = visited_nodes_in_order
        animation_data["path_nodes"] = path
        animation_data["open_set_nodes"] = open_set_nodes
        animation_data["current_visited_step"] = 0
        animation_data["current_path_step"] = 0
        animation_data["animation_phase"] = "visited"
        animating = True
        pygame.display.set_caption(f"Visualizing {ALGO_NAMES[current_algorithm]}...")


    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if not animating:
                if event.type == pygame.KEYDOWN:
                    # Algorithm Selection
                    if event.key == pygame.K_a:
                        current_algorithm = "A_STAR"
                        setting_target_d_lite = False # Exit D* target mode if switching
                    elif event.key == pygame.K_l: # 'L' for D* Lite
                        current_algorithm = "D_STAR_LITE"
                    elif event.key == pygame.K_k: # Using K for Dijkstra
                        current_algorithm = "DIJKSTRA"
                        setting_target_d_lite = False
                    elif event.key == pygame.K_b: # 'B' for Bidirectional
                        current_algorithm = "BIDIRECTIONAL"
                        setting_target_d_lite = False
                    elif event.key == pygame.K_j: # 'J' for JPS
                        current_algorithm = "JPS"
                        setting_target_d_lite = False

                    elif event.key == pygame.K_d: # Toggle Diagonal Movement
                        grid_instance.set_allow_diagonal_movement(not grid_instance.allow_diagonal_movement)
                        # No need to reset algorithm states here unless an algorithm is running/paused
                        # update_all_node_neighbors is called by set_allow_diagonal_movement
                        print(f"Diagonal movement {'enabled' if grid_instance.allow_diagonal_movement else 'disabled'}")

                    # Terrain Painting Mode Selection
                    elif event.key == pygame.K_1:
                        painting_terrain_type = 1 # Normal
                        setting_start, setting_end, setting_target_d_lite = False, False, False
                        print(f"Set to paint terrain: {TERRAIN_COSTS[1]['name']}")
                    elif event.key == pygame.K_2:
                        painting_terrain_type = 2 # Mud
                        setting_start, setting_end, setting_target_d_lite = False, False, False
                        print(f"Set to paint terrain: {TERRAIN_COSTS[2]['name']}")
                    elif event.key == pygame.K_3:
                        painting_terrain_type = 3 # Water
                        setting_start, setting_end, setting_target_d_lite = False, False, False
                        print(f"Set to paint terrain: {TERRAIN_COSTS[3]['name']}")
                    elif event.key == pygame.K_0 or event.key == pygame.K_ESCAPE: # Turn off terrain painting
                        painting_terrain_type = 0
                        print("Terrain painting mode OFF")


                    # Mode Selection (Start, End, D* Target) - Ensure these turn off terrain painting
                    if event.key == pygame.K_s:
                        setting_start = True
                        setting_end, setting_target_d_lite, painting_terrain_type = False, False, 0
                    elif event.key == pygame.K_e:
                        setting_end = True
                        setting_start, setting_target_d_lite, painting_terrain_type = False, False, 0
                    elif event.key == pygame.K_t and current_algorithm == "D_STAR_LITE":
                        setting_target_d_lite = True
                        setting_start, setting_end, painting_terrain_type = False, False, 0

                    # If any mode key (S, E, T, or terrain keys) was pressed, other modes should be off.
                    # The above logic handles this by setting other modes to False when one is activated.
                    # If terrain painting was just activated, other modes (S,E,T) are already false.


                    elif event.key == pygame.K_RETURN: # Enter key to run selected algorithm
                        if grid_instance.start_node and grid_instance.end_node:
                            grid_instance.reset_algorithm_states()
                            # grid_instance.update_all_node_neighbors() # This is now handled by set_allow_diagonal_movement or grid init
                                                                    # However, if obstacles were added/removed, it should be called.
                                                                    # Obstacle toggling calls it. Reset calls it via new Grid().
                                                                    # Explicitly calling here ensures neighbors are fresh if state changed some other way.
                            grid_instance.update_all_node_neighbors()


                            algo_func = ALGORITHMS[current_algorithm]
                            algo_name_display = ALGO_NAMES[current_algorithm]
                            diag_status_msg = "with" if grid_instance.allow_diagonal_movement else "without"
                            print(f"Running {algo_name_display} Algorithm {diag_status_msg} diagonal movement...")

                            if current_algorithm == "D_STAR_LITE":
                                path, visited, open_set = algo_func(grid_instance, grid_instance.start_node, grid_instance.end_node, heuristic)
                                start_animation(path, visited, open_set) # Standard animation
                            elif current_algorithm == "BIDIRECTIONAL":
                                # Bidirectional returns: path, visited_fwd, visited_bwd, open_fwd, open_bwd
                                path, visited_fwd, visited_bwd, open_fwd, open_bwd = algo_func(grid_instance, grid_instance.start_node, grid_instance.end_node)
                                # For now, combine visited and open sets for basic animation.
                                # TODO: Enhance start_animation or create a new one for distinct bidirectional visualization
                                combined_visited = visited_fwd + list(set(visited_bwd) - set(visited_fwd)) # Crude merge
                                combined_open = list(set(open_fwd + open_bwd)) # Crude merge
                                start_animation(path, combined_visited, combined_open)
                            elif current_algorithm == "JPS":
                                # JPS returns path, visited_jump_points, open_set_jump_points
                                path, visited_jp, open_set_jp = algo_func(grid_instance, grid_instance.start_node, grid_instance.end_node)
                                # For visualization, visited_jp are the jump points considered.
                                # The actual "visited" cells during jumps are not explicitly returned by this simplified JPS.
                                # So, we'll animate based on the jump points.
                                start_animation(path, visited_jp, open_set_jp)
                            else: # A*, Dijkstra
                                path, visited, open_set = algo_func(grid_instance, grid_instance.start_node, grid_instance.end_node)
                                start_animation(path, visited, open_set)

                            if path:
                                print(f"{algo_name_display} Path found. Length: {len(path)}")
                            else:
                                print(f"{algo_name_display} No path found.")
                        else:
                            print("Error: Set Start and End nodes before running an algorithm.")

                    elif event.key == pygame.K_r: # Reset grid
                        # Preserve current diagonal setting when resetting
                        current_diag_setting = grid_instance.allow_diagonal_movement
                        grid_instance = Grid(GRID_ROWS, GRID_COLS, CELL_WIDTH, CELL_HEIGHT)
                        grid_instance.set_allow_diagonal_movement(current_diag_setting) # Restore
                        setting_start = False
                        setting_end = False
                        setting_target_d_lite = False
                        painting_terrain_type = 0 # Also reset terrain painting mode
                        d_star_lite_initialized_run = False # Reset D* Lite state
                        d_star_lite_pq.clear()
                        d_star_lite_open_set_tracker.clear()
                        animating = False
                        animation_data["animation_phase"] = "stopped"

                    update_caption() # Update caption after any relevant key press

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    clicked_node = get_gui_node_from_mouse_pos(mouse_pos, grid_instance, CELL_WIDTH, CELL_HEIGHT)

                    if clicked_node:
                        old_obstacle_status = clicked_node.is_obstacle
                        action_taken = False # Flag to see if an action requires D* update

                        if setting_start:
                            if clicked_node == grid_instance.end_node: grid_instance.end_node = None
                            if grid_instance.start_node:
                                grid_instance.start_node.is_obstacle = False
                                grid_instance.start_node.terrain_cost = TERRAIN_COSTS[1]["cost"]
                            grid_instance.start_node = clicked_node
                            grid_instance.start_node.is_obstacle = False
                            grid_instance.start_node.terrain_cost = TERRAIN_COSTS[1]["cost"]
                            print(f"Set start node at ({clicked_node.row}, {clicked_node.col})")
                            setting_start = False
                            d_star_lite_initialized_run = False # Start/End change invalidates D* state
                            action_taken = True
                        elif setting_end:
                            old_target_for_d_lite = grid_instance.end_node
                            if clicked_node == grid_instance.start_node: grid_instance.start_node = None
                            if grid_instance.end_node:
                                grid_instance.end_node.is_obstacle = False
                                grid_instance.end_node.terrain_cost = TERRAIN_COSTS[1]["cost"]
                            grid_instance.end_node = clicked_node
                            grid_instance.end_node.is_obstacle = False
                            grid_instance.end_node.terrain_cost = TERRAIN_COSTS[1]["cost"]
                            print(f"Set end node at ({clicked_node.row}, {clicked_node.col})")
                            setting_end = False
                            action_taken = True
                            if current_algorithm == "D_STAR_LITE" and d_star_lite_initialized_run and grid_instance.start_node:
                                d_star_lite_target_move_update(grid_instance, grid_instance.end_node, old_target_for_d_lite,
                                                               d_star_lite_pq, d_star_lite_open_set_tracker,
                                                               grid_instance.start_node, heuristic)
                                path, visited, open_set = d_star_lite_compute_shortest_path(
                                    grid_instance, grid_instance.start_node, grid_instance.end_node, heuristic,
                                    d_star_lite_pq, d_star_lite_open_set_tracker)
                                start_animation(path, visited, open_set)
                            else:
                                d_star_lite_initialized_run = False # End change invalidates D* state if not handled above

                        elif setting_target_d_lite and current_algorithm == "D_STAR_LITE":
                            if clicked_node.is_obstacle or clicked_node == grid_instance.start_node :
                                print(f"Cannot set D* Target at ({clicked_node.row}, {clicked_node.col}) as it's an obstacle or start node.")
                            else:
                                old_target_node_instance = grid_instance.end_node # Store before it's changed by set_target_node
                                if grid_instance.set_target_node(clicked_node.row, clicked_node.col): # This sets grid_instance.end_node
                                    print(f"D* Lite Target moved to ({clicked_node.row}, {clicked_node.col}). Replanning...")
                                    if grid_instance.start_node and d_star_lite_initialized_run:
                                        grid_instance.reset_algorithm_states() # Reset visual/temp states, not g/rhs for D*
                                        d_star_lite_target_move_update(grid_instance, grid_instance.end_node, old_target_node_instance,
                                                                       d_star_lite_pq, d_star_lite_open_set_tracker,
                                                                       grid_instance.start_node, heuristic)
                                        path, visited, open_set = d_star_lite_compute_shortest_path(
                                            grid_instance, grid_instance.start_node, grid_instance.end_node, heuristic,
                                            d_star_lite_pq, d_star_lite_open_set_tracker)
                                        start_animation(path, visited, open_set)
                                    elif grid_instance.start_node: # Not initialized, run full D*
                                        d_star_lite_initialized_run = False # Force re-init
                                        # Simulate pressing Enter to run full D*
                                        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
                                    else:
                                        print("Cannot replan D* Lite without a start node.")
                            setting_target_d_lite = False
                            action_taken = True

                        elif painting_terrain_type > 0:
                            if clicked_node != grid_instance.start_node and clicked_node != grid_instance.end_node:
                                terrain_info = TERRAIN_COSTS.get(painting_terrain_type)
                                if terrain_info:
                                    clicked_node.terrain_cost = terrain_info["cost"]
                                    if clicked_node.is_obstacle:
                                        clicked_node.is_obstacle = False
                                        # This is an obstacle change, needs D* update if active
                                        if current_algorithm == "D_STAR_LITE" and d_star_lite_initialized_run and grid_instance.start_node and grid_instance.end_node:
                                            d_star_lite_obstacle_change_update(grid_instance, clicked_node.row, clicked_node.col,
                                                                               d_star_lite_pq, d_star_lite_open_set_tracker,
                                                                               grid_instance.start_node, grid_instance.end_node, heuristic)
                                            # Trigger replan by simulating Enter press
                                            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
                                        else: # Regular update if D* not active or not initialized
                                            grid_instance.update_all_node_neighbors()
                                    print(f"Set terrain at ({clicked_node.row}, {clicked_node.col}) to {terrain_info['name']} (Cost: {terrain_info['cost']})")
                                    action_taken = True # Terrain cost change might affect D* path
                        else: # Default: Toggle obstacle
                            if clicked_node != grid_instance.start_node and clicked_node != grid_instance.end_node:
                                clicked_node.is_obstacle = not clicked_node.is_obstacle
                                if clicked_node.is_obstacle:
                                    clicked_node.terrain_cost = TERRAIN_COSTS[1]["cost"]

                                print(f"Toggled obstacle at ({clicked_node.row}, {clicked_node.col}) to {clicked_node.is_obstacle}")
                                action_taken = True
                                if current_algorithm == "D_STAR_LITE" and d_star_lite_initialized_run and grid_instance.start_node and grid_instance.end_node:
                                    grid_instance.reset_algorithm_states() # Reset visual/temp states
                                    d_star_lite_obstacle_change_update(grid_instance, clicked_node.row, clicked_node.col,
                                                                       d_star_lite_pq, d_star_lite_open_set_tracker,
                                                                       grid_instance.start_node, grid_instance.end_node, heuristic)
                                    # Trigger replan by simulating Enter press if a path was already computed
                                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
                                else: # Regular update if D* not active or not initialized
                                    grid_instance.update_all_node_neighbors()

                        update_caption()


        if animating:
            if animation_data["animation_phase"] == "visited":
                if animation_data["current_visited_step"] < len(animation_data["visited_nodes_in_order"]):
                    node = animation_data["visited_nodes_in_order"][animation_data["current_visited_step"]]
                    if node != grid_instance.start_node and node != grid_instance.end_node: # Use grid_instance
                        node.is_visited_by_algorithm = True
                    animation_data["current_visited_step"] += 1
                    pygame.time.delay(ANIMATION_DELAY_MS)
                else:
                    for node_obj in animation_data["open_set_nodes"]:
                        if node_obj != grid_instance.start_node and node_obj != grid_instance.end_node and \
                           not node_obj.is_visited_by_algorithm and not node_obj.is_part_of_path:
                             node_obj.is_in_open_set_for_algorithm = True
                    animation_data["animation_phase"] = "path"
                    animation_data["current_path_step"] = 0

            elif animation_data["animation_phase"] == "path":
                if animation_data["current_path_step"] < len(animation_data["path_nodes"]):
                    node = animation_data["path_nodes"][animation_data["current_path_step"]]
                    if node != grid_instance.start_node and node != grid_instance.end_node: # Use grid_instance
                        node.is_part_of_path = True
                        node.is_visited_by_algorithm = False
                        node.is_in_open_set_for_algorithm = False
                    animation_data["current_path_step"] += 1
                    pygame.time.delay(ANIMATION_DELAY_MS)
                else:
                    animation_data["animation_phase"] = "finished"
                    pygame.display.set_caption("Visualization Finished. R to Reset. S:Start, E:End, Space:Dijkstra, A:A*")

            elif animation_data["animation_phase"] == "finished":
                animating = False # Stop animation, allow new interactions


        screen.fill(WHITE)
        draw_all_nodes(screen, grid_instance) # Use the new drawing function for all nodes
        draw_grid_lines(screen, GRID_ROWS, GRID_COLS, WINDOW_WIDTH, WINDOW_HEIGHT, CELL_WIDTH, CELL_HEIGHT) # Draw lines on top
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main_gui() # Call the renamed main GUI function
