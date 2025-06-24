import pygame
import json # For saving and loading
import os # For checking file existence

# --- Default Application Configuration ---
# This dictionary serves as the fallback if config.json is missing or invalid.
# It's also used as the template to create a new config.json if one doesn't exist.
DEFAULT_CONFIG = {
    "grid_settings": {
        "rows": 20,
        "cols": 20
    },
    "window_settings": {
        "total_width": 600,
        "total_height": 600,
        "info_panel_height": 40
    },
    "font_settings": {
        "name": None, # Pygame's default system font will be used if None
        "size_info_panel": 24
    },
    "colors": { # Colors are stored as lists, converted to tuples for Pygame
        "background": [255, 255, 255],
        "grid_lines": [200, 200, 200],
        "info_panel_background": [0, 0, 0],
        "info_panel_text": [255, 255, 255],
        "obstacle": [255, 0, 0],
        "start_node": [0, 255, 0],
        "end_node": [0, 0, 255],
        "path": [255, 0, 255],
        "standard_open_set": [255, 165, 0],
        "standard_closed_set": [0, 180, 180],
        "bi_open_fwd": [255, 215, 0],
        "bi_closed_fwd": [173, 216, 230],
        "bi_open_bwd": [255, 182, 193],
        "bi_closed_bwd": [144, 238, 144],
        "bi_meeting_visited": [128, 0, 128],
        "terrain_mud": [139, 69, 19],
        "terrain_water": [30, 144, 255]
    },
    "default_settings": {
        "algorithm_key": "A_STAR", # Internal key for the default algorithm
        "animation_speed_name": "Normal" # User-friendly name for default speed
    }
}

CONFIG_FILE_PATH = "config.json" # Path to the application's configuration file

def load_app_config(filepath: str, defaults: dict) -> dict:
    """
    Loads application settings from a JSON file.

    Starts with a copy of the provided defaults. If the specified filepath
    exists and contains valid JSON, those settings are merged into the defaults.
    If the file does not exist, it's created using the default settings.
    If the file is found but is malformed (invalid JSON) or an IOError occurs,
    a warning is printed, and the function returns the initial defaults.

    The merge is a shallow merge for top-level keys, but for nested dictionaries
    (like 'colors', 'grid_settings', etc.), it performs a one-level update,
    allowing users to override specific nested values without redefining the entire group.

    Args:
        filepath: The path to the configuration JSON file.
        defaults: A dictionary containing the default configuration values.

    Returns:
        A dictionary representing the fully resolved application configuration.
    """
    config = defaults.copy()

    if not os.path.exists(filepath):
        print(f"Info: Configuration file '{filepath}' not found. Using default settings.")
        try:
            with open(filepath, 'w') as f:
                json.dump(defaults, f, indent=4)
            print(f"Info: Created default configuration file at '{filepath}'.")
        except IOError as e:
            print(f"Warning: Could not create default config file '{filepath}': {e}")
        return config

    try:
        with open(filepath, 'r') as f:
            user_config = json.load(f)

        for key, value in user_config.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
        print(f"Info: Loaded configuration from '{filepath}'.")

    except json.JSONDecodeError:
        print(f"Warning: Error decoding JSON from '{filepath}'. Using default settings and original defaults.")
        config = defaults.copy()
    except IOError as e:
        print(f"Warning: Could not read config file '{filepath}': {e}. Using default settings.")
        config = defaults.copy()

    return config

# APP_CONFIG holds the active configuration (loaded from file or defaults).
APP_CONFIG = load_app_config(CONFIG_FILE_PATH, DEFAULT_CONFIG)

# --- Global constants derived from APP_CONFIG ---
# These are set once at startup based on the loaded configuration.
WINDOW_WIDTH = APP_CONFIG["window_settings"]["total_width"]
WINDOW_HEIGHT = APP_CONFIG["window_settings"]["total_height"]
INFO_PANEL_HEIGHT = APP_CONFIG["window_settings"]["info_panel_height"]
GAME_WINDOW_HEIGHT = WINDOW_HEIGHT - INFO_PANEL_HEIGHT

# Default grid dimensions from config, used for initial grid and reset.
GRID_ROWS = APP_CONFIG["grid_settings"]["rows"]
GRID_COLS = APP_CONFIG["grid_settings"]["cols"]


# Core logic imports
from core_logic import (Node, Grid, dijkstra, a_star, run_d_star_lite, heuristic,
                        bidirectional_search, jps_search,
                        d_star_lite_obstacle_change_update, d_star_lite_target_move_update,
                        d_star_lite_initialize, d_star_lite_compute_shortest_path)

# TERRAIN_COSTS define the logical movement cost for terrain types.
# These are not typically user-configurable visual settings, so they remain hardcoded.
# The visual colors for these terrains are in APP_CONFIG["colors"].
TERRAIN_COSTS = {
    1: {"cost": 1.0, "name": "Normal"},
    2: {"cost": 3.0, "name": "Mud"},
    3: {"cost": 5.0, "name": "Water"}
}

# --- GUI specific Helper Functions ---

def save_grid_to_file(grid_instance: Grid, filepath: str) -> bool:
    """
    Saves the current state of the grid (dimensions, start/end nodes, obstacles, terrain)
    to a JSON file. This is for saving specific scenarios, distinct from app_config.json
    which stores application-level settings.

    Args:
        grid_instance: The Grid object whose state is to be saved.
        filepath: The path to the file where the grid configuration will be saved.

    Returns:
        True if saving was successful, False otherwise.
    """
    config_data = {
        "dimensions": {"rows": grid_instance.rows, "cols": grid_instance.cols},
        "start_node": grid_instance.start_node.get_pos() if grid_instance.start_node else None,
        "end_node": grid_instance.end_node.get_pos() if grid_instance.end_node else None,
        "allow_diagonal_movement": grid_instance.allow_diagonal_movement,
        "obstacles": [],
        "terrain_costs": []
    }
    for r in range(grid_instance.rows):
        for c in range(grid_instance.cols):
            node = grid_instance.nodes[r][c]
            if node.is_obstacle:
                config_data["obstacles"].append([r, c])
            if node.terrain_cost != TERRAIN_COSTS[1]["cost"]:
                config_data["terrain_costs"].append({"coords": [r,c], "cost": node.terrain_cost})
    try:
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=4)
        print(f"Grid configuration saved to {filepath}")
        return True
    except IOError as e:
        print(f"Error saving grid to {filepath}: {e}")
        return False

def load_grid_from_file(filepath: str,
                        default_rows_fallback: int, default_cols_fallback: int,
                        initial_cell_width: int, initial_cell_height: int) -> Grid | None:
    """
    Loads a grid state (dimensions, start/end, obstacles, terrain) from a JSON file.
    This is for loading specific scenarios.

    Args:
        filepath: Path to the grid configuration file.
        default_rows_fallback: Fallback rows if not specified in the file (typically from APP_CONFIG).
        default_cols_fallback: Fallback columns if not specified in the file (typically from APP_CONFIG).
        initial_cell_width: Cell width for initializing Nodes if the loaded grid's dimensions differ.
                            This should be based on the overall window size and the loaded grid's rows/cols.
        initial_cell_height: Cell height for initializing Nodes.

    Returns:
        A new Grid object populated from the file, or None if loading fails.
    """
    try:
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}.")
        return None
    except IOError as e:
        print(f"Error loading grid from {filepath}: {e}")
        return None

    grid_dims = loaded_data.get("dimensions", {})
    rows = grid_dims.get("rows", default_rows_fallback)
    cols = grid_dims.get("cols", default_cols_fallback)

    # Calculate cell sizes for this specific grid being loaded, based on main window dimensions
    # and the loaded grid's topology.
    loaded_cell_width = WINDOW_WIDTH // cols
    loaded_cell_height = GAME_WINDOW_HEIGHT // rows

    new_grid = Grid(rows, cols, loaded_cell_width, loaded_cell_height)
    new_grid.set_allow_diagonal_movement(loaded_data.get("allow_diagonal_movement", True))

    start_pos = loaded_data.get("start_node")
    if start_pos and isinstance(start_pos, list) and len(start_pos) == 2:
        r, c = start_pos
        if 0 <= r < new_grid.rows and 0 <= c < new_grid.cols:
            new_grid.start_node = new_grid.nodes[r][c]
            if new_grid.start_node:
                new_grid.start_node.is_obstacle = False
                new_grid.start_node.terrain_cost = TERRAIN_COSTS[1]["cost"]

    end_pos = loaded_data.get("end_node")
    if end_pos and isinstance(end_pos, list) and len(end_pos) == 2:
        r, c = end_pos
        if 0 <= r < new_grid.rows and 0 <= c < new_grid.cols:
            new_grid.end_node = new_grid.nodes[r][c]
            if new_grid.end_node:
                new_grid.end_node.is_obstacle = False
                new_grid.end_node.terrain_cost = TERRAIN_COSTS[1]["cost"]

    obstacles = loaded_data.get("obstacles", [])
    for r, c in obstacles:
        if 0 <= r < new_grid.rows and 0 <= c < new_grid.cols:
            node = new_grid.nodes[r][c]
            if node != new_grid.start_node and node != new_grid.end_node:
                node.is_obstacle = True
                node.terrain_cost = TERRAIN_COSTS[1]["cost"]

    terrain_costs_data = loaded_data.get("terrain_costs", [])
    for item in terrain_costs_data:
        coords, cost_val = item.get("coords"), item.get("cost") # Renamed cost to cost_val
        if coords and isinstance(coords, list) and len(coords) == 2 and isinstance(cost_val, (int, float)):
            r, c = coords
            if 0 <= r < new_grid.rows and 0 <= c < new_grid.cols:
                node = new_grid.nodes[r][c]
                if not node.is_obstacle and node != new_grid.start_node and node != new_grid.end_node:
                    node.terrain_cost = float(cost_val)

    new_grid.update_all_node_neighbors()
    print(f"Grid configuration loaded from {filepath}")
    return new_grid

def get_gui_node_from_mouse_pos(mouse_pos: tuple[int, int], grid_instance: Grid,
                                cell_width: int, cell_height: int) -> Node | None:
    """
    Converts mouse screen coordinates to a grid Node based on current cell dimensions.

    Args:
        mouse_pos: Tuple (x, y) of mouse coordinates.
        grid_instance: The current Grid object.
        cell_width: The current width of a grid cell in pixels.
        cell_height: The current height of a grid cell in pixels.

    Returns:
        The Node at the mouse position, or None if out of bounds or error.
    """
    try:
        if cell_width == 0 or cell_height == 0: return None
        col = mouse_pos[0] // cell_width
        row = mouse_pos[1] // cell_height
        if 0 <= row < grid_instance.rows and 0 <= col < grid_instance.cols:
            return grid_instance.nodes[row][col]
        return None
    except IndexError:
        return None

def draw_grid_lines(surface: pygame.Surface, rows: int, cols: int,
                    window_total_width: int, game_area_height: int,
                    cell_width: int, cell_height: int):
    """
    Draws the grid lines on the given Pygame surface.

    Args:
        surface: The Pygame surface to draw on.
        rows: Number of rows in the current grid.
        cols: Number of columns in the current grid.
        window_total_width: Total width of the window (used for horizontal lines).
        game_area_height: Height of the grid drawing area (used for vertical lines).
        cell_width: Current width of a grid cell.
        cell_height: Current height of a grid cell.
    """
    grid_line_color = tuple(APP_CONFIG["colors"]["grid_lines"])
    for r_idx in range(rows + 1):
        pygame.draw.line(surface, grid_line_color, (0, r_idx * cell_height), (window_total_width, r_idx * cell_height))
    for c_idx in range(cols + 1):
        pygame.draw.line(surface, grid_line_color, (c_idx * cell_width, 0), (c_idx * cell_width, game_area_height))

def draw_all_nodes(surface: pygame.Surface, grid_instance: Grid):
    """
    Draws all nodes of the grid on the surface, coloring them based on their state
    (e.g., obstacle, start/end, path, terrain, visited/open sets).
    Colors are sourced from APP_CONFIG.
    The order of `if/elif` conditions determines drawing precedence for node coloring.
    """
    c_background = tuple(APP_CONFIG["colors"]["background"])
    c_terrain_mud = tuple(APP_CONFIG["colors"]["terrain_mud"])
    c_terrain_water = tuple(APP_CONFIG["colors"]["terrain_water"])
    c_meeting_visited = tuple(APP_CONFIG["colors"]["bi_meeting_visited"])
    c_closed_fwd = tuple(APP_CONFIG["colors"]["bi_closed_fwd"])
    c_closed_bwd = tuple(APP_CONFIG["colors"]["bi_closed_bwd"])
    c_standard_closed = tuple(APP_CONFIG["colors"]["standard_closed_set"])
    c_standard_open = tuple(APP_CONFIG["colors"]["standard_open_set"])
    c_open_fwd = tuple(APP_CONFIG["colors"]["bi_open_fwd"])
    c_open_bwd = tuple(APP_CONFIG["colors"]["bi_open_bwd"])
    c_path = tuple(APP_CONFIG["colors"]["path"])
    c_obstacle = tuple(APP_CONFIG["colors"]["obstacle"])
    c_start_node = tuple(APP_CONFIG["colors"]["start_node"])
    c_end_node = tuple(APP_CONFIG["colors"]["end_node"])

    for r in range(grid_instance.rows):
        for c in range(grid_instance.cols):
            node = grid_instance.nodes[r][c]
            rect = pygame.Rect(node.x, node.y, node.width, node.height)

            node_color = c_background
            if node.terrain_cost == TERRAIN_COSTS[2]["cost"]: node_color = c_terrain_mud
            elif node.terrain_cost == TERRAIN_COSTS[3]["cost"]: node_color = c_terrain_water

            # Algorithm visualization states override terrain colors
            if node.is_in_closed_set_fwd and node.is_in_closed_set_bwd:
                node_color = c_meeting_visited
            elif node.is_in_closed_set_fwd:
                node_color = c_closed_fwd
            elif node.is_in_closed_set_bwd:
                node_color = c_closed_bwd
            elif node.is_visited_by_algorithm:
                node_color = c_standard_closed

            if node.is_in_open_set_fwd and node.is_in_open_set_bwd:
                node_color = c_standard_open
            elif node.is_in_open_set_fwd:
                node_color = c_open_fwd
            elif node.is_in_open_set_bwd:
                node_color = c_open_bwd
            elif node.is_in_open_set_for_algorithm:
                node_color = c_standard_open

            # Path, obstacle, start/end have highest precedence
            if node.is_part_of_path: node_color = c_path
            if node.is_obstacle: node_color = c_obstacle
            if node == grid_instance.start_node: node_color = c_start_node
            if node == grid_instance.end_node: node_color = c_end_node

            pygame.draw.rect(surface, node_color, rect)

def draw_info_panel(surface: pygame.Surface, font: pygame.font.Font,
                    algo_name: str, path_len: int | None, visited_count: int | None, speed_name: str,
                    game_area_height: int, window_total_width: int, info_panel_h: int):
    """
    Draws the information panel at the bottom of the screen.
    Displays current algorithm, path length, visited nodes count, and animation speed.
    Uses colors from APP_CONFIG and dimensions passed as parameters.
    """
    panel_y = game_area_height
    panel_bg_color = tuple(APP_CONFIG["colors"]["info_panel_background"])
    panel_text_color = tuple(APP_CONFIG["colors"]["info_panel_text"])

    pygame.draw.rect(surface, panel_bg_color, (0, panel_y, window_total_width, info_panel_h))

    algo_text = f"Algo: {algo_name}"
    speed_text = f"Speed: {speed_name} (+/-)"
    path_text = f"Path: {path_len if path_len is not None and path_len > 0 else 'N/A'}"
    visited_text = f"Visited: {visited_count if visited_count is not None else 'N/A'}"

    algo_surf = font.render(algo_text, True, panel_text_color)
    speed_surf = font.render(speed_text, True, panel_text_color)
    path_surf = font.render(path_text, True, panel_text_color)
    visited_surf = font.render(visited_text, True, panel_text_color)

    surface.blit(algo_surf, (5, panel_y + 5))
    surface.blit(speed_surf, (algo_surf.get_width() + 15, panel_y + 5))
    surface.blit(path_surf, (window_total_width - visited_surf.get_width() - path_surf.get_width() - 15, panel_y + 5))
    surface.blit(visited_surf, (window_total_width - visited_surf.get_width() - 5, panel_y + 5))


def main_gui():
    """
    Main function to initialize Pygame, set up the GUI, handle user input,
    run pathfinding algorithms, and manage the application's main event loop.
    Loads application settings from `config.json` or uses defaults.
    """
    pygame.init()
    pygame.font.init()

    # Load dimensions and settings from APP_CONFIG (sourced from config.json or defaults)
    conf_grid_rows = APP_CONFIG["grid_settings"]["rows"]
    conf_grid_cols = APP_CONFIG["grid_settings"]["cols"]
    conf_window_width = APP_CONFIG["window_settings"]["total_width"]
    conf_window_height = APP_CONFIG["window_settings"]["total_height"]
    conf_info_panel_height = APP_CONFIG["window_settings"]["info_panel_height"]
    conf_game_window_height = conf_window_height - conf_info_panel_height

    # Calculate cell dimensions for the *initial* grid based on configured rows/cols.
    initial_cell_width = conf_window_width // conf_grid_cols
    initial_cell_height = conf_game_window_height // conf_grid_rows

    # Load font settings from APP_CONFIG
    font_name_config = APP_CONFIG["font_settings"]["name"]
    font_size_info_config = APP_CONFIG["font_settings"]["size_info_panel"]
    ui_font = pygame.font.Font(font_name_config, font_size_info_config)

    screen = pygame.display.set_mode((conf_window_width, conf_window_height))
    clock = pygame.time.Clock()

    # Create the initial grid instance using configured dimensions and calculated initial cell sizes
    grid_instance = Grid(conf_grid_rows, conf_grid_cols, initial_cell_width, initial_cell_height)

    # current_cell_width/height are used for mouse interactions and drawing grid lines.
    # They are based on the *active* grid_instance's dimensions and the fixed game window area.
    current_cell_width = conf_window_width // grid_instance.cols
    current_cell_height = conf_game_window_height // grid_instance.rows

    # Application state variables
    setting_start, setting_end, setting_target_d_lite = False, False, False
    painting_terrain_type = 0 # 0: off, 1-3: terrain types

    d_star_lite_pq, d_star_lite_open_set_tracker = [], set()
    d_star_lite_initialized_run = False

    last_path_length, last_visited_count = None, None
    current_algorithm = APP_CONFIG["default_settings"]["algorithm_key"]

    ALGORITHMS = {
        "A_STAR": a_star, "DIJKSTRA": dijkstra, "D_STAR_LITE": run_d_star_lite,
        "BIDIRECTIONAL": bidirectional_search, "JPS": jps_search
    }
    ALGO_NAMES = {
        "A_STAR": "A*", "DIJKSTRA": "Dijkstra", "D_STAR_LITE": "D* Lite",
        "BIDIRECTIONAL": "Bidirectional", "JPS": "JPS"
    }
    ANIMATION_SPEED_SETTINGS = [
        {"name": "Instant", "delay": 0}, {"name": "Fast", "delay": 10},
        {"name": "Normal", "delay": 25}, {"name": "Slow", "delay": 50},
        {"name": "Very Slow", "delay": 100}
    ]

    # Initialize animation speed from config
    default_speed_name = APP_CONFIG["default_settings"]["animation_speed_name"]
    current_speed_index = 0
    for i, speed_setting in enumerate(ANIMATION_SPEED_SETTINGS):
        if speed_setting["name"] == default_speed_name:
            current_speed_index = i
            break
    animation_delay_ms = ANIMATION_SPEED_SETTINGS[current_speed_index]["delay"]

    animating = False
    animation_data = {
        "visited_fwd_list": [], "visited_bwd_list": [],
        "open_fwd_list": [], "open_bwd_list": [],
        "path_nodes": [],
        "current_visited_fwd_step": 0, "current_visited_bwd_step": 0,
        "current_path_step": 0,
        "animation_phase": "stopped"
    }

    def update_gui_caption():
        """Updates the Pygame window caption based on current mode and settings."""
        nonlocal setting_start, setting_end, setting_target_d_lite, painting_terrain_type
        mode_text = "Default (Obstacles)"
        if setting_start: mode_text = "Set Start"
        elif setting_end: mode_text = "Set End"
        elif setting_target_d_lite: mode_text = "Set D* Target"
        elif painting_terrain_type > 0:
            mode_text = f"Paint: {TERRAIN_COSTS[painting_terrain_type]['name']}"

        diag_status = "ON" if grid_instance.allow_diagonal_movement else "OFF"

        caption_text = f"Mode: {mode_text} | Diag: {diag_status}"
        bindings = " | Keys: S,E,D,R,C | Algos:K,A,L,B,J | Terrain:1-3 | Speed:+/- | F5/F6:Save/Load | Enter:Run"
        pygame.display.set_caption(caption_text + bindings)

    update_gui_caption()

    def start_animation_enhanced(path: list[Node] | None,
                                 visited_fwd: list[Node], open_fwd: list[Node],
                                 visited_bwd: list[Node] | None = None, open_bwd: list[Node] | None = None):
        """
        Initializes and starts the step-by-step visualization of an algorithm's execution.

        Args:
            path: The final path found.
            visited_fwd: Nodes closed by forward search (or main search for uni-directional).
            open_fwd: Nodes in open set of forward search at termination.
            visited_bwd: Nodes closed by backward search (for bidirectional).
            open_bwd: Nodes in open set of backward search at termination (for bidirectional).
        """
        nonlocal animating, animation_data, current_algorithm
        grid_instance.clear_visualizations()

        animation_data["path_nodes"] = path if path else []
        animation_data["visited_fwd_list"] = visited_fwd
        animation_data["visited_bwd_list"] = visited_bwd if visited_bwd else []
        animation_data["open_fwd_list"] = open_fwd
        animation_data["open_bwd_list"] = open_bwd if open_bwd else []

        animation_data["current_visited_fwd_step"] = 0
        animation_data["current_visited_bwd_step"] = 0
        animation_data["current_path_step"] = 0

        if current_algorithm == "BIDIRECTIONAL":
            if animation_data["visited_fwd_list"]: animation_data["animation_phase"] = "visited_fwd"
            elif animation_data["visited_bwd_list"]: animation_data["animation_phase"] = "visited_bwd"
            else: animation_data["animation_phase"] = "path"
        else:
            if animation_data["visited_fwd_list"]: animation_data["animation_phase"] = "visited_fwd"
            else: animation_data["animation_phase"] = "path"

        animating = True

    # Main application loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if not animating:
                if event.type == pygame.KEYDOWN:
                    # Algorithm selection
                    if event.key == pygame.K_a: current_algorithm = "A_STAR"
                    elif event.key == pygame.K_l: current_algorithm = "D_STAR_LITE"
                    elif event.key == pygame.K_k: current_algorithm = "DIJKSTRA"
                    elif event.key == pygame.K_b: current_algorithm = "BIDIRECTIONAL"
                    elif event.key == pygame.K_j: current_algorithm = "JPS"

                    if event.key in [pygame.K_a, pygame.K_l, pygame.K_k, pygame.K_b, pygame.K_j]:
                        setting_target_d_lite = False; painting_terrain_type = 0
                        setting_start = False; setting_end = False

                    elif event.key == pygame.K_d: # Toggle diagonal movement
                        grid_instance.set_allow_diagonal_movement(not grid_instance.allow_diagonal_movement)
                        print(f"Diagonal movement {'enabled' if grid_instance.allow_diagonal_movement else 'disabled'}")

                    # Terrain painting
                    elif event.key == pygame.K_1: painting_terrain_type = 1
                    elif event.key == pygame.K_2: painting_terrain_type = 2
                    elif event.key == pygame.K_3: painting_terrain_type = 3
                    if event.key in [pygame.K_1, pygame.K_2, pygame.K_3]:
                        setting_start, setting_end, setting_target_d_lite = False, False, False
                        print(f"Set to paint terrain: {TERRAIN_COSTS[painting_terrain_type]['name']}")
                    elif event.key == pygame.K_0 or event.key == pygame.K_ESCAPE:
                        painting_terrain_type = 0
                        setting_start, setting_end, setting_target_d_lite = False, False, False # Also exit other modes
                        print("Terrain painting mode OFF. Default obstacle mode.")

                    # Animation speed
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        current_speed_index = max(0, current_speed_index - 1)
                        animation_delay_ms = ANIMATION_SPEED_SETTINGS[current_speed_index]["delay"]
                    elif event.key == pygame.K_MINUS:
                        current_speed_index = min(len(ANIMATION_SPEED_SETTINGS) - 1, current_speed_index + 1)
                        animation_delay_ms = ANIMATION_SPEED_SETTINGS[current_speed_index]["delay"]

                    # Save/Load Grid State (from grid_config.json)
                    elif event.key == pygame.K_F5: save_grid_to_file(grid_instance, "grid_config.json")
                    elif event.key == pygame.K_F6:
                        loaded_grid = load_grid_from_file("grid_config.json",
                                                        APP_CONFIG["grid_settings"]["rows"],
                                                        APP_CONFIG["grid_settings"]["cols"],
                                                        conf_window_width // APP_CONFIG["grid_settings"]["cols"],
                                                        conf_game_window_height // APP_CONFIG["grid_settings"]["rows"])
                        if loaded_grid:
                            grid_instance = loaded_grid
                            current_cell_width = conf_window_width // grid_instance.cols
                            current_cell_height = conf_game_window_height // grid_instance.rows

                            setting_start,setting_end,setting_target_d_lite,painting_terrain_type = False,False,False,0
                            d_star_lite_initialized_run = False
                            d_star_lite_pq.clear(); d_star_lite_open_set_tracker.clear()
                            animating = False; grid_instance.reset_algorithm_states()
                            last_path_length, last_visited_count = None, None
                            print("Grid loaded.")
                        else: print("Failed to load grid.")

                    # Set Start/End/D* Target modes
                    elif event.key == pygame.K_s: setting_start = True; setting_end,setting_target_d_lite,painting_terrain_type = False,False,0
                    elif event.key == pygame.K_e: setting_end = True; setting_start,setting_target_d_lite,painting_terrain_type = False,False,0
                    elif event.key == pygame.K_t and current_algorithm == "D_STAR_LITE":
                        setting_target_d_lite = True; setting_start,setting_end,painting_terrain_type = False,False,0

                    # Run Algorithm
                    elif event.key == pygame.K_RETURN:
                        if grid_instance.start_node and grid_instance.end_node:
                            grid_instance.reset_algorithm_states()
                            grid_instance.update_all_node_neighbors()

                            algo_func = ALGORITHMS[current_algorithm]
                            print(f"Running {ALGO_NAMES[current_algorithm]}...")

                            path_res, visited_fwd_res, open_fwd_res = None, [], []
                            visited_bwd_res, open_bwd_res = None, None

                            if current_algorithm == "D_STAR_LITE":
                                d_star_lite_pq.clear()
                                d_star_lite_open_set_tracker.clear()
                                d_star_lite_initialize(grid_instance, grid_instance.start_node, grid_instance.end_node, heuristic, d_star_lite_pq, d_star_lite_open_set_tracker)
                                path_res, visited_fwd_res, open_fwd_res = d_star_lite_compute_shortest_path(
                                    grid_instance, grid_instance.start_node, grid_instance.end_node, heuristic,
                                    d_star_lite_pq, d_star_lite_open_set_tracker
                                )
                                d_star_lite_initialized_run = True
                            elif current_algorithm == "BIDIRECTIONAL":
                                path_res, visited_fwd_res, visited_bwd_res, open_fwd_res, open_bwd_res = algo_func(grid_instance, grid_instance.start_node, grid_instance.end_node)
                            else:
                                path_res, visited_fwd_res, open_fwd_res = algo_func(grid_instance, grid_instance.start_node, grid_instance.end_node)

                            start_animation_enhanced(path_res, visited_fwd_res, open_fwd_res, visited_bwd_res, open_bwd_res)

                            if path_res:
                                last_path_length = len(path_res)
                                unique_visited_nodes = set(visited_fwd_res)
                                if visited_bwd_res: unique_visited_nodes.update(visited_bwd_res)
                                last_visited_count = len(unique_visited_nodes)
                                print(f"{ALGO_NAMES[current_algorithm]} Path: {last_path_length}, Visited: {last_visited_count}")
                            else:
                                last_path_length = 0
                                unique_visited_nodes = set(visited_fwd_res)
                                if visited_bwd_res: unique_visited_nodes.update(visited_bwd_res)
                                last_visited_count = len(unique_visited_nodes)
                                print(f"{ALGO_NAMES[current_algorithm]} No path. Visited: {last_visited_count}")
                        else:
                            print("Error: Set Start and End nodes before running an algorithm.")
                            last_path_length, last_visited_count = None, None

                    # Reset Grid
                    elif event.key == pygame.K_r:
                        grid_instance = Grid(APP_CONFIG["grid_settings"]["rows"],
                                             APP_CONFIG["grid_settings"]["cols"],
                                             conf_window_width // APP_CONFIG["grid_settings"]["cols"],
                                             conf_game_window_height // APP_CONFIG["grid_settings"]["rows"])
                        current_cell_width = conf_window_width // grid_instance.cols
                        current_cell_height = conf_game_window_height // grid_instance.rows
                        setting_start,setting_end,setting_target_d_lite,painting_terrain_type = False,False,False,0
                        d_star_lite_initialized_run = False
                        d_star_lite_pq.clear(); d_star_lite_open_set_tracker.clear()
                        animating = False; animation_data["animation_phase"] = "stopped"
                        last_path_length, last_visited_count = None, None
                        print("Grid reset to default configuration.")

                    # Clear Visualizations
                    elif event.key == pygame.K_c:
                        if animating:
                            animating = False; animation_data["animation_phase"] = "stopped"
                        grid_instance.clear_visualizations()
                        last_path_length, last_visited_count = None, None
                        print("Path visualizations cleared.")

                    update_gui_caption()

                # Mouse click handling
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if mouse_pos[1] >= conf_game_window_height: continue

                    clicked_node = get_gui_node_from_mouse_pos(mouse_pos, grid_instance, current_cell_width, current_cell_height)
                    if clicked_node:
                        if setting_start:
                            if grid_instance.start_node: grid_instance.start_node.terrain_cost = TERRAIN_COSTS[1]["cost"]
                            grid_instance.start_node = clicked_node
                            clicked_node.is_obstacle = False
                            clicked_node.terrain_cost = TERRAIN_COSTS[1]["cost"]
                            setting_start = False; d_star_lite_initialized_run = False
                        elif setting_end:
                            old_end_node = grid_instance.end_node
                            if grid_instance.end_node: grid_instance.end_node.terrain_cost = TERRAIN_COSTS[1]["cost"]
                            grid_instance.end_node = clicked_node
                            clicked_node.is_obstacle = False
                            clicked_node.terrain_cost = TERRAIN_COSTS[1]["cost"]
                            setting_end = False
                            if current_algorithm == "D_STAR_LITE" and d_star_lite_initialized_run and grid_instance.start_node:
                                d_star_lite_target_move_update(grid_instance, grid_instance.end_node, old_end_node, d_star_lite_pq, d_star_lite_open_set_tracker, grid_instance.start_node, heuristic)
                                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
                            else: d_star_lite_initialized_run = False
                        elif setting_target_d_lite and current_algorithm == "D_STAR_LITE":
                            if not clicked_node.is_obstacle and clicked_node != grid_instance.start_node:
                                old_target_node_d_lite = grid_instance.end_node
                                grid_instance.set_target_node(clicked_node.row, clicked_node.col)
                                if grid_instance.start_node and d_star_lite_initialized_run:
                                    d_star_lite_target_move_update(grid_instance, grid_instance.end_node, old_target_node_d_lite, d_star_lite_pq, d_star_lite_open_set_tracker, grid_instance.start_node, heuristic)
                                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
                                elif grid_instance.start_node :
                                     d_star_lite_initialized_run = False
                                     pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
                            setting_target_d_lite = False
                        elif painting_terrain_type > 0:
                            if clicked_node != grid_instance.start_node and clicked_node != grid_instance.end_node:
                                clicked_node.terrain_cost = TERRAIN_COSTS[painting_terrain_type]["cost"]
                                if clicked_node.is_obstacle:
                                    clicked_node.is_obstacle = False
                                    if current_algorithm == "D_STAR_LITE" and d_star_lite_initialized_run:
                                        d_star_lite_obstacle_change_update(grid_instance, clicked_node.row, clicked_node.col, d_star_lite_pq, d_star_lite_open_set_tracker, grid_instance.start_node, grid_instance.end_node, heuristic)
                                        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
                                    else: grid_instance.update_all_node_neighbors()
                        else: # Default mode: Toggle obstacle
                            if clicked_node != grid_instance.start_node and clicked_node != grid_instance.end_node:
                                clicked_node.is_obstacle = not clicked_node.is_obstacle
                                if clicked_node.is_obstacle: clicked_node.terrain_cost = TERRAIN_COSTS[1]["cost"]
                                if current_algorithm == "D_STAR_LITE" and d_star_lite_initialized_run:
                                    d_star_lite_obstacle_change_update(grid_instance, clicked_node.row, clicked_node.col, d_star_lite_pq, d_star_lite_open_set_tracker, grid_instance.start_node, grid_instance.end_node, heuristic)
                                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
                                else: grid_instance.update_all_node_neighbors()
                        update_gui_caption()

        # Animation Loop Logic
        if animating:
            current_phase = animation_data["animation_phase"]

            if current_phase == "visited_fwd":
                if animation_data["current_visited_fwd_step"] < len(animation_data["visited_fwd_list"]):
                    node = animation_data["visited_fwd_list"][animation_data["current_visited_fwd_step"]]
                    if node != grid_instance.start_node and node != grid_instance.end_node:
                        if current_algorithm == "BIDIRECTIONAL": node.is_in_closed_set_fwd = True
                        else: node.is_visited_by_algorithm = True
                    animation_data["current_visited_fwd_step"] += 1
                    if animation_delay_ms > 0: pygame.time.delay(animation_delay_ms)
                else:
                    if current_algorithm == "BIDIRECTIONAL" and animation_data["visited_bwd_list"]:
                        animation_data["animation_phase"] = "visited_bwd"
                    else:
                        animation_data["animation_phase"] = "set_open"

            elif current_phase == "visited_bwd":
                if animation_data["current_visited_bwd_step"] < len(animation_data["visited_bwd_list"]):
                    node = animation_data["visited_bwd_list"][animation_data["current_visited_bwd_step"]]
                    if node != grid_instance.start_node and node != grid_instance.end_node:
                        node.is_in_closed_set_bwd = True
                    animation_data["current_visited_bwd_step"] += 1
                    if animation_delay_ms > 0: pygame.time.delay(animation_delay_ms)
                else:
                    animation_data["animation_phase"] = "set_open"

            elif current_phase == "set_open":
                if current_algorithm == "BIDIRECTIONAL":
                    for node_obj in animation_data["open_fwd_list"]:
                        if not node_obj.is_in_closed_set_fwd and not node_obj.is_in_closed_set_bwd and not node_obj.is_part_of_path:
                            node_obj.is_in_open_set_fwd = True
                    for node_obj in animation_data["open_bwd_list"]:
                        if not node_obj.is_in_closed_set_fwd and not node_obj.is_in_closed_set_bwd and not node_obj.is_part_of_path:
                            node_obj.is_in_open_set_bwd = True
                else:
                     for node_obj in animation_data.get("open_fwd_list", []): # Use open_fwd_list which holds the final open set from algos
                        if not node_obj.is_visited_by_algorithm and not node_obj.is_part_of_path:
                             node_obj.is_in_open_set_for_algorithm = True
                animation_data["animation_phase"] = "path"
                animation_data["current_path_step"] = 0
                if animation_delay_ms > 0: pygame.time.delay(animation_delay_ms)

            elif current_phase == "path":
                if animation_data["current_path_step"] < len(animation_data["path_nodes"]):
                    node = animation_data["path_nodes"][animation_data["current_path_step"]]
                    if node != grid_instance.start_node and node != grid_instance.end_node:
                        node.is_part_of_path = True
                        node.is_visited_by_algorithm = False; node.is_in_open_set_for_algorithm = False
                        node.is_in_closed_set_fwd = False; node.is_in_closed_set_bwd = False
                        node.is_in_open_set_fwd = False; node.is_in_open_set_bwd = False
                    animation_data["current_path_step"] += 1
                    if animation_delay_ms > 0: pygame.time.delay(animation_delay_ms)
                else:
                    animation_data["animation_phase"] = "finished"

            elif current_phase == "finished":
                animating = False
                update_gui_caption()

        # Drawing operations (every frame)
        screen.fill(tuple(APP_CONFIG["colors"]["background"]))
        draw_all_nodes(screen, grid_instance)
        draw_grid_lines(screen, grid_instance.rows, grid_instance.cols,
                        conf_window_width, conf_game_window_height,
                        current_cell_width, current_cell_height)

        draw_info_panel(screen, ui_font, ALGO_NAMES[current_algorithm],
                        last_path_length, last_visited_count,
                        ANIMATION_SPEED_SETTINGS[current_speed_index]["name"],
                        conf_game_window_height, conf_window_width, conf_info_panel_height)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main_gui()
