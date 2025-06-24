import pygame
import json # For saving and loading
import os # For checking file existence

# --- Default Application Configuration ---
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
        "name": None,
        "size_info_panel": 24
    },
    "colors": {
        "background": [255, 255, 255], # WHITE
        "grid_lines": [200, 200, 200], # GREY
        "info_panel_background": [0, 0, 0],   # BLACK
        "info_panel_text": [255, 255, 255], # WHITE
        "obstacle": [255, 0, 0],           # RED_OBSTACLE
        "start_node": [0, 255, 0],         # GREEN_START
        "end_node": [0, 0, 255],           # BLUE_END
        "path": [255, 0, 255],             # MAGENTA_PATH
        "standard_open_set": [255, 165, 0],  # ORANGE
        "standard_closed_set": [0, 180, 180], # CYAN
        "bi_open_fwd": [255, 215, 0],      # GOLD
        "bi_closed_fwd": [173, 216, 230],   # LIGHT BLUE
        "bi_open_bwd": [255, 182, 193],    # LIGHT PINK
        "bi_closed_bwd": [144, 238, 144],   # LIGHT GREEN
        "bi_meeting_visited": [128, 0, 128],# PURPLE
        "terrain_mud": [139, 69, 19],
        "terrain_water": [30, 144, 255]
    },
    "default_settings": {
        "algorithm_key": "A_STAR",
        "animation_speed_name": "Normal"
    }
}

CONFIG_FILE_PATH = "config.json"

def load_app_config(filepath: str, defaults: dict) -> dict:
    config = defaults.copy() # Start with defaults

    if not os.path.exists(filepath):
        print(f"Info: Configuration file '{filepath}' not found. Using default settings.")
        # Optionally, create a default config file here if it doesn't exist
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

        # Deep merge user_config into config (simple one-level merge for nested dicts)
        for key, value in user_config.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value) # Update nested dicts like 'colors', 'grid_settings'
            else:
                config[key] = value # Overwrite top-level keys or non-dict values
        print(f"Info: Loaded configuration from '{filepath}'.")

    except json.JSONDecodeError:
        print(f"Warning: Error decoding JSON from '{filepath}'. Using default settings.")
        # config is already defaults, so just return it
    except IOError as e:
        print(f"Warning: Could not read config file '{filepath}': {e}. Using default settings.")

    return config

# Load configuration at startup
APP_CONFIG = load_app_config(CONFIG_FILE_PATH, DEFAULT_CONFIG)

# --- Constants (now potentially from APP_CONFIG) ---
# These will be replaced in a later step by direct usage of APP_CONFIG values
WINDOW_WIDTH = APP_CONFIG["window_settings"]["total_width"]
WINDOW_HEIGHT = APP_CONFIG["window_settings"]["total_height"]
INFO_PANEL_HEIGHT = APP_CONFIG["window_settings"]["info_panel_height"]
GAME_WINDOW_HEIGHT = WINDOW_HEIGHT - INFO_PANEL_HEIGHT

GRID_ROWS = APP_CONFIG["grid_settings"]["rows"]
GRID_COLS = APP_CONFIG["grid_settings"]["cols"]

# CELL_WIDTH and CELL_HEIGHT will be calculated in main_gui after pygame.init
# CELL_WIDTH = WINDOW_WIDTH // GRID_COLS
# CELL_HEIGHT = GAME_WINDOW_HEIGHT // GRID_ROWS

# --- Colors (from APP_CONFIG) ---
# Access via APP_CONFIG["colors"]["color_name"]
# Example: C_BACKGROUND = tuple(APP_CONFIG["colors"]["background"])

# Core logic imports
from core_logic import (Node, Grid, dijkstra, a_star, run_d_star_lite, heuristic,
                        bidirectional_search, jps_search,
                        d_star_lite_obstacle_change_update, d_star_lite_target_move_update,
                        d_star_lite_initialize, d_star_lite_compute_shortest_path)

# Terrain Colors & Costs
# TERRAIN_DEFAULT_COLOR will use APP_CONFIG["colors"]["background"]
# TERRAIN_MUD_COLOR will use APP_CONFIG["colors"]["terrain_mud"]
# TERRAIN_WATER_COLOR will use APP_CONFIG["colors"]["terrain_water"]

TERRAIN_COSTS = { # This remains as it defines costs, not just colors.
    1: {"cost": 1.0, "name": "Normal"},
    2: {"cost": 3.0, "name": "Mud"},
    3: {"cost": 5.0, "name": "Water"}
}

# --- GUI specific Helper Functions ---

def save_grid_to_file(grid_instance: Grid, filepath: str):
    config = {
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
                config["obstacles"].append([r, c])
            if node.terrain_cost != TERRAIN_COSTS[1]["cost"]: # Compare against default terrain cost
                config["terrain_costs"].append({"coords": [r,c], "cost": node.terrain_cost})
    try:
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Grid configuration saved to {filepath}")
        return True
    except IOError as e:
        print(f"Error saving grid to {filepath}: {e}")
        return False

def load_grid_from_file(filepath: str, default_rows: int, default_cols: int, cell_width: int, cell_height: int) -> Grid | None:
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}.")
        return None
    except IOError as e:
        print(f"Error loading grid from {filepath}: {e}")
        return None

    # Use grid dimensions from APP_CONFIG if not loading from file, or from file if they exist there?
    # The current grid_config.json does store dimensions. Let's assume it's for that specific saved grid.
    # The load_grid_from_file parameters default_rows/cols are now effectively from APP_CONFIG.
    new_grid = Grid(config.get("dimensions", {}).get("rows", default_rows),
                    config.get("dimensions", {}).get("cols", default_cols),
                    cell_width, cell_height)
    new_grid.set_allow_diagonal_movement(config.get("allow_diagonal_movement", True))
    start_pos = config.get("start_node")
    if start_pos and isinstance(start_pos, list) and len(start_pos) == 2:
        r, c = start_pos
        if 0 <= r < new_grid.rows and 0 <= c < new_grid.cols:
            new_grid.start_node = new_grid.nodes[r][c]
            new_grid.start_node.is_obstacle = False
            new_grid.start_node.terrain_cost = TERRAIN_COSTS[1]["cost"]
    end_pos = config.get("end_node")
    if end_pos and isinstance(end_pos, list) and len(end_pos) == 2:
        r, c = end_pos
        if 0 <= r < new_grid.rows and 0 <= c < new_grid.cols:
            new_grid.end_node = new_grid.nodes[r][c]
            new_grid.end_node.is_obstacle = False
            new_grid.end_node.terrain_cost = TERRAIN_COSTS[1]["cost"]
    obstacles = config.get("obstacles", [])
    for r, c in obstacles:
        if 0 <= r < new_grid.rows and 0 <= c < new_grid.cols:
            node = new_grid.nodes[r][c]
            if node != new_grid.start_node and node != new_grid.end_node:
                node.is_obstacle = True
                node.terrain_cost = TERRAIN_COSTS[1]["cost"]
    terrain_costs_data = config.get("terrain_costs", [])
    for item in terrain_costs_data:
        coords, cost = item.get("coords"), item.get("cost")
        if coords and isinstance(coords, list) and len(coords) == 2 and isinstance(cost, (int, float)):
            r, c = coords
            if 0 <= r < new_grid.rows and 0 <= c < new_grid.cols:
                node = new_grid.nodes[r][c]
                if not node.is_obstacle and node != new_grid.start_node and node != new_grid.end_node:
                    node.terrain_cost = float(cost)
    new_grid.update_all_node_neighbors()
    print(f"Grid configuration loaded from {filepath}")
    return new_grid

def get_gui_node_from_mouse_pos(mouse_pos, grid_instance: Grid, cell_width, cell_height):
    try:
        col = mouse_pos[0] // cell_width
        row = mouse_pos[1] // cell_height
        if 0 <= row < grid_instance.rows and 0 <= col < grid_instance.cols:
            return grid_instance.nodes[row][col]
        return None
    except IndexError:
        return None

def draw_grid_lines(surface, rows, cols, window_total_width, game_area_height, cell_width, cell_height): # Renamed params for clarity
    grid_line_color = tuple(APP_CONFIG["colors"]["grid_lines"])
    for r_idx in range(rows + 1): # Renamed r to r_idx
        pygame.draw.line(surface, grid_line_color, (0, r_idx * cell_height), (window_total_width, r_idx * cell_height))
    for c_idx in range(cols + 1): # Renamed c to c_idx
        pygame.draw.line(surface, grid_line_color, (c_idx * cell_width, 0), (c_idx * cell_width, game_area_height))

def draw_all_nodes(surface, grid_instance: Grid):
    # Color assignments using APP_CONFIG
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

            node_color = c_background # Default
            if node.terrain_cost == TERRAIN_COSTS[2]["cost"]: node_color = c_terrain_mud
            elif node.terrain_cost == TERRAIN_COSTS[3]["cost"]: node_color = c_terrain_water

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

            if node.is_part_of_path: node_color = c_path
            if node.is_obstacle: node_color = c_obstacle
            if node == grid_instance.start_node: node_color = c_start_node
            if node == grid_instance.end_node: node_color = c_end_node

            pygame.draw.rect(surface, node_color, rect)

def draw_info_panel(surface, font, algo_name, path_len, visited_count, speed_name,
                    game_window_h, window_total_w, info_panel_h): # Pass dimensions
    panel_y = game_window_h
    panel_bg_color = tuple(APP_CONFIG["colors"]["info_panel_background"])
    panel_text_color = tuple(APP_CONFIG["colors"]["info_panel_text"])
    pygame.draw.rect(surface, panel_bg_color, (0, panel_y, window_total_w, info_panel_h))

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
    surface.blit(path_surf, (window_total_w - visited_surf.get_width() - path_surf.get_width() - 15, panel_y + 5))
    surface.blit(visited_surf, (window_total_w - visited_surf.get_width() - 5, panel_y + 5))


def main_gui():
    pygame.init()
    pygame.font.init()

    # Get dimensions from APP_CONFIG
    conf_grid_rows = APP_CONFIG["grid_settings"]["rows"]
    conf_grid_cols = APP_CONFIG["grid_settings"]["cols"]
    conf_window_width = APP_CONFIG["window_settings"]["total_width"]
    conf_window_height = APP_CONFIG["window_settings"]["total_height"]
    conf_info_panel_height = APP_CONFIG["window_settings"]["info_panel_height"]
    conf_game_window_height = conf_window_height - conf_info_panel_height

    conf_cell_width = conf_window_width // conf_grid_cols
    conf_cell_height = conf_game_window_height // conf_grid_rows

    font_name = APP_CONFIG["font_settings"]["name"]
    font_size_info = APP_CONFIG["font_settings"]["size_info_panel"]
    ui_font = pygame.font.Font(font_name, font_size_info)


    screen = pygame.display.set_mode((conf_window_width, conf_window_height))
    clock = pygame.time.Clock()

    grid_instance = Grid(conf_grid_rows, conf_grid_cols, conf_cell_width, conf_cell_height)

    # Dynamically calculated cell dimensions based on current grid and window config
    current_cell_width = conf_window_width // grid_instance.cols
    current_cell_height = conf_game_window_height // grid_instance.rows

    setting_start, setting_end, setting_target_d_lite, painting_terrain_type = False, False, False, 0
    d_star_lite_pq, d_star_lite_open_set_tracker, d_star_lite_initialized_run = [], set(), False
    last_path_length, last_visited_count = None, None
    current_algorithm = APP_CONFIG["default_settings"]["algorithm_key"]

    ALGORITHMS = {
        "A_STAR": a_star, "DIJKSTRA": dijkstra, "D_STAR_LITE": run_d_star_lite,
        "BIDIRECTIONAL": bidirectional_search, "JPS": jps_search
    }
    ALGO_NAMES = { # These are for display, can remain hardcoded or moved to config if desired (later)
        "A_STAR": "A*", "DIJKSTRA": "Dijkstra", "D_STAR_LITE": "D* Lite",
        "BIDIRECTIONAL": "Bidirectional", "JPS": "JPS"
    }
    ANIMATION_SPEED_SETTINGS = [ # Structure for speeds, can remain hardcoded
        {"name": "Instant", "delay": 0}, {"name": "Fast", "delay": 10},
        {"name": "Normal", "delay": 25}, {"name": "Slow", "delay": 50},
        {"name": "Very Slow", "delay": 100}
    ]

    # Set initial speed from config
    default_speed_name = APP_CONFIG["default_settings"]["animation_speed_name"]
    current_speed_index = 0 # Default to first speed if name not found
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
        "current_path_step": 0, "animation_phase": "stopped"
    }

    def update_gui_caption(): # Renamed to avoid conflict with any Pygame internal
        nonlocal setting_start, setting_end, setting_target_d_lite, current_algorithm, painting_terrain_type, current_speed_index
        mode_text = ""
        if setting_start: mode_text = "Set Start"
        elif setting_end: mode_text = "Set End"
        elif setting_target_d_lite: mode_text = "Set D* Target"
        elif painting_terrain_type > 0:
            mode_text = f"Paint: {TERRAIN_COSTS[painting_terrain_type]['name']}"

        diag_status = "ON" if grid_instance.allow_diagonal_movement else "OFF"
        # Algorithm name is now on info panel, not caption. Speed also.
        # Caption can show mode and general help.

        caption_text = f"Mode: {mode_text if mode_text else 'Default (Obstacles)'} | Diag: {diag_status}"
        bindings = " | Keys: S,E,D,R,C | Algos:K,A,L,B,J | Terrain:1-3 | Speed:+/- | F5/F6:Save/Load | Enter:Run"
        pygame.display.set_caption(caption_text + bindings)

    update_gui_caption()

    def start_animation_enhanced(path, visited_fwd, open_fwd, visited_bwd=None, open_bwd=None):
        nonlocal animating, animation_data, current_algorithm, grid_instance
        grid_instance.clear_visualizations() # Clear previous viz flags, but not g/h scores etc.
                                           # Use reset_algorithm_attributes if a full reset is needed before new algo run.

        animation_data["path_nodes"] = path
        animation_data["visited_fwd_list"] = visited_fwd
        animation_data["visited_bwd_list"] = visited_bwd if visited_bwd else []
        animation_data["open_fwd_list"] = open_fwd
        animation_data["open_bwd_list"] = open_bwd if open_bwd else []

        animation_data["current_visited_fwd_step"] = 0
        animation_data["current_visited_bwd_step"] = 0
        animation_data["current_path_step"] = 0

        if current_algorithm == "BIDIRECTIONAL":
            if animation_data["visited_fwd_list"]:
                animation_data["animation_phase"] = "visited_fwd"
            elif animation_data["visited_bwd_list"]:
                animation_data["animation_phase"] = "visited_bwd"
            else:
                animation_data["animation_phase"] = "path"
        else: # For other algorithms
            animation_data["animation_phase"] = "visited_fwd" # Use fwd lists for generic visited/open

        animating = True
        # Caption update is now handled by the main loop's draw_info_panel
        # pygame.display.set_caption(f"Visualizing {ALGO_NAMES[current_algorithm]}...")


    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if not animating:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a: current_algorithm = "A_STAR"
                    elif event.key == pygame.K_l: current_algorithm = "D_STAR_LITE"
                    elif event.key == pygame.K_k: current_algorithm = "DIJKSTRA"
                    elif event.key == pygame.K_b: current_algorithm = "BIDIRECTIONAL"
                    elif event.key == pygame.K_j: current_algorithm = "JPS"

                    if event.key in [pygame.K_a, pygame.K_l, pygame.K_k, pygame.K_b, pygame.K_j]:
                        setting_target_d_lite = False
                        painting_terrain_type = 0
                        setting_start = False
                        setting_end = False

                    elif event.key == pygame.K_d:
                        grid_instance.set_allow_diagonal_movement(not grid_instance.allow_diagonal_movement)
                        print(f"Diagonal movement {'enabled' if grid_instance.allow_diagonal_movement else 'disabled'}")

                    elif event.key == pygame.K_1: painting_terrain_type = 1
                    elif event.key == pygame.K_2: painting_terrain_type = 2
                    elif event.key == pygame.K_3: painting_terrain_type = 3
                    if event.key in [pygame.K_1, pygame.K_2, pygame.K_3]:
                        setting_start, setting_end, setting_target_d_lite = False, False, False
                        print(f"Set to paint terrain: {TERRAIN_COSTS[painting_terrain_type]['name']}")
                    elif event.key == pygame.K_0 or event.key == pygame.K_ESCAPE:
                        painting_terrain_type = 0
                        print("Terrain painting mode OFF")

                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        current_speed_index = max(0, current_speed_index - 1)
                        animation_delay_ms = ANIMATION_SPEED_SETTINGS[current_speed_index]["delay"]
                    elif event.key == pygame.K_MINUS:
                        current_speed_index = min(len(ANIMATION_SPEED_SETTINGS) - 1, current_speed_index + 1)
                        animation_delay_ms = ANIMATION_SPEED_SETTINGS[current_speed_index]["delay"]

                    elif event.key == pygame.K_F5: save_grid_to_file(grid_instance, "grid_config.json")
                    elif event.key == pygame.K_F6:
                        loaded_grid = load_grid_from_file("grid_config.json",
                                                        conf_grid_rows, conf_grid_cols,
                                                        conf_cell_width, conf_cell_height)
                        if loaded_grid:
                            grid_instance = loaded_grid
                            # Recalculate cell dimensions for the potentially new grid topology
                            current_cell_width = conf_window_width // grid_instance.cols
                            current_cell_height = conf_game_window_height // grid_instance.rows

                            setting_start,setting_end,setting_target_d_lite,painting_terrain_type = False,False,False,0
                            d_star_lite_initialized_run = False
                            d_star_lite_pq.clear(); d_star_lite_open_set_tracker.clear()
                            animating = False; grid_instance.reset_algorithm_states() # existing reset is good
                            last_path_length, last_visited_count = None, None
                            print("Grid loaded.")
                        else: print("Failed to load grid.")

                    elif event.key == pygame.K_s: setting_start = True; setting_end,setting_target_d_lite,painting_terrain_type = False,False,0
                    elif event.key == pygame.K_e: setting_end = True; setting_start,setting_target_d_lite,painting_terrain_type = False,False,0
                    elif event.key == pygame.K_t and current_algorithm == "D_STAR_LITE":
                        setting_target_d_lite = True; setting_start,setting_end,painting_terrain_type = False,False,0

                    elif event.key == pygame.K_RETURN:
                        if grid_instance.start_node and grid_instance.end_node:
                            grid_instance.reset_algorithm_states() # Full reset for new search scores
                            grid_instance.update_all_node_neighbors()

                            algo_func = ALGORITHMS[current_algorithm]
                            print(f"Running {ALGO_NAMES[current_algorithm]}...")

                            path_res, visited_fwd_res, open_fwd_res = None, [], []
                            visited_bwd_res, open_bwd_res = None, None

                            if current_algorithm == "D_STAR_LITE":
                                # For D* Lite initial run, reset its persistent state
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
                            else: # A*, Dijkstra, JPS
                                path_res, visited_fwd_res, open_fwd_res = algo_func(grid_instance, grid_instance.start_node, grid_instance.end_node)

                            start_animation_enhanced(path_res, visited_fwd_res, open_fwd_res, visited_bwd_res, open_bwd_res)

                            if path_res:
                                last_path_length = len(path_res)
                                combined_visited_count = len(visited_fwd_res) + (len(visited_bwd_res) if visited_bwd_res else 0)
                                # This count for bidi is not unique nodes yet, just sum of lists.
                                last_visited_count = combined_visited_count
                                print(f"{ALGO_NAMES[current_algorithm]} Path: {last_path_length}, Visited: {last_visited_count}")
                            else:
                                last_path_length = 0
                                combined_visited_count = len(visited_fwd_res) + (len(visited_bwd_res) if visited_bwd_res else 0)
                                last_visited_count = combined_visited_count
                                print(f"{ALGO_NAMES[current_algorithm]} No path. Visited: {last_visited_count}")
                        else:
                            print("Error: Set Start and End nodes.")
                            last_path_length, last_visited_count = None, None

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

                    elif event.key == pygame.K_c:
                        if animating: animating = False; animation_data["animation_phase"] = "stopped"
                        grid_instance.clear_visualizations()
                        last_path_length, last_visited_count = None, None
                        print("Path visualizations cleared.")
                    update_gui_caption()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if mouse_pos[1] >= conf_game_window_height: continue # Click in info panel ignored for grid ops

                    clicked_node = get_gui_node_from_mouse_pos(mouse_pos, grid_instance, current_cell_width, current_cell_height)
                    if clicked_node:
                        if setting_start:
                            if grid_instance.start_node: grid_instance.start_node.terrain_cost = TERRAIN_COSTS[1]["cost"]
                            grid_instance.start_node = clicked_node; clicked_node.is_obstacle = False; clicked_node.terrain_cost = TERRAIN_COSTS[1]["cost"]
                            setting_start = False; d_star_lite_initialized_run = False
                        elif setting_end:
                            old_end = grid_instance.end_node
                            if grid_instance.end_node: grid_instance.end_node.terrain_cost = TERRAIN_COSTS[1]["cost"]
                            grid_instance.end_node = clicked_node; clicked_node.is_obstacle = False; clicked_node.terrain_cost = TERRAIN_COSTS[1]["cost"]
                            setting_end = False
                            if current_algorithm == "D_STAR_LITE" and d_star_lite_initialized_run and grid_instance.start_node:
                                d_star_lite_target_move_update(grid_instance, grid_instance.end_node, old_end, d_star_lite_pq, d_star_lite_open_set_tracker, grid_instance.start_node, heuristic)
                                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)) # Trigger replan
                            else: d_star_lite_initialized_run = False
                        elif setting_target_d_lite and current_algorithm == "D_STAR_LITE":
                            if not clicked_node.is_obstacle and clicked_node != grid_instance.start_node:
                                old_target_node = grid_instance.end_node
                                grid_instance.set_target_node(clicked_node.row, clicked_node.col) # sets grid_instance.end_node
                                if grid_instance.start_node and d_star_lite_initialized_run:
                                    d_star_lite_target_move_update(grid_instance, grid_instance.end_node, old_target_node, d_star_lite_pq, d_star_lite_open_set_tracker, grid_instance.start_node, heuristic)
                                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
                                elif grid_instance.start_node : # Not initialized, run full
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
                        else: # Toggle obstacle
                            if clicked_node != grid_instance.start_node and clicked_node != grid_instance.end_node:
                                clicked_node.is_obstacle = not clicked_node.is_obstacle
                                if clicked_node.is_obstacle: clicked_node.terrain_cost = TERRAIN_COSTS[1]["cost"]
                                if current_algorithm == "D_STAR_LITE" and d_star_lite_initialized_run:
                                    d_star_lite_obstacle_change_update(grid_instance, clicked_node.row, clicked_node.col, d_star_lite_pq, d_star_lite_open_set_tracker, grid_instance.start_node, grid_instance.end_node, heuristic)
                                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
                                else: grid_instance.update_all_node_neighbors()
                        update_gui_caption()

        # Animation Loop
        if animating:
            current_phase = animation_data["animation_phase"]

            if current_phase == "visited_fwd":
                if animation_data["current_visited_fwd_step"] < len(animation_data["visited_fwd_list"]):
                    node = animation_data["visited_fwd_list"][animation_data["current_visited_fwd_step"]]
                    if node != grid_instance.start_node and node != grid_instance.end_node:
                        if current_algorithm == "BIDIRECTIONAL": node.is_in_closed_set_fwd = True
                        else: node.is_visited_by_algorithm = True # Standard visited
                    animation_data["current_visited_fwd_step"] += 1
                    pygame.time.delay(animation_delay_ms)
                else: # Move to next phase
                    if current_algorithm == "BIDIRECTIONAL" and animation_data["visited_bwd_list"]:
                        animation_data["animation_phase"] = "visited_bwd"
                    else:
                        animation_data["animation_phase"] = "set_open" # New intermediate phase

            elif current_phase == "visited_bwd": # Only for Bidirectional
                if animation_data["current_visited_bwd_step"] < len(animation_data["visited_bwd_list"]):
                    node = animation_data["visited_bwd_list"][animation_data["current_visited_bwd_step"]]
                    if node != grid_instance.start_node and node != grid_instance.end_node:
                        node.is_in_closed_set_bwd = True
                    animation_data["current_visited_bwd_step"] += 1
                    pygame.time.delay(animation_delay_ms)
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
                else: # For other algorithms, use combined open_set_nodes
                     for node_obj in animation_data.get("open_set_nodes", []): # Use .get for safety if JPS doesn't populate it yet
                        if not node_obj.is_visited_by_algorithm and not node_obj.is_part_of_path:
                             node_obj.is_in_open_set_for_algorithm = True
                animation_data["animation_phase"] = "path"
                animation_data["current_path_step"] = 0

            elif current_phase == "path":
                if animation_data["current_path_step"] < len(animation_data["path_nodes"]):
                    node = animation_data["path_nodes"][animation_data["current_path_step"]]
                    if node != grid_instance.start_node and node != grid_instance.end_node:
                        node.is_part_of_path = True
                        # Clear other viz flags for path emphasis
                        node.is_visited_by_algorithm = False; node.is_in_open_set_for_algorithm = False
                        node.is_in_closed_set_fwd = False; node.is_in_closed_set_bwd = False
                        node.is_in_open_set_fwd = False; node.is_in_open_set_bwd = False
                    animation_data["current_path_step"] += 1
                    pygame.time.delay(animation_delay_ms)
                else:
                    animation_data["animation_phase"] = "finished"

            elif current_phase == "finished":
                animating = False
                update_gui_caption() # Update caption to show normal bindings again

        screen.fill(tuple(APP_CONFIG["colors"]["background"]))
        draw_all_nodes(screen, grid_instance) # Uses APP_CONFIG internally for colors
        draw_grid_lines(screen, grid_instance.rows, grid_instance.cols,
                        conf_window_width, conf_game_window_height,
                        current_cell_width, current_cell_height)

        # Draw Info Panel
        draw_info_panel(screen, ui_font, ALGO_NAMES[current_algorithm],
                        last_path_length, last_visited_count,
                        ANIMATION_SPEED_SETTINGS[current_speed_index]["name"],
                        conf_game_window_height, conf_window_width, conf_info_panel_height)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main_gui()

[end of main.py]
