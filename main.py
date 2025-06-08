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
from core_logic import Node, Grid, dijkstra, a_star
# heuristic is used by a_star in core_logic.py, not directly needed in main_gui.py

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

            node_color = WHITE # Default
            if node.is_part_of_path:
                node_color = MAGENTA_PATH
            elif node.is_in_open_set_for_algorithm:
                node_color = ORANGE_OPEN
            elif node.is_visited_by_algorithm:
                node_color = CYAN_VISITED

            # Obstacle, start, and end display properties
            if node.is_obstacle:
                node_color = RED_OBSTACLE
            elif node == grid_instance.start_node: # Check if it's the start_node instance
                node_color = GREEN_START
            elif node == grid_instance.end_node: # Check if it's the end_node instance
                node_color = BLUE_END

            pygame.draw.rect(surface, node_color, rect)


def main_gui(): # Renamed main to main_gui to avoid confusion if we ever import main from somewhere
    pygame.init() # Initialize Pygame modules
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Pathfinding Visualizer - S:Start, E:End, Space:Dijkstra, A:A*, R:Reset")
    clock = pygame.time.Clock()

    # Instantiate the Grid from core_logic, providing cell dimensions for node drawing attributes
    grid_instance = Grid(GRID_ROWS, GRID_COLS, CELL_WIDTH, CELL_HEIGHT)
    # This was the old Grid instantiation from main.py, now replaced by grid_instance from core_logic
    # grid = Grid(GRID_ROWS, GRID_COLS, CELL_WIDTH, CELL_HEIGHT)

    setting_start = False
    setting_end = False

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
        nonlocal animating, animation_data
        grid_instance.reset_algorithm_states() # Use grid_instance from core_logic

        animation_data["visited_nodes_in_order"] = visited_nodes_in_order
        animation_data["path_nodes"] = path
        animation_data["open_set_nodes"] = open_set_nodes # Store for final static display
        animation_data["current_visited_step"] = 0
        animation_data["current_path_step"] = 0
        animation_data["animation_phase"] = "visited"
        animating = True
        pygame.display.set_caption("Visualizing Algorithm...")


    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if not animating: # Only handle new actions if not currently animating
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        setting_start = True
                        setting_end = False
                        pygame.display.set_caption("Setting Start Node... Click on a cell.")
                    elif event.key == pygame.K_e:
                        setting_end = True
                        setting_start = False
                        pygame.display.set_caption("Setting End Node... Click on a cell.")
                    elif event.key == pygame.K_SPACE or event.key == pygame.K_a: # Dijkstra or A*
                        if grid_instance.start_node and grid_instance.end_node: # Use grid_instance
                            grid_instance.reset_algorithm_states()
                            grid_instance.update_all_node_neighbors()

                            path, visited, open_set = [], [], []
                            if event.key == pygame.K_SPACE:
                                print("Running Dijkstra's Algorithm...")
                                path, visited, open_set = dijkstra(grid_instance, grid_instance.start_node, grid_instance.end_node)
                            else: # A*
                                print("Running A* Algorithm...")
                                path, visited, open_set = a_star(grid_instance, grid_instance.start_node, grid_instance.end_node)

                            if path:
                                print("Path found. Length:", len(path)) # Path is list of Node objects
                                start_animation(path, visited, open_set)
                            else:
                                print("No path found.")
                                start_animation([], visited, open_set) # Still show visited nodes
                        else:
                            pygame.display.set_caption("Set Start and End nodes first! S:Start, E:End, Space:Dijkstra, A:A*, R:Reset")

                    elif event.key == pygame.K_r: # Reset grid
                        grid_instance = Grid(GRID_ROWS, GRID_COLS, CELL_WIDTH, CELL_HEIGHT) # New core_logic.Grid
                        setting_start = False
                        setting_end = False
                        animating = False
                        animation_data["animation_phase"] = "stopped"
                        pygame.display.set_caption("Grid Reset. S:Start, E:End, Space:Dijkstra, A:A*, R:Reset")


                if event.type == pygame.MOUSEBUTTONDOWN: # MOUSEBUTTONDOWN
                    if event.button == 1: # Left mouse button
                        mouse_pos = pygame.mouse.get_pos()
                        # Use the new GUI helper for mouse clicks
                        clicked_node = get_gui_node_from_mouse_pos(mouse_pos, grid_instance, CELL_WIDTH, CELL_HEIGHT)

                        if clicked_node:
                            changed_mode = False
                            if setting_start: # Setting start node
                                if clicked_node == grid_instance.end_node: grid_instance.end_node = None # Clear end if same
                                if grid_instance.start_node: grid_instance.start_node.is_obstacle = False # Clear old start visual state
                                grid_instance.start_node = clicked_node
                                grid_instance.start_node.is_obstacle = False # Start node cannot be obstacle
                                print(f"Set start node at ({clicked_node.row}, {clicked_node.col})")
                                setting_start = False
                                changed_mode = True
                            elif setting_end: # Setting end node
                                if clicked_node == grid_instance.start_node: grid_instance.start_node = None # Clear start if same
                                if grid_instance.end_node: grid_instance.end_node.is_obstacle = False # Clear old end visual state
                                grid_instance.end_node = clicked_node
                                grid_instance.end_node.is_obstacle = False # End node cannot be obstacle
                                print(f"Set end node at ({clicked_node.row}, {clicked_node.col})")
                                setting_end = False
                                changed_mode = True
                            else: # Toggle obstacle
                                if clicked_node != grid_instance.start_node and clicked_node != grid_instance.end_node:
                                    clicked_node.is_obstacle = not clicked_node.is_obstacle
                                    grid_instance.update_all_node_neighbors() # Crucial for core logic
                                    print(f"Toggled obstacle at ({clicked_node.row}, {clicked_node.col})")

                            if changed_mode: # If mode (setting_start/end) was changed
                                pygame.display.set_caption("S:Start, E:End, Space:Dijkstra, A:A*, R:Reset")


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
