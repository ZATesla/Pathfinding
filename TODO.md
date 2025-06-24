# Project TODO List

## General Project Enhancements:
1.  Add detailed usage instructions to `README.md` (how to run, dependencies).
2.  Create `requirements.txt` for easier dependency management (Pygame).
3.  Implement a settings/configuration file (e.g., `config.json`) for grid size, colors, default algorithm, etc.
4.  Add comments and docstrings throughout `main.py` and `core_logic.py` where they are lacking or could be improved.
5.  Refactor `main.py` to further separate GUI components from event handling logic (e.g., into classes).

## Algorithm & Core Logic Improvements:
6.  Implement diagonal movement option for algorithms (with appropriate cost adjustment). - **COMPLETED**
    *   Allows toggling diagonal moves.
    *   Uses sqrt(2) for diagonal costs, 1 for cardinal.
    *   Prevents cutting corners through obstacles.
    *   Heuristics (Octile/Manhattan) adapt to this setting.
7.  Add support for weighted nodes/edges (e.g., different terrain types having different movement costs). - **COMPLETED**
    *   Node class has `terrain_cost` attribute.
    *   `get_move_cost` incorporates `node.terrain_cost`.
    *   GUI allows painting Normal (1.0), Mud (3.0), Water (5.0) terrain costs using keys 1, 2, 3.
    *   Terrain types are visually represented.
    *   Start/End nodes and obstacles reset to default terrain cost.
8.  Implement Bidirectional Search algorithm. - **COMPLETED**
    *   Implemented Dijkstra-based bidirectional search.
    *   Maintains separate open/closed sets for forward and backward searches.
    *   Path reconstruction combines segments from meeting point.
    *   Integrated into GUI with 'B' key.
    *   Basic visualization merges visited/open sets from both searches.
9.  Implement Jump Point Search (JPS) for grid-based speed-up over A*. - **IN PROGRESS (Skeleton)**
    *   Added placeholder functions (`jps_search`, `_jps_jump`, `_jps_identify_successors`) in `core_logic.py`.
    *   Integrated into GUI with 'J' key for selection.
    *   Basic placeholder test added.
    *   **Core JPS logic (pruning, forced neighbors, jump mechanics, cost accumulation) still needs full implementation.**
10. Optimize D* Lite: Implement efficient updates when obstacles change or target moves, instead of full re-computation (current `main.py` does a full replan). - **COMPLETED**
    *   `d_star_lite_obstacle_change_update` function added to `core_logic.py`.
    *   `d_star_lite_target_move_update` function added to `core_logic.py`.
    *   GUI in `main.py` now calls these update functions and re-uses D* Lite's PQ/open_set for replans.
11. Improve D* Lite `_d_star_open_set_tracker` and `_d_star_pq` handling to avoid global-like variables or pass them more cleanly. - **COMPLETED**
    *   D* Lite functions (`initialize`, `update_node`, `compute_shortest_path`) now take `pq` and `open_set_tracker` as parameters.
    *   Removed global-like D* Lite state from `core_logic.py`.
    *   Persistent `pq` and `open_set_tracker` for D* Lite are now managed in `main_gui`.

## GUI & Visualization Enhancements:
12. Add a visual indicator for the current algorithm selected on the GUI itself (not just window caption). - **COMPLETED**
    *   Displays "Algorithm: [Name]" at the top-left of the screen.
    *   Updates dynamically when algorithm is changed.
13. Allow users to save and load grid configurations (obstacles, start/end points). - **COMPLETED**
    *   Saves to/loads from `grid_config.json`.
    *   Includes dimensions, start/end, obstacles, terrain, diagonal settings.
    *   GUI controls: F5 (Save), F6 (Load).
14. Implement adjustable animation speed for algorithm visualization. - **COMPLETED**
    *   User can press `+`/`=` to speed up animation (decrease delay) and `-` to slow down (increase delay).
    *   Cycles through predefined speed levels: Instant, Fast, Normal, Slow, Very Slow.
    *   Current speed setting displayed in window caption.
15. Display path length and number of nodes visited/expanded after an algorithm run. - **COMPLETED**
    *   Shows "Path Length: [value]" and "Nodes Visited: [value]" in the top-right of the screen.
    *   "Path Length: N/A" if no path found. Visited count still shown.
    *   Stats are cleared on grid reset.
16. Add a "Clear Path" button to remove only the visualized path/visited nodes without resetting obstacles/start/end. - **COMPLETED**
    *   Pressing 'C' clears path, visited, and open set highlights.
    *   Stops ongoing animation.
    *   Obstacles, start/end, terrain, and other settings remain.
    *   Displayed stats are cleared.
17. Improve visual distinction between open set, closed set, and path nodes during/after animation. - **COMPLETED**
    *   Defined new colors for standard open/closed sets (Orange/Cyan).
    *   Defined distinct colors for Bidirectional Search:
        *   Forward Open: Gold
        *   Forward Closed: Light Blue
        *   Backward Open: Light Pink
        *   Backward Closed: Light Green
        *   Nodes visited by both: Purple
    *   Updated `Node` class with `is_in_open/closed_set_fwd/bwd` attributes.
    *   Modified `start_animation_enhanced` and `draw_all_nodes` to use these new flags and colors.
    *   Adjusted GUI layout to include an info panel at the bottom for text, moving algorithm name and stats there.

## Testing & Quality:
18. Expand `test_pathfinding.py` to include more complex scenarios and edge cases for all algorithms (especially D* Lite dynamic updates).
19. Add tests for GUI interactions if feasible (might require a GUI testing framework or specific design).
20. Implement basic linting (e.g., Flake8) and code formatting (e.g., Black) checks, possibly as a pre-commit hook or CI step.
