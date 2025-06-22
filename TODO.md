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
10. Optimize D* Lite: Implement efficient updates when obstacles change or target moves, instead of full re-computation (current `main.py` does a full replan).
11. Improve D* Lite `_d_star_open_set_tracker` and `_d_star_pq` handling to avoid global-like variables or pass them more cleanly.

## GUI & Visualization Enhancements:
12. Add a visual indicator for the current algorithm selected on the GUI itself (not just window caption).
13. Allow users to save and load grid configurations (obstacles, start/end points).
14. Implement adjustable animation speed for algorithm visualization.
15. Display path length and number of nodes visited/expanded after an algorithm run.
16. Add a "Clear Path" button to remove only the visualized path/visited nodes without resetting obstacles/start/end.
17. Improve visual distinction between open set, closed set, and path nodes during/after animation.

## Testing & Quality:
18. Expand `test_pathfinding.py` to include more complex scenarios and edge cases for all algorithms (especially D* Lite dynamic updates).
19. Add tests for GUI interactions if feasible (might require a GUI testing framework or specific design).
20. Implement basic linting (e.g., Flake8) and code formatting (e.g., Black) checks, possibly as a pre-commit hook or CI step.
