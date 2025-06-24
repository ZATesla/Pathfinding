# Pathfinding Visualizer

## Project Overview
This project is a Python application that demonstrates various pathfinding algorithms on a grid-based GUI. Users can visualize how algorithms like Dijkstra's and A* find the shortest path between two points, and can add obstacles to create more complex scenarios.

## Features
- Interactive n x n grid GUI.
- Visualization of pathfinding algorithms in real-time.
- Ability to add and remove obstacles on the grid.
- Support for multiple pathfinding algorithms (Dijkstra, A*, D* Lite).
- Toggleable diagonal movement for all algorithms.
- Ability to "paint" different terrain types (Normal, Mud, Water) with varying movement costs.
- Save and Load grid configurations (obstacles, start/end points, terrain, diagonal settings) to/from `grid_config.json`.
- Adjustable animation speed for algorithm visualization.
- Displays path length and number of nodes visited/expanded after an algorithm run.
- Ability to clear path visualizations (path, visited, open set) without resetting the grid.
- Enhanced visualization with distinct colors for Bidirectional Search frontiers (forward/backward open and closed sets) and meeting points.
- Clear display of the starting point, ending point, obstacles, terrain types, visited nodes, and the final path.

## Algorithms Implemented
- Dijkstra's Algorithm
- A* (A-Star) Algorithm
- D* Lite Algorithm (with efficient replanning for obstacle/target changes)
- Bidirectional Search (Dijkstra-based)
- Jump Point Search (JPS) (Initial placeholder)

## Key Controls
- **Mouse Left Click**:
    - Set Start Node (if 'S' mode is active).
    - Set End Node (if 'E' mode is active).
    - Set D* Lite Target Node (if 'T' mode is active for D* Lite).
    - Toggle Obstacles (default mode).
    - Paint Terrain (if terrain paint mode is active: 1, 2, or 3).
- **S**: Enter "Set Start Node" mode.
- **E**: Enter "Set End Node" mode.
- **D**: Toggle diagonal movement ON/OFF.
- **Enter**: Run the selected pathfinding algorithm.
- **R**: Reset the grid (start/end, obstacles, terrain).
- **K**: Select Dijkstra's Algorithm.
- **A**: Select A* Algorithm.
- **L**: Select D* Lite Algorithm.
- **B**: Select Bidirectional Search.
- **J**: Select Jump Point Search.
- **C**: Clear current path visualization (path, visited, open set nodes).
- **T**: (Only when D* Lite is selected) Enter "Set D* Lite Target Node" mode.
- **F5**: Save current grid configuration to `grid_config.json`.
- **F6**: Load grid configuration from `grid_config.json`.
- **1**: Activate "Paint Normal Terrain" mode (cost 1.0).
- **2**: Activate "Paint Mud Terrain" mode (cost 3.0).
- **3**: Activate "Paint Water Terrain" mode (cost 5.0).
- **0** or **ESC**: Deactivate terrain painting mode (return to obstacle toggling).
- **+**/**=**: Increase animation speed (decrease delay).
- **-**: Decrease animation speed (increase delay).

## Usage

### Prerequisites
- Python 3.x installed on your system.
- Pygame library. You can install it using pip:
  ```bash
  pip install pygame
  ```

### Running the Application
1.  Navigate to the project directory in your terminal.
2.  Run the application using:
    ```bash
    python main.py
    ```
3.  A window titled "Pathfinding Visualizer" will open, displaying a grid. This is where you'll interact with the pathfinding algorithms. The window caption also provides a quick reference to common key bindings and the current mode.

### Getting Started with the Visualizer
Here's a typical workflow to visualize your first path:

1.  **Set the Start Node**:
    - Press the `S` key. The window caption will indicate you are in "Set Start" mode.
    - Click on any cell in the grid. This cell will turn green, marking it as the starting point for the pathfinding algorithm.
2.  **Set the End Node**:
    - Press the `E` key. The caption will switch to "Set End" mode.
    - Click on a different cell. This cell will turn blue, marking it as the destination.
3.  **Add Obstacles (Optional)**:
    - By default (or after pressing `ESC` or `0` to exit other modes), clicking on any empty grid cell will toggle it as an obstacle. Obstacles are shown in red and cannot be traversed by the algorithms. Click again to remove an obstacle.
4.  **Choose an Algorithm**:
    - You can select different pathfinding algorithms using their respective keys:
        - `K` for Dijkstra's Algorithm
        - `A` for A* (A-Star)
        - `L` for D* Lite
        - `B` for Bidirectional Search
        - `J` for Jump Point Search (currently a placeholder)
    - The currently selected algorithm is displayed in the info panel at the bottom of the screen.
5.  **Run the Visualization**:
    - Press the `Enter` key.
    - The selected algorithm will start searching for a path from the start node to the end node.
    - You will see nodes being "visited" (typically colored cyan/light blue or other colors for bidirectional search) and nodes in the "open set" (orange/gold/pink) as the algorithm explores the grid.
    - Once the path is found, it will be highlighted (usually in magenta). If no path exists, this will be indicated.
    - Statistics like path length and nodes visited will appear in the info panel.

### Other Interactions
- **Terrain**: Press `1`, `2`, or `3` to paint different terrain types (Normal, Mud, Water) with varying movement costs. Click on cells to apply the selected terrain. Press `0` or `ESC` to return to obstacle mode.
- **Diagonal Movement**: Press `D` to toggle diagonal movements for the algorithms. The current status (ON/OFF) is shown in the window caption.
- **Animation Speed**: Use `+`/`=` to speed up the animation and `-` to slow it down. The current speed setting is displayed.
- **Saving/Loading**: Press `F5` to save your current grid setup (start/end points, obstacles, terrain) to `grid_config.json`. Press `F6` to load a previously saved configuration.
- **Clearing**: Press `C` to clear the current path, visited nodes, and open set highlights without resetting your obstacles or start/end points. Press `R` to reset the entire grid to its default state.

For a full list of controls and their functions, refer to the **Key Controls** section below.

## Key Controls
(This section remains as is)
... (rest of the Key Controls section) ...

## Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes, ensuring code is well-commented and tested where applicable.
4. Submit a pull request for review.

## License
This project is licensed under the terms of the LICENSE file.