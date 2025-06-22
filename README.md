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
- Clear display of the starting point, ending point, obstacles, terrain types, visited nodes, and the final path.

## Algorithms Implemented
The following pathfinding algorithms are implemented:
- Dijkstra's Algorithm
- A* (A-Star) Algorithm
- D* Lite Algorithm
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
- **T**: (Only when D* Lite is selected) Enter "Set D* Lite Target Node" mode.
- **1**: Activate "Paint Normal Terrain" mode (cost 1.0).
- **2**: Activate "Paint Mud Terrain" mode (cost 3.0).
- **3**: Activate "Paint Water Terrain" mode (cost 5.0).
- **0** or **ESC**: Deactivate terrain painting mode (return to obstacle toggling).

## Usage
1.  Ensure you have Python and Pygame installed.
    ```bash
    pip install pygame
    ```
2.  Run the application:
    ```bash
    python main.py
    ```
3.  Use the key controls listed above to interact with the visualizer.

## Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes, ensuring code is well-commented and tested where applicable.
4. Submit a pull request for review.

## License
This project is licensed under the terms of the LICENSE file.