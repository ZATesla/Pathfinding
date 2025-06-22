# Pathfinding Visualizer

## Project Overview
This project is a Python application that demonstrates various pathfinding algorithms on a grid-based GUI. Users can visualize how algorithms like Dijkstra's and A* find the shortest path between two points, and can add obstacles to create more complex scenarios.

## Features
- Interactive n x n grid GUI.
- Visualization of pathfinding algorithms in real-time.
- Ability to add and remove obstacles on the grid.
- Support for multiple pathfinding algorithms (Dijkstra, A*, D* Lite).
- Toggleable diagonal movement for all algorithms.
- Clear display of the starting point, ending point, obstacles, visited nodes, and the final path.

## Algorithms Implemented
The following pathfinding algorithms are implemented:
- Dijkstra's Algorithm
- A* (A-Star) Algorithm
- D* Lite Algorithm

## Key Controls
- **Mouse Left Click**:
    - Set Start Node (if 'S' mode is active).
    - Set End Node (if 'E' mode is active).
    - Set D* Lite Target Node (if 'T' mode is active for D* Lite).
    - Toggle Obstacles (default mode).
- **S**: Enter "Set Start Node" mode.
- **E**: Enter "Set End Node" mode.
- **D**: Toggle diagonal movement ON/OFF.
- **Enter**: Run the selected pathfinding algorithm.
- **R**: Reset the grid, start/end nodes, and obstacles.
- **K**: Select Dijkstra's Algorithm.
- **A**: Select A* Algorithm.
- **L**: Select D* Lite Algorithm.
- **T**: (Only when D* Lite is selected) Enter "Set D* Lite Target Node" mode. This allows moving the target and observing D* Lite replan.

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