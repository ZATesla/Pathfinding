import unittest
# Updated import to use core_logic
from core_logic import Node, Grid, dijkstra, a_star, heuristic, run_d_star_lite, reset_d_star_lite_internals

# Constants for grid creation in tests, can be small
TEST_CELL_WIDTH = 10
TEST_CELL_HEIGHT = 10

class TestPathfindingAlgorithms(unittest.TestCase):

    def _convert_path_to_coords(self, path_nodes):
        """Helper to convert a list of Node objects to (row, col) tuples."""
        if not path_nodes:
            return []
        return [(node.row, node.col) for node in path_nodes]

    def setUp(self):
        """Common setup for some tests, though many will define their own grid."""
        # Standard grid size for some tests, can be overridden
        self.rows = 5
        self.cols = 5
        # Grid now takes cell_width/height as optional for tests (nodes will have 0 width/height if not passed)
        # This is fine as tests operate on row/col, not pixel dimensions.
        self.grid = Grid(self.rows, self.cols)

    # --- Dijkstra Tests ---
    def test_dijkstra_simple_path(self):
        # Test with diagonal movement enabled (default for Grid)
        grid_diag = Grid(3, 3)
        grid_diag.start_node = grid_diag.nodes[0][0]
        grid_diag.end_node = grid_diag.nodes[2][2]
        # grid_diag.update_all_node_neighbors() # Called in Grid constructor

        path_nodes_diag, _, _ = dijkstra(grid_diag, grid_diag.start_node, grid_diag.end_node)
        path_coords_diag = self._convert_path_to_coords(path_nodes_diag)
        # Expected path with diagonals: (0,0) -> (1,1) -> (2,2) - Length 3
        self.assertEqual(len(path_coords_diag), 3, f"Diagonal path length incorrect: {len(path_coords_diag)}")
        self.assertEqual(path_coords_diag, [(0,0), (1,1), (2,2)], "Diagonal path incorrect")

        # Test with diagonal movement disabled
        grid_no_diag = Grid(3, 3)
        grid_no_diag.set_allow_diagonal_movement(False)
        grid_no_diag.start_node = grid_no_diag.nodes[0][0]
        grid_no_diag.end_node = grid_no_diag.nodes[2][2]

        path_nodes_no_diag, _, _ = dijkstra(grid_no_diag, grid_no_diag.start_node, grid_no_diag.end_node)
        path_coords_no_diag = self._convert_path_to_coords(path_nodes_no_diag)
        # Expected path without diagonals (Manhattan): Length 5
        self.assertEqual(len(path_coords_no_diag), 5, f"Non-diagonal path length incorrect: {len(path_coords_no_diag)}")
        # One possible non-diagonal path
        possible_paths_no_diag = [
            [(0,0), (1,0), (2,0), (2,1), (2,2)],
            [(0,0), (0,1), (0,2), (1,2), (2,2)],
            [(0,0), (1,0), (1,1), (2,1), (2,2)],
            [(0,0), (0,1), (1,1), (1,2), (2,2)],
            [(0,0), (1,0), (1,1), (1,2), (2,2)], # Path through (1,1) cardinal only
            [(0,0), (0,1), (1,1), (2,1), (2,2)]  # Path through (1,1) cardinal only
        ]
        self.assertIn(path_coords_no_diag, possible_paths_no_diag, f"Non-diagonal path {path_coords_no_diag} not one of expected {possible_paths_no_diag}")


    def test_dijkstra_with_obstacles(self):
        # Diagonal enabled
        grid_diag = Grid(3, 3)
        grid_diag.start_node = grid_diag.nodes[0][0]
        grid_diag.end_node = grid_diag.nodes[2][2]
        grid_diag.nodes[1][1].is_obstacle = True # Obstacle in the middle of direct diagonal path
        grid_diag.update_all_node_neighbors() # Update due to obstacle

        path_nodes_diag, _, _ = dijkstra(grid_diag, grid_diag.start_node, grid_diag.end_node)
        path_coords_diag = self._convert_path_to_coords(path_nodes_diag)
        possible_paths_diag_obs = [
             [(0,0), (1,0), (2,1), (2,2)],
             [(0,0), (0,1), (1,2), (2,2)]
        ]
        self.assertIn(path_coords_diag, possible_paths_diag_obs, f"Diagonal path with obstacle {path_coords_diag} not in {possible_paths_diag_obs}")


        # Diagonal disabled
        grid_no_diag = Grid(3, 3)
        grid_no_diag.set_allow_diagonal_movement(False)
        grid_no_diag.start_node = grid_no_diag.nodes[0][0]
        grid_no_diag.end_node = grid_no_diag.nodes[2][2]
        grid_no_diag.nodes[1][0].is_obstacle = True # Obstacles forcing a specific path
        grid_no_diag.nodes[1][1].is_obstacle = True
        grid_no_diag.update_all_node_neighbors()

        path_nodes_no_diag, _, _ = dijkstra(grid_no_diag, grid_no_diag.start_node, grid_no_diag.end_node)
        path_coords_no_diag = self._convert_path_to_coords(path_nodes_no_diag)
        expected_no_diag_obs = [(0,0), (0,1), (0,2), (1,2), (2,2)]
        self.assertEqual(path_coords_no_diag, expected_no_diag_obs, "Non-diagonal path with obstacles incorrect")


    def test_dijkstra_no_path(self):
        for allow_diag in [True, False]:
            grid = Grid(3, 3)
            grid.set_allow_diagonal_movement(allow_diag)
            grid.start_node = grid.nodes[0][0]
            grid.end_node = grid.nodes[2][2]

            grid.nodes[0][1].is_obstacle = True
            grid.nodes[1][0].is_obstacle = True
            if allow_diag:
                 grid.nodes[1][1].is_obstacle = True
            grid.update_all_node_neighbors()

            path_nodes, _, _ = dijkstra(grid, grid.start_node, grid.end_node)
            path_coords = self._convert_path_to_coords(path_nodes)
            self.assertEqual(path_coords, [], f"No path scenario failed for diagonal={allow_diag}")

    def test_dijkstra_start_is_obstacle(self):
        for allow_diag in [True, False]:
            grid = Grid(3, 3)
            grid.set_allow_diagonal_movement(allow_diag)
            grid.start_node = grid.nodes[0][0]
            grid.end_node = grid.nodes[2][2]
            grid.start_node.is_obstacle = True
            grid.update_all_node_neighbors()

            path_nodes, _, _ = dijkstra(grid, grid.start_node, grid.end_node)
            self.assertEqual(self._convert_path_to_coords(path_nodes), [], f"Start is obstacle failed for diagonal={allow_diag}")

    def test_dijkstra_end_is_obstacle(self):
        for allow_diag in [True, False]:
            grid = Grid(3, 3)
            grid.set_allow_diagonal_movement(allow_diag)
            grid.start_node = grid.nodes[0][0]
            grid.end_node = grid.nodes[2][2]
            grid.end_node.is_obstacle = True
            grid.update_all_node_neighbors()

            path_nodes, _, _ = dijkstra(grid, grid.start_node, grid.end_node)
            self.assertEqual(self._convert_path_to_coords(path_nodes), [], f"End is obstacle failed for diagonal={allow_diag}")

    def test_dijkstra_1x1_grid(self):
        for allow_diag in [True, False]:
            grid = Grid(1, 1)
            grid.set_allow_diagonal_movement(allow_diag)
            grid.start_node = grid.nodes[0][0]
            grid.end_node = grid.nodes[0][0]
            grid.update_all_node_neighbors()

            path_nodes, _, _ = dijkstra(grid, grid.start_node, grid.end_node)
            self.assertEqual(self._convert_path_to_coords(path_nodes), [(0,0)], f"1x1 grid failed for diagonal={allow_diag}")

    def test_dijkstra_start_equals_end(self):
        for allow_diag in [True, False]:
            grid = Grid(3, 3)
            grid.set_allow_diagonal_movement(allow_diag)
            grid.start_node = grid.nodes[1][1]
            grid.end_node = grid.nodes[1][1]
            grid.update_all_node_neighbors()

            path_nodes, _, _ = dijkstra(grid, grid.start_node, grid.end_node)
            self.assertEqual(self._convert_path_to_coords(path_nodes), [(1,1)], f"Start equals end failed for diagonal={allow_diag}")

    # --- A* Tests ---
    def test_a_star_simple_path(self):
        # Diagonal enabled
        grid_diag = Grid(3, 3)
        grid_diag.start_node = grid_diag.nodes[0][0]
        grid_diag.end_node = grid_diag.nodes[2][2]

        path_nodes_diag, _, _ = a_star(grid_diag, grid_diag.start_node, grid_diag.end_node)
        path_coords_diag = self._convert_path_to_coords(path_nodes_diag)
        expected_diag = [(0,0), (1,1), (2,2)]
        self.assertEqual(path_coords_diag, expected_diag, "A* Diagonal simple path incorrect")

        # Diagonal disabled
        grid_no_diag = Grid(3, 3)
        grid_no_diag.set_allow_diagonal_movement(False)
        grid_no_diag.start_node = grid_no_diag.nodes[0][0]
        grid_no_diag.end_node = grid_no_diag.nodes[2][2]

        path_nodes_no_diag, _, _ = a_star(grid_no_diag, grid_no_diag.start_node, grid_no_diag.end_node)
        path_coords_no_diag = self._convert_path_to_coords(path_nodes_no_diag)
        possible_paths_no_diag = [
            [(0,0), (1,0), (2,0), (2,1), (2,2)],
            [(0,0), (0,1), (0,2), (1,2), (2,2)],
            [(0,0), (1,0), (1,1), (2,1), (2,2)],
            [(0,0), (0,1), (1,1), (1,2), (2,2)],
            [(0,0), (1,0), (1,1), (1,2), (2,2)],
            [(0,0), (0,1), (1,1), (2,1), (2,2)]
        ]
        self.assertEqual(len(path_coords_no_diag), 5, "A* Non-diagonal path length incorrect")
        self.assertIn(path_coords_no_diag, possible_paths_no_diag, f"A* Non-diagonal path {path_coords_no_diag} not one of expected {possible_paths_no_diag}")


    def test_a_star_with_obstacles(self):
        # Diagonal enabled
        grid_diag = Grid(3, 3)
        grid_diag.start_node = grid_diag.nodes[0][0]
        grid_diag.end_node = grid_diag.nodes[2][2]
        grid_diag.nodes[1][1].is_obstacle = True
        grid_diag.update_all_node_neighbors()

        path_nodes_diag, _, _ = a_star(grid_diag, grid_diag.start_node, grid_diag.end_node)
        path_coords_diag = self._convert_path_to_coords(path_nodes_diag)
        expected_paths_diag_obs = [
            [(0,0), (1,0), (2,1), (2,2)],
            [(0,0), (0,1), (1,2), (2,2)],
        ]
        self.assertIn(path_coords_diag, expected_paths_diag_obs, f"A* Diagonal path with obstacle {path_coords_diag} not in {expected_paths_diag_obs}")


        # Diagonal disabled
        grid_no_diag = Grid(3, 3)
        grid_no_diag.set_allow_diagonal_movement(False)
        grid_no_diag.start_node = grid_no_diag.nodes[0][0]
        grid_no_diag.end_node = grid_no_diag.nodes[2][2]
        grid_no_diag.nodes[1][0].is_obstacle = True
        grid_no_diag.nodes[1][1].is_obstacle = True
        grid_no_diag.update_all_node_neighbors()

        path_nodes_no_diag, _, _ = a_star(grid_no_diag, grid_no_diag.start_node, grid_no_diag.end_node)
        path_coords_no_diag = self._convert_path_to_coords(path_nodes_no_diag)
        expected_no_diag_obs = [(0,0), (0,1), (0,2), (1,2), (2,2)]
        self.assertEqual(path_coords_no_diag, expected_no_diag_obs, "A* Non-diagonal path with obstacles incorrect")


    def test_a_star_no_path(self):
        for allow_diag in [True, False]:
            grid = Grid(3, 3)
            grid.set_allow_diagonal_movement(allow_diag)
            grid.start_node = grid.nodes[0][0]
            grid.end_node = grid.nodes[2][2]

            grid.nodes[0][1].is_obstacle = True
            grid.nodes[1][0].is_obstacle = True
            if allow_diag:
                 grid.nodes[1][1].is_obstacle = True
            grid.update_all_node_neighbors()

            path_nodes, _, _ = a_star(grid, grid.start_node, grid.end_node)
            self.assertEqual(self._convert_path_to_coords(path_nodes), [], f"A* No path failed for diagonal={allow_diag}")

    def test_a_star_start_is_obstacle(self):
        for allow_diag in [True, False]:
            grid = Grid(3, 3)
            grid.set_allow_diagonal_movement(allow_diag)
            grid.start_node = grid.nodes[0][0]
            grid.end_node = grid.nodes[2][2]
            grid.start_node.is_obstacle = True
            grid.update_all_node_neighbors()

            path_nodes, _, _ = a_star(grid, grid.start_node, grid.end_node)
            self.assertEqual(self._convert_path_to_coords(path_nodes), [], f"A* Start is obstacle failed for diagonal={allow_diag}")

    def test_a_star_end_is_obstacle(self):
        for allow_diag in [True, False]:
            grid = Grid(3, 3)
            grid.set_allow_diagonal_movement(allow_diag)
            grid.start_node = grid.nodes[0][0]
            grid.end_node = grid.nodes[2][2]
            grid.end_node.is_obstacle = True
            grid.update_all_node_neighbors()

            path_nodes, _, _ = a_star(grid, grid.start_node, grid.end_node)
            self.assertEqual(self._convert_path_to_coords(path_nodes), [], f"A* End is obstacle failed for diagonal={allow_diag}")

    def test_a_star_1x1_grid(self):
        for allow_diag in [True, False]:
            grid = Grid(1, 1)
            grid.set_allow_diagonal_movement(allow_diag)
            grid.start_node = grid.nodes[0][0]
            grid.end_node = grid.nodes[0][0]
            grid.update_all_node_neighbors()

            path_nodes, _, _ = a_star(grid, grid.start_node, grid.end_node)
            self.assertEqual(self._convert_path_to_coords(path_nodes), [(0,0)], f"A* 1x1 grid failed for diagonal={allow_diag}")

    def test_a_star_start_equals_end(self):
        for allow_diag in [True, False]:
            grid = Grid(3, 3)
            grid.set_allow_diagonal_movement(allow_diag)
            grid.start_node = grid.nodes[1][1]
            grid.end_node = grid.nodes[1][1]
            grid.update_all_node_neighbors()

            path_nodes, _, _ = a_star(grid, grid.start_node, grid.end_node)
            self.assertEqual(self._convert_path_to_coords(path_nodes), [(1,1)], f"A* Start equals end failed for diagonal={allow_diag}")


    def test_heuristic_manhattan_distance(self):
        node1 = Node(0, 0, 3, 3)
        node2 = Node(2, 2, 3, 3)
        self.assertAlmostEqual(heuristic(node1, node2, allow_diagonal=False), 4.0)

        node3 = Node(0, 0, 5, 5)
        node4 = Node(4, 0, 5, 5)
        self.assertAlmostEqual(heuristic(node3, node4, allow_diagonal=False), 4.0)

    def test_heuristic_octile_distance(self):
        node1 = Node(0, 0, 3, 3)
        node2 = Node(2, 2, 3, 3)
        self.assertAlmostEqual(heuristic(node1, node2, allow_diagonal=True), 2 * 1.41421356, places=5)

        node3 = Node(0, 0, 5, 5)
        node4 = Node(4, 1, 5, 5)
        self.assertAlmostEqual(heuristic(node3, node4, allow_diagonal=True), 1.41421356 * 1 + 1 * (4-1), places=5)


    def test_grid_instantiation_only(self):
        grid = Grid(1, 1)
        self.assertIsNotNone(grid)
        node = grid.nodes[0][0]
        self.assertIsNotNone(node)

    # --- D* Lite Tests ---
    def test_d_star_lite_simple_path(self):
        reset_d_star_lite_internals()
        # Diagonal enabled
        grid_diag = Grid(3, 3)
        grid_diag.start_node = grid_diag.nodes[0][0]
        grid_diag.set_target_node(2, 2)

        path_nodes_diag, _, _ = run_d_star_lite(grid_diag, grid_diag.start_node, grid_diag.end_node, heuristic)
        path_coords_diag = self._convert_path_to_coords(path_nodes_diag)
        self.assertEqual(path_coords_diag, [(0,0), (1,1), (2,2)], "D* Lite simple diagonal path incorrect")

        reset_d_star_lite_internals()
        # Diagonal disabled
        grid_no_diag = Grid(3, 3)
        grid_no_diag.set_allow_diagonal_movement(False)
        grid_no_diag.start_node = grid_no_diag.nodes[0][0]
        grid_no_diag.set_target_node(2, 2)

        path_nodes_no_diag, _, _ = run_d_star_lite(grid_no_diag, grid_no_diag.start_node, grid_no_diag.end_node, heuristic)
        path_coords_no_diag = self._convert_path_to_coords(path_nodes_no_diag)
        self.assertEqual(len(path_coords_no_diag), 5, "D* Lite simple non-diagonal path length incorrect")
        self.assertEqual(path_coords_no_diag[0], (0,0))
        self.assertEqual(path_coords_no_diag[-1], (2,2))


    def test_d_star_lite_with_obstacles(self):
        reset_d_star_lite_internals()
        # Diagonal enabled
        grid_diag = Grid(3, 3)
        grid_diag.start_node = grid_diag.nodes[0][0]
        grid_diag.set_target_node(2, 2)
        grid_diag.nodes[1][1].is_obstacle = True
        grid_diag.update_all_node_neighbors()

        path_nodes_diag, _, _ = run_d_star_lite(grid_diag, grid_diag.start_node, grid_diag.end_node, heuristic)
        path_coords_diag = self._convert_path_to_coords(path_nodes_diag)
        expected_paths_diag_obs = [
            [(0,0), (1,0), (2,1), (2,2)],
            [(0,0), (0,1), (1,2), (2,2)],
        ]
        self.assertIn(path_coords_diag, expected_paths_diag_obs, f"D* Lite diagonal path with obstacle {path_coords_diag} not in {expected_paths_diag_obs}")

        reset_d_star_lite_internals()
        # Diagonal disabled
        grid_no_diag = Grid(3, 3)
        grid_no_diag.set_allow_diagonal_movement(False)
        grid_no_diag.start_node = grid_no_diag.nodes[0][0]
        grid_no_diag.set_target_node(2, 2)
        grid_no_diag.nodes[1][0].is_obstacle = True
        grid_no_diag.nodes[1][1].is_obstacle = True
        grid_no_diag.update_all_node_neighbors()

        path_nodes_no_diag, _, _ = run_d_star_lite(grid_no_diag, grid_no_diag.start_node, grid_no_diag.end_node, heuristic)
        path_coords_no_diag = self._convert_path_to_coords(path_nodes_no_diag)
        expected_no_diag_obs = [(0,0), (0,1), (0,2), (1,2), (2,2)]
        self.assertEqual(path_coords_no_diag, expected_no_diag_obs, "D* Lite non-diagonal path with obstacles incorrect")


    def test_d_star_lite_no_path(self):
        for allow_diag in [True, False]:
            reset_d_star_lite_internals()
            grid = Grid(3, 3)
            grid.set_allow_diagonal_movement(allow_diag)
            grid.start_node = grid.nodes[0][0]
            grid.set_target_node(2, 2)

            grid.nodes[0][1].is_obstacle = True
            grid.nodes[1][0].is_obstacle = True
            if allow_diag:
                 grid.nodes[1][1].is_obstacle = True
            grid.update_all_node_neighbors()

            path_nodes, _, _ = run_d_star_lite(grid, grid.start_node, grid.end_node, heuristic)
            self.assertEqual(self._convert_path_to_coords(path_nodes), [], f"D* Lite no path scenario failed for diagonal={allow_diag}")

    def test_d_star_lite_start_is_obstacle(self):
        for allow_diag in [True, False]:
            reset_d_star_lite_internals()
            grid = Grid(3, 3)
            grid.set_allow_diagonal_movement(allow_diag)
            start_node_obj = grid.nodes[0][0]
            grid.start_node = start_node_obj
            grid.set_target_node(2, 2)
            start_node_obj.is_obstacle = True
            grid.update_all_node_neighbors()

            path_nodes, _, _ = run_d_star_lite(grid, grid.start_node, grid.end_node, heuristic)
            self.assertEqual(self._convert_path_to_coords(path_nodes), [], f"D* Lite start is obstacle scenario failed for diagonal={allow_diag}")

    def test_d_star_lite_target_is_obstacle(self):
        for allow_diag in [True, False]:
            reset_d_star_lite_internals()
            grid = Grid(3, 3)
            grid.set_allow_diagonal_movement(allow_diag)
            grid.start_node = grid.nodes[0][0] # This line was the reported error
            target_node_obj = grid.nodes[2][2]
            grid.set_target_node(2,2)
            target_node_obj.is_obstacle = True
            grid.update_all_node_neighbors()

            path_nodes, _, _ = run_d_star_lite(grid, grid.start_node, grid.end_node, heuristic)
            self.assertEqual(self._convert_path_to_coords(path_nodes), [], f"D* Lite target is obstacle scenario failed for diagonal={allow_diag}")

    def test_d_star_lite_target_move_simple(self):
        # Test with diagonal enabled first
        reset_d_star_lite_internals()
        grid_diag = Grid(3, 3)
        grid_diag.start_node = grid_diag.nodes[0][0]

        grid_diag.set_target_node(2, 2)
        path_nodes1_diag, _, _ = run_d_star_lite(grid_diag, grid_diag.start_node, grid_diag.end_node, heuristic)
        self.assertEqual(self._convert_path_to_coords(path_nodes1_diag), [(0,0),(1,1),(2,2)], "D* Lite (diag) initial path incorrect")

        reset_d_star_lite_internals()
        grid_diag.set_target_node(0, 2)
        path_nodes2_diag, _, _ = run_d_star_lite(grid_diag, grid_diag.start_node, grid_diag.end_node, heuristic)
        self.assertEqual(self._convert_path_to_coords(path_nodes2_diag), [(0,0),(0,1),(0,2)], "D* Lite (diag) moved target path incorrect")

        # Test with diagonal disabled
        reset_d_star_lite_internals()
        grid_no_diag = Grid(3, 3)
        grid_no_diag.set_allow_diagonal_movement(False)
        grid_no_diag.start_node = grid_no_diag.nodes[0][0]

        grid_no_diag.set_target_node(2, 2)
        path_nodes1_no_diag, _, _ = run_d_star_lite(grid_no_diag, grid_no_diag.start_node, grid_no_diag.end_node, heuristic)
        self.assertEqual(len(self._convert_path_to_coords(path_nodes1_no_diag)), 5, "D* Lite (no-diag) initial path length incorrect")

        reset_d_star_lite_internals()
        grid_no_diag.set_target_node(0, 2)
        path_nodes2_no_diag, _, _ = run_d_star_lite(grid_no_diag, grid_no_diag.start_node, grid_no_diag.end_node, heuristic)
        self.assertEqual(self._convert_path_to_coords(path_nodes2_no_diag), [(0,0),(0,1),(0,2)], "D* Lite (no-diag) moved target path incorrect")


    def test_d_star_lite_target_move_with_obstacles(self):
        # Diagonal enabled
        reset_d_star_lite_internals()
        grid_diag = Grid(4, 4)
        grid_diag.start_node = grid_diag.nodes[0][0]
        grid_diag.nodes[1][0].is_obstacle = True
        grid_diag.nodes[1][1].is_obstacle = True
        grid_diag.nodes[1][2].is_obstacle = True
        grid_diag.update_all_node_neighbors()

        grid_diag.set_target_node(0, 3)
        path_nodes1_diag, _, _ = run_d_star_lite(grid_diag, grid_diag.start_node, grid_diag.end_node, heuristic)
        self.assertEqual(self._convert_path_to_coords(path_nodes1_diag), [(0,0),(0,1),(0,2),(0,3)], "D* Lite (diag) initial path with obs incorrect")

        reset_d_star_lite_internals()
        grid_diag.set_target_node(3,3)
        path_nodes2_diag, _, _ = run_d_star_lite(grid_diag, grid_diag.start_node, grid_diag.end_node, heuristic)
        expected_diag_obs_moved = [(0,0),(0,1),(0,2),(1,3),(2,3),(3,3)] # Corrected expected path
        self.assertEqual(self._convert_path_to_coords(path_nodes2_diag), expected_diag_obs_moved, "D* Lite (diag) moved target path with obs incorrect")

        # Diagonal disabled
        reset_d_star_lite_internals()
        grid_no_diag = Grid(4, 4)
        grid_no_diag.set_allow_diagonal_movement(False)
        grid_no_diag.start_node = grid_no_diag.nodes[0][0]
        grid_no_diag.nodes[1][0].is_obstacle = True
        grid_no_diag.nodes[1][1].is_obstacle = True
        grid_no_diag.nodes[1][2].is_obstacle = True
        grid_no_diag.update_all_node_neighbors()

        grid_no_diag.set_target_node(0, 3)
        path_nodes1_no_diag, _, _ = run_d_star_lite(grid_no_diag, grid_no_diag.start_node, grid_no_diag.end_node, heuristic)
        self.assertEqual(self._convert_path_to_coords(path_nodes1_no_diag), [(0,0),(0,1),(0,2),(0,3)], "D* Lite (no-diag) initial path with obs incorrect")

        reset_d_star_lite_internals()
        grid_no_diag.set_target_node(3,3)
        path_nodes2_no_diag, _, _ = run_d_star_lite(grid_no_diag, grid_no_diag.start_node, grid_no_diag.end_node, heuristic)
        expected_no_diag_obs_moved = [(0,0),(0,1),(0,2),(0,3),(1,3),(2,3),(3,3)]
        self.assertEqual(self._convert_path_to_coords(path_nodes2_no_diag), expected_no_diag_obs_moved, "D* Lite (no-diag) moved target path with obs incorrect")


if __name__ == '__main__':
    unittest.main()
