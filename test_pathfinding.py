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
        grid = Grid(3, 3) # TEST_CELL_WIDTH, TEST_CELL_HEIGHT removed as they are optional
        grid.start_node = grid.nodes[0][0]
        grid.end_node = grid.nodes[2][2]
        grid.update_all_node_neighbors()

        path_nodes, _, _ = dijkstra(grid, grid.start_node, grid.end_node)
        path_coords = self._convert_path_to_coords(path_nodes)

        # Expected path can be one of many valid ones, let's pick one
        # e.g., (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2) OR (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) etc.
        # A known shortest path for Dijkstra on an open grid:
        expected_path_variant1 = [(0,0), (1,0), (2,0), (2,1), (2,2)] # Length 5, 4 steps
        expected_path_variant2 = [(0,0), (0,1), (0,2), (1,2), (2,2)] # Length 5, 4 steps
        expected_path_variant3 = [(0,0), (1,0), (1,1), (1,2), (2,2)]
        expected_path_variant4 = [(0,0), (0,1), (1,1), (2,1), (2,2)]

        self.assertTrue(len(path_coords) == 5, f"Path length incorrect: {len(path_coords)}")
        # Check if it's one of the valid shortest paths, as Dijkstra might return different ones
        # For a 3x3 grid from (0,0) to (2,2), any path of length 5 is valid if no obstacles.
        # Example: (0,0)->(1,0)->(1,1)->(2,1)->(2,2)
        # (0,0)->(0,1)->(1,1)->(1,2)->(2,2)
        # (0,0)->(1,1) is not allowed (diagonal)
        # This test is a bit loose, a more specific small grid might be better.
        # For (0,0) to (2,2) without obstacles, path length must be 5 (4 steps).
        self.assertEqual(path_coords[0], (0,0))
        self.assertEqual(path_coords[-1], (2,2))


    def test_dijkstra_with_obstacles(self):
        grid = Grid(3, 3)
        grid.start_node = grid.nodes[0][0]
        grid.end_node = grid.nodes[2][2]

        # Add obstacle
        grid.nodes[1][0].is_obstacle = True
        grid.nodes[1][1].is_obstacle = True
        # grid.nodes[0][1].is_obstacle = True # Alternative obstacle
        grid.update_all_node_neighbors()

        path_nodes, _, _ = dijkstra(grid, grid.start_node, grid.end_node)
        path_coords = self._convert_path_to_coords(path_nodes)

        # Expected path: (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2)
        expected_path = [(0,0), (0,1), (0,2), (1,2), (2,2)]
        self.assertEqual(path_coords, expected_path)

    def test_dijkstra_no_path(self):
        grid = Grid(3, 3)
        grid.start_node = grid.nodes[0][0]
        grid.end_node = grid.nodes[2][2]

        # Wall off the start node
        grid.nodes[0][1].is_obstacle = True
        grid.nodes[1][0].is_obstacle = True
        grid.nodes[1][1].is_obstacle = True # if start is (0,0)
        grid.update_all_node_neighbors()

        path_nodes, _, _ = dijkstra(grid, grid.start_node, grid.end_node)
        path_coords = self._convert_path_to_coords(path_nodes)
        self.assertEqual(path_coords, [])

    def test_dijkstra_start_is_obstacle(self):
        grid = Grid(3, 3)
        grid.start_node = grid.nodes[0][0]
        grid.end_node = grid.nodes[2][2]
        grid.start_node.is_obstacle = True # Set start as obstacle
        grid.update_all_node_neighbors()

        path_nodes, _, _ = dijkstra(grid, grid.start_node, grid.end_node)
        self.assertEqual(self._convert_path_to_coords(path_nodes), [])

    def test_dijkstra_end_is_obstacle(self):
        grid = Grid(3, 3)
        grid.start_node = grid.nodes[0][0]
        grid.end_node = grid.nodes[2][2]
        grid.end_node.is_obstacle = True # Set end as obstacle
        grid.update_all_node_neighbors()

        path_nodes, _, _ = dijkstra(grid, grid.start_node, grid.end_node)
        self.assertEqual(self._convert_path_to_coords(path_nodes), [])

    def test_dijkstra_1x1_grid(self):
        grid = Grid(1, 1)
        grid.start_node = grid.nodes[0][0]
        grid.end_node = grid.nodes[0][0]
        grid.update_all_node_neighbors()

        path_nodes, _, _ = dijkstra(grid, grid.start_node, grid.end_node)
        self.assertEqual(self._convert_path_to_coords(path_nodes), [(0,0)])

    def test_dijkstra_start_equals_end(self):
        grid = Grid(3, 3)
        grid.start_node = grid.nodes[1][1]
        grid.end_node = grid.nodes[1][1]
        grid.update_all_node_neighbors()

        path_nodes, _, _ = dijkstra(grid, grid.start_node, grid.end_node)
        self.assertEqual(self._convert_path_to_coords(path_nodes), [(1,1)])

    # --- A* Tests ---
    def test_a_star_simple_path(self):
        grid = Grid(3, 3)
        grid.start_node = grid.nodes[0][0]
        grid.end_node = grid.nodes[2][2]
        grid.update_all_node_neighbors()

        path_nodes, _, _ = a_star(grid, grid.start_node, grid.end_node)
        path_coords = self._convert_path_to_coords(path_nodes)

        # A* should also find a path of length 5. It might be more deterministic due to heuristic.
        # One common A* path: (0,0)->(1,0)->(2,0)->(2,1)->(2,2) or (0,0)->(0,1)->(1,1)->(2,1)->(2,2) etc.
        # For (0,0) to (2,2), Manhattan distance to end from (1,0) is 3. From (0,1) is 3.
        # (0,0) g=0, h=4, f=4
        # Neighbors: (1,0) g=1, h=3, f=4.  (0,1) g=1, h=3, f=4.  A* might pick one based on tie-breaking (heap order).
        # Let's assume it explores (1,0) first if tie broken that way.
        # (1,0) -> (2,0) g=2, h=2, f=4.  (1,0) -> (1,1) g=2, h=2, f=4
        # This test needs to accept any valid shortest path of length 5.
        self.assertEqual(len(path_coords), 5)
        self.assertEqual(path_coords[0], (0,0))
        self.assertEqual(path_coords[-1], (2,2))


    def test_a_star_with_obstacles(self):
        grid = Grid(3, 3)
        grid.start_node = grid.nodes[0][0]
        grid.end_node = grid.nodes[2][2]

        grid.nodes[1][0].is_obstacle = True
        grid.nodes[1][1].is_obstacle = True
        grid.update_all_node_neighbors()

        path_nodes, _, _ = a_star(grid, grid.start_node, grid.end_node)
        path_coords = self._convert_path_to_coords(path_nodes)
        expected_path = [(0,0), (0,1), (0,2), (1,2), (2,2)] # Same as Dijkstra for this setup
        self.assertEqual(path_coords, expected_path)

    def test_a_star_no_path(self):
        grid = Grid(3, 3)
        grid.start_node = grid.nodes[0][0]
        grid.end_node = grid.nodes[2][2]

        grid.nodes[0][1].is_obstacle = True
        grid.nodes[1][0].is_obstacle = True
        grid.nodes[1][1].is_obstacle = True
        grid.update_all_node_neighbors()

        path_nodes, _, _ = a_star(grid, grid.start_node, grid.end_node)
        self.assertEqual(self._convert_path_to_coords(path_nodes), [])

    def test_a_star_start_is_obstacle(self):
        grid = Grid(3, 3)
        grid.start_node = grid.nodes[0][0]
        grid.end_node = grid.nodes[2][2]
        grid.start_node.is_obstacle = True
        grid.update_all_node_neighbors()

        path_nodes, _, _ = a_star(grid, grid.start_node, grid.end_node)
        self.assertEqual(self._convert_path_to_coords(path_nodes), [])

    def test_a_star_end_is_obstacle(self):
        grid = Grid(3, 3)
        grid.start_node = grid.nodes[0][0]
        grid.end_node = grid.nodes[2][2]
        grid.end_node.is_obstacle = True
        grid.update_all_node_neighbors()

        path_nodes, _, _ = a_star(grid, grid.start_node, grid.end_node)
        self.assertEqual(self._convert_path_to_coords(path_nodes), [])

    def test_a_star_1x1_grid(self):
        grid = Grid(1, 1)
        grid.start_node = grid.nodes[0][0]
        grid.end_node = grid.nodes[0][0]
        grid.update_all_node_neighbors()

        path_nodes, _, _ = a_star(grid, grid.start_node, grid.end_node)
        self.assertEqual(self._convert_path_to_coords(path_nodes), [(0,0)])

    def test_a_star_start_equals_end(self):
        grid = Grid(3, 3)
        grid.start_node = grid.nodes[1][1]
        grid.end_node = grid.nodes[1][1]
        grid.update_all_node_neighbors()

        path_nodes, _, _ = a_star(grid, grid.start_node, grid.end_node)
        self.assertEqual(self._convert_path_to_coords(path_nodes), [(1,1)])

    # --- Heuristic Test (Optional, as A* tests it implicitly) ---
    def test_heuristic_manhattan_distance(self):
        # Node constructor: row, col, total_rows, total_cols, width=0, height=0
        node1 = Node(0, 0, 3, 3)
        node2 = Node(2, 2, 3, 3)
        self.assertEqual(heuristic(node1, node2), 4)

        node3 = Node(0, 0, 5, 5)
        node4 = Node(4, 0, 5, 5)
        self.assertEqual(heuristic(node3, node4), 4)

    def test_grid_instantiation_only(self): # Keep this simple test
        """Tests if Grid instantiation alone causes issues."""
        grid = Grid(1, 1)
        self.assertIsNotNone(grid)
        node = grid.nodes[0][0]
        self.assertIsNotNone(node)

    # --- D* Lite Tests ---
    def test_d_star_lite_simple_path(self):
        reset_d_star_lite_internals()
        grid = Grid(3, 3)
        grid.start_node = grid.nodes[0][0]
        grid.set_target_node(2, 2) # Sets grid.end_node
        grid.update_all_node_neighbors()

        path_nodes, _, _ = run_d_star_lite(grid, grid.start_node, grid.end_node, heuristic)
        path_coords = self._convert_path_to_coords(path_nodes)

        self.assertEqual(len(path_coords), 5, "D* Lite simple path length incorrect")
        self.assertEqual(path_coords[0], (0,0), "D* Lite simple path start incorrect")
        self.assertEqual(path_coords[-1], (2,2), "D* Lite simple path end incorrect")

    def test_d_star_lite_with_obstacles(self):
        reset_d_star_lite_internals()
        grid = Grid(3, 3)
        grid.start_node = grid.nodes[0][0]
        grid.set_target_node(2, 2)

        grid.nodes[1][0].is_obstacle = True
        grid.nodes[1][1].is_obstacle = True
        grid.update_all_node_neighbors()

        path_nodes, _, _ = run_d_star_lite(grid, grid.start_node, grid.end_node, heuristic)
        path_coords = self._convert_path_to_coords(path_nodes)
        expected_path = [(0,0), (0,1), (0,2), (1,2), (2,2)]
        self.assertEqual(path_coords, expected_path, "D* Lite path with obstacles incorrect")

    def test_d_star_lite_no_path(self):
        reset_d_star_lite_internals()
        grid = Grid(3, 3)
        grid.start_node = grid.nodes[0][0]
        grid.set_target_node(2, 2)

        grid.nodes[0][1].is_obstacle = True
        grid.nodes[1][0].is_obstacle = True
        grid.nodes[1][1].is_obstacle = True # Wall off start node
        grid.update_all_node_neighbors()

        path_nodes, _, _ = run_d_star_lite(grid, grid.start_node, grid.end_node, heuristic)
        self.assertEqual(self._convert_path_to_coords(path_nodes), [], "D* Lite no path scenario failed")

    def test_d_star_lite_start_is_obstacle(self):
        reset_d_star_lite_internals()
        grid = Grid(3, 3)
        start_node_obj = grid.nodes[0][0]
        grid.start_node = start_node_obj
        grid.set_target_node(2, 2)
        start_node_obj.is_obstacle = True # Set start as obstacle
        grid.update_all_node_neighbors() # Important for D* Lite to know about the obstacle

        path_nodes, _, _ = run_d_star_lite(grid, grid.start_node, grid.end_node, heuristic)
        self.assertEqual(self._convert_path_to_coords(path_nodes), [], "D* Lite start is obstacle scenario failed")

    def test_d_star_lite_target_is_obstacle(self): # Renamed from end_is_obstacle for D* context
        reset_d_star_lite_internals()
        grid = Grid(3, 3)
        grid.start_node = grid.nodes[0][0]
        # Try to set target on an obstacle node using set_target_node
        # grid.nodes[2][2].is_obstacle = True # Make the target an obstacle first
        # self.assertFalse(grid.set_target_node(2,2)) # This should fail

        # Or, if target is set, then it becomes an obstacle (D* should handle this)
        target_node_obj = grid.nodes[2][2]
        grid.set_target_node(2,2) # Set target first
        target_node_obj.is_obstacle = True # Then make it an obstacle
        grid.update_all_node_neighbors()


        path_nodes, _, _ = run_d_star_lite(grid, grid.start_node, grid.end_node, heuristic)
        # D* Lite should not find a path if the goal itself is an obstacle.
        self.assertEqual(self._convert_path_to_coords(path_nodes), [], "D* Lite target is obstacle scenario failed")

    def test_d_star_lite_target_move_simple(self):
        reset_d_star_lite_internals()
        grid = Grid(3, 3)
        grid.start_node = grid.nodes[0][0]

        # Initial target and path
        grid.set_target_node(2, 2)
        grid.update_all_node_neighbors()
        path_nodes1, _, _ = run_d_star_lite(grid, grid.start_node, grid.end_node, heuristic)
        path_coords1 = self._convert_path_to_coords(path_nodes1)
        self.assertEqual(len(path_coords1), 5, "D* Lite initial path length incorrect")

        reset_d_star_lite_internals() # Reset for the next run with moved target
        # Move target
        grid.set_target_node(0, 2) # New target
        # Re-running D* Lite (which includes initialization)
        # No new obstacles, so update_all_node_neighbors not strictly needed unless target change affects it.
        # For D* Lite, if only target moves, heuristic values change, which is handled by re-calc of keys.
        # The run_d_star_lite function re-initializes, so it's a full replan.
        path_nodes2, _, _ = run_d_star_lite(grid, grid.start_node, grid.end_node, heuristic)
        path_coords2 = self._convert_path_to_coords(path_nodes2)
        expected_path2 = [(0,0), (0,1), (0,2)]
        self.assertEqual(path_coords2, expected_path2, "D* Lite moved target path incorrect")

    def test_d_star_lite_target_move_with_obstacles(self):
        reset_d_star_lite_internals()
        grid = Grid(4, 4)
        grid.start_node = grid.nodes[0][0]

        # Obstacles
        grid.nodes[1][0].is_obstacle = True
        grid.nodes[1][1].is_obstacle = True
        grid.nodes[1][2].is_obstacle = True
        grid.update_all_node_neighbors()

        # Initial target and path (e.g., to (0,3))
        grid.set_target_node(0, 3)
        path_nodes1, _, _ = run_d_star_lite(grid, grid.start_node, grid.end_node, heuristic)
        path_coords1 = self._convert_path_to_coords(path_nodes1)
        # Expected: (0,0) -> (0,1) -> (0,2) -> (0,3) - This path is blocked by (1,0),(1,1),(1,2) if start is (0,0)
        # Corrected for obstacle: No, (0,0) to (0,3) is fine with obstacles at row 1.
        expected_path1 = [(0,0), (0,1), (0,2), (0,3)]
        self.assertEqual(path_coords1, expected_path1, "D* Lite initial path with obstacles incorrect")


        reset_d_star_lite_internals()
        # Move target to (3,3), path must go around obstacles at row 1
        grid.set_target_node(3, 3)
        # Obstacles already set, neighbors updated. If set_target_node could change obstacle status, re-update.
        # run_d_star_lite will re-initialize and compute.
        path_nodes2, _, _ = run_d_star_lite(grid, grid.start_node, grid.end_node, heuristic)
        path_coords2 = self._convert_path_to_coords(path_nodes2)

        # Expected path might be (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3)->(3,3) (length 7)
        # Or (0,0) -> (x,y) ... -> (3,3)
        # Path: (0,0) -> (0,1) -> (0,2) -> (0,3) -> (1,3) -> (2,3) -> (3,3)
        # Path: (0,0) (0,0) -> (0,1) -> (0,2) -> (0,3) -> (1,3) -> (2,3) -> (3,3)
        # (0,0) -> (0,1) -> (0,2) -> (0,3)  [cost 3, h to (3,3) is 3, f=6]
        # (1,3) g=4, h=2, f=6
        # (2,3) g=5, h=1, f=6
        # (3,3) g=6, h=0, f=6
        # This path is [(0,0), (0,1), (0,2), (0,3), (1,3), (2,3), (3,3)]
        self.assertEqual(len(path_coords2), 7, "D* Lite moved target path with obstacles length incorrect")
        self.assertEqual(path_coords2[-1], (3,3), "D* Lite moved target path with obstacles end incorrect")


if __name__ == '__main__':
    unittest.main()
