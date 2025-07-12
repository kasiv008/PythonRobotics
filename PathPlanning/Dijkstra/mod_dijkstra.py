"""
Grid based Dijkstra planning with costmap integration

Modified from original author: Atsushi Sakai(@Atsushi_twi)
Added costmap support for realistic path planning
"""

import matplotlib.pyplot as plt
import math
import numpy as np

show_animation = True

class DijkstraPlanner:

    def __init__(self, costmap_file, resolution, robot_radius, map_origin=[0.0, 0.0]):
        """
        Initialize map for dijkstra planning with costmap

        costmap_file: path to numpy file containing costmap
        resolution: grid resolution [m]
        robot_radius: robot radius[m]
        map_origin: [x, y] world coordinates of costmap[0,0]
        """
        
        # Load costmap from numpy file
        self.costmap = np.load(costmap_file)
        print(f"Loaded costmap with shape: {self.costmap.shape}")
        
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.map_origin = map_origin
        
        # Calculate map bounds based on costmap dimensions
        self.min_x = map_origin[0]
        self.min_y = map_origin[1]
        self.max_x = map_origin[0] + self.costmap.shape[1] * resolution
        self.max_y = map_origin[1] + self.costmap.shape[0] * resolution
        
        self.x_width = self.costmap.shape[1]
        self.y_width = self.costmap.shape[0]
        
        print(f"Map bounds: x[{self.min_x:.2f}, {self.max_x:.2f}], y[{self.min_y:.2f}, {self.max_y:.2f}]")
        print(f"Grid size: {self.x_width} x {self.y_width}")
        
        # Create obstacle map from costmap (high cost areas become obstacles)
        self.obstacle_threshold = 50  # Adjust this threshold based on your costmap values
        self.create_obstacle_map_from_costmap()
        
        self.motion = self.get_motion_model()

    def create_obstacle_map_from_costmap(self):
        """Create binary obstacle map from costmap"""
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        
        # Mark high-cost areas as obstacles
        for ix in range(self.x_width):
            for iy in range(self.y_width):
                if self.costmap[iy, ix] > self.obstacle_threshold:
                    self.obstacle_map[ix][iy] = True
        
        print(f"Created obstacle map with threshold: {self.obstacle_threshold}")

    def world_to_grid(self, world_x, world_y):
        """Convert world coordinates to grid indices"""
        grid_x = int((world_x - self.min_x) / self.resolution)
        grid_y = int((world_y - self.min_y) / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid indices to world coordinates"""
        world_x = grid_x * self.resolution + self.min_x
        world_y = grid_y * self.resolution + self.min_y
        return world_x, world_y

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index  # index of previous Node

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        dijkstra path search with costmap integration

        input:
            sx: start x position [m]
            sy: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        # Convert world coordinates to grid indices
        start_x, start_y = self.world_to_grid(sx, sy)
        goal_x, goal_y = self.world_to_grid(gx, gy)
        
        start_node = self.Node(start_x, start_y, 0.0, -1)
        goal_node = self.Node(goal_x, goal_y, 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_index(start_node)] = start_node

        while True:
            if not open_set:
                print("No path found!")
                return [], []
                
            c_id = min(open_set, key=lambda o: open_set[o].cost)
            current = open_set[c_id]

            # # show graph
            # if show_animation:  # pragma: no cover
            #     world_x, world_y = self.grid_to_world(current.x, current.y)
            #     plt.plot(world_x, world_y, "xc")
            #     # for stopping simulation with the esc key.
            #     plt.gcf().canvas.mpl_connect(
            #         'key_release_event',
            #         lambda event: [exit(0) if event.key == 'escape' else None])
            #     if len(closed_set.keys()) % 10 == 0:
            #         plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand search grid based on motion model
            for move_x, move_y, move_cost in self.motion:
                node = self.Node(current.x + move_x,
                                 current.y + move_y,
                                 current.cost + move_cost, c_id)
                n_id = self.calc_index(node)

                if n_id in closed_set:
                    continue

                if not self.verify_node(node):
                    continue

                # Add costmap cost to the movement cost
                costmap_cost = self.get_costmap_cost(node.x, node.y)
                node.cost = current.cost + move_cost + costmap_cost

                if n_id not in open_set:
                    open_set[n_id] = node  # Discover a new node
                else:
                    if open_set[n_id].cost >= node.cost:
                        # This path is the best until now. record it!
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)
        return rx, ry

    def get_costmap_cost(self, grid_x, grid_y):
        """Get cost from costmap at given grid position"""
        if (0 <= grid_x < self.x_width and 0 <= grid_y < self.y_width):
            # Normalize costmap values to reasonable range for pathfinding
            raw_cost = self.costmap[grid_y, grid_x]
            return raw_cost * 0.1  # Scale factor - adjust as needed
        else:
            return 1000  # High cost for out-of-bounds

    def calc_final_path(self, goal_node, closed_set):
        # generate final course in world coordinates
        world_x, world_y = self.grid_to_world(goal_node.x, goal_node.y)
        rx, ry = [world_x], [world_y]
        
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            world_x, world_y = self.grid_to_world(n.x, n.y)
            rx.append(world_x)
            ry.append(world_y)
            parent_index = n.parent_index

        return rx, ry

    def calc_index(self, node):
        return node.y * self.x_width + node.x

    def verify_node(self, node):
        # Check bounds
        if node.x < 0 or node.x >= self.x_width:
            return False
        if node.y < 0 or node.y >= self.y_width:
            return False

        # Check obstacle map
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion

    def visualize_costmap(self):
        """Visualize the costmap"""
        if show_animation:
            plt.figure(figsize=(12, 8))
            
            # Create extent for proper coordinate display
            extent = [self.min_x, self.max_x, self.min_y, self.max_y]
            
            plt.imshow(self.costmap, cmap='hot', alpha=0.7, origin='lower', extent=extent)
            plt.colorbar(label='Cost')
            plt.title('Costmap')
            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.grid(True, alpha=0.3)


def main():
    print(__file__ + " start!!")

    # Costmap parameters
    costmap_file = "/home/level5_kasi/Downloads/cavasos_costmap_final.npy"  # Replace with your actual costmap file
    grid_size = 3  # [m] - should match your costmap resolution
    robot_radius = 2  # [m]
    map_origin = [0.0, 0.0]  # [m] - world coordinates of costmap[0,0]

    # start and goal position (in world coordinates)
    sx = 100.0   # [m]
    sy = 100.0   # [m]
    gx = 700.0  # [m]
    gy = 700.0  # [m]

    try:
        # Initialize planner with costmap
        dijkstra = DijkstraPlanner(costmap_file, grid_size, robot_radius, map_origin)
        
        # if show_animation:
        #     # Visualize costmap
        #     dijkstra.visualize_costmap()
            
        #     # Plot start and goal
        #     plt.plot(sx, sy, "og", markersize=10, label="Start")
        #     plt.plot(gx, gy, "xb", markersize=10, label="Goal")
        #     plt.legend()
        #     plt.title('Dijkstra Path Planning with Costmap')

        # Plan path
        print(f"Planning path from ({sx}, {sy}) to ({gx}, {gy})")
        rx, ry = dijkstra.planning(sx, sy, gx, gy)

        if rx and ry:
            print(f"Path found with {len(rx)} waypoints")
            print(f"Path length: {sum(math.hypot(rx[i+1]-rx[i], ry[i+1]-ry[i]) for i in range(len(rx)-1)):.2f} m")
            
            if show_animation:
                plt.plot(rx, ry, "-r", linewidth=3, label="Dijkstra Path")
                plt.legend()
                plt.grid(True)
                plt.axis("equal")
                plt.show()
        else:
            print("No path found!")
            
    except FileNotFoundError:
        print(f"Error: Could not load costmap file '{costmap_file}'")
        print("Please make sure the file exists and the path is correct.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
