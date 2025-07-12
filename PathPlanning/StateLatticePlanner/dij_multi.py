import numpy as np
import heapq
from time import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy

class Node:
    def __init__(self, value, is_start=False):
        self.value = value
        self.dist = value if is_start else np.inf
        self.backtrack = -1 if is_start else None
        self.visited = False

class Grid:
    def __init__(self, grid, start):
        self.grid = np.empty(grid.shape, dtype=object)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                self.grid[i, j] = Node(grid[i, j], is_start=(i, j) == tuple(start))

class Dijkstra:
    def __init__(self, grid, start, goal):
        self.grid = grid.grid
        self.start = start
        self.goal = goal
        self.found_path = False
        self.unvisited = []
        heapq.heappush(self.unvisited, (0, tuple(start)))

    def find_path(self):
        while self.unvisited:
            current_dist, current_node = heapq.heappop(self.unvisited)
            x, y = current_node

            if self.grid[x, y].visited:
                continue

            self.grid[x, y].visited = True

            if current_node == tuple(self.goal):
                self.found_path = True
                break

            for nx, ny in self.get_neighbours(x, y):
                if not self.grid[nx, ny].visited:
                    tentative_dist = self.grid[x, y].dist + self.grid[nx, ny].value
                    if tentative_dist < self.grid[nx, ny].dist:
                        self.grid[nx, ny].dist = tentative_dist
                        self.grid[nx, ny].backtrack = (x, y)
                        heapq.heappush(self.unvisited, (tentative_dist, (nx, ny)))

    def get_neighbours(self, x, y):
        max_x, max_y = self.grid.shape
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        return [
            (x + dx, y + dy)
            for dx, dy in directions
            if 0 <= x + dx < max_x and 0 <= y + dy < max_y
        ]

    def backtrack_path(self):
        if not self.found_path:
            return None
        path = []
        current = tuple(self.goal)
        while current != -1:
            path.append(current)
            current = self.grid[current[0], current[1]].backtrack
        return path[::-1]

class DynamicPathPlanner:
    def __init__(self, grid_data, start, goal, waypoint_step=30):
        self.original_grid = grid_data.copy()
        self.start = start
        self.goal = goal
        self.waypoint_step = waypoint_step
        self.current_position = start.copy()
        self.all_paths = []
        self.waypoints = [tuple(start)]
        self.reached_goal = False
        
    def plan_and_move(self):
        """Plan path from current position and move waypoint_step steps"""
        if self.reached_goal:
            return False
            
        # Create fresh grid for planning
        grid = Grid(self.original_grid, self.current_position)
        dijkstra_solver = Dijkstra(grid, self.current_position, self.goal)
        dijkstra_solver.find_path()
        
        path = dijkstra_solver.backtrack_path()
        if not path:
            print("No path found from current position!")
            return False
            
        self.all_paths.append(path)
        
        # Move along the path for waypoint_step steps
        steps_taken = 0
        for i, point in enumerate(path[1:], 1):  # Skip current position
            self.current_position = list(point)
            self.waypoints.append(point)
            steps_taken += 1
            
            # Check if reached goal
            if point == tuple(self.goal):
                self.reached_goal = True
                print(f"Goal reached! Total waypoints: {len(self.waypoints)}")
                return False
                
            # Check if taken enough steps
            if steps_taken >= self.waypoint_step:
                break
                
        return True
    
    def simulate(self):
        """Run the complete simulation"""
        print("Starting dynamic path planning simulation...")
        iteration = 0
        
        while not self.reached_goal:
            iteration += 1
            print(f"Iteration {iteration}: Planning from {self.current_position}")
            
            if not self.plan_and_move():
                break
                
        print(f"Simulation completed in {iteration} iterations")
        return self.waypoints, self.all_paths

class PathAnimator:
    def __init__(self, grid_data, waypoints, all_paths, start, goal):
        self.grid_data = grid_data.copy()
        self.waypoints = waypoints
        self.all_paths = all_paths
        self.start = start
        self.goal = goal
        
        # Setup the plot
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.current_cmap = plt.cm.Blues
        self.current_cmap.set_bad(color='red')
        
        # Initialize the plot
        self.im = self.ax.matshow(self.grid_data, cmap=self.current_cmap, vmin=0, vmax=3)
        self.ax.set_title("Dynamic Dijkstra Path Planning Simulation")
        
        # Plot start and goal
        self.ax.plot(start[1], start[0], 'go', markersize=10, label='Start')
        self.ax.plot(goal[1], goal[0], 'ro', markersize=10, label='Goal')
        self.ax.legend()
        
        self.current_frame = 0
        
    def animate(self, frame):
        """Animation function"""
        if frame >= len(self.waypoints):
            return [self.im]
            
        # Clear previous paths and waypoints (except start/goal)
        display_grid = self.grid_data.copy()
        
        # Show current path if available
        path_index = min(frame // 30, len(self.all_paths) - 1)
        if path_index < len(self.all_paths):
            current_path = self.all_paths[path_index]
            for point in current_path:
                display_grid[point[0], point[1]] = 2  # Path color
        
        # Show waypoints up to current frame
        for i in range(min(frame + 1, len(self.waypoints))):
            waypoint = self.waypoints[i]
            display_grid[waypoint[0], waypoint[1]] = 3  # Waypoint color
            
        # Highlight current position
        if frame < len(self.waypoints):
            current_pos = self.waypoints[frame]
            display_grid[current_pos[0], current_pos[1]] = 3
            
        self.im.set_array(display_grid)
        self.ax.set_title(f"Dynamic Dijkstra - Waypoint {frame + 1}/{len(self.waypoints)}")
        
        return [self.im]
    
    def start_animation(self, interval=100):
        """Start the animation"""
        anim = FuncAnimation(
            self.fig, self.animate, frames=len(self.waypoints) + 10,
            interval=interval, blit=False, repeat=True
        )
        plt.show()
        return anim

def plot_final_result(grid_data, waypoints, all_paths, start, goal):
    """Plot the final result with all paths and waypoints"""
    display_grid = grid_data.copy()
    
    # Plot all paths
    for path in all_paths:
        for point in path:
            display_grid[point[0], point[1]] = 2
    
    # Plot waypoints
    for waypoint in waypoints:
        display_grid[waypoint[0], waypoint[1]] = 3
    
    current_cmap = plt.cm.Blues
    current_cmap.set_bad(color='red')
    
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.matshow(display_grid, cmap=current_cmap, vmin=0, vmax=3)
    ax.plot(start[1], start[0], 'go', markersize=12, label='Start')
    ax.plot(goal[1], goal[0], 'ro', markersize=12, label='Goal')
    ax.set_title(f"Complete Dynamic Path Planning Result\nTotal Waypoints: {len(waypoints)}")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    # Load your grid data
    grid_data = np.load('/home/level5_kasi/Downloads/cavasos_costmap_final.npy', allow_pickle=True)
    start = [1300, 300]
    goal = [1048, 1122]
    
    # Run dynamic path planning simulation
    start_time = time()
    planner = DynamicPathPlanner(grid_data, start, goal, waypoint_step=30)
    waypoints, all_paths = planner.simulate()
    print("Total execution time:", time() - start_time)
    
    # Create and start animation
    animator = PathAnimator(grid_data, waypoints, all_paths, start, goal)
    animation = animator.start_animation(interval=200)  # 200ms between frames
    
    # Plot final result
    plot_final_result(grid_data, waypoints, all_paths, start, goal)
