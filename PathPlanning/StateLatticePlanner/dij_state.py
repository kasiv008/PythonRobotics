import numpy as np
import heapq
from time import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Arrow
import math
import sys
import os
import pathlib
from collections import deque

# Add paths for state lattice modules (adjust as needed)
sys.path.append(str(pathlib.Path(__file__).parent.parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

try:
    import ModelPredictiveTrajectoryGenerator.trajectory_generator as planner
    import ModelPredictiveTrajectoryGenerator.motion_model as motion_model
except:
    print("State lattice modules not found, using simplified optimization")

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
        self.start = [int(start[0]), int(start[1])]
        self.goal = [int(goal[0]), int(goal[1])]
        self.found_path = False
        self.unvisited = []
        heapq.heappush(self.unvisited, (0, tuple(self.start)))
        print(f"    DIJKSTRA RUNNING: {self.start} -> {self.goal}")

    def find_path(self):
        nodes_explored = 0
        
        while self.unvisited:
            current_dist, current_node = heapq.heappop(self.unvisited)
            x, y = current_node

            x, y = int(x), int(y)
            if x < 0 or x >= self.grid.shape[0] or y < 0 or y >= self.grid.shape[1]:
                continue

            if self.grid[x, y].visited:
                continue

            self.grid[x, y].visited = True
            nodes_explored += 1
            
            # Debug output every 1000 nodes
            if nodes_explored % 1000 == 0:
                print(f"      Dijkstra explored {nodes_explored} nodes, current: ({x}, {y}), cost: {current_dist:.1f}")

            if current_node == tuple(self.goal):
                self.found_path = True
                print(f"    DIJKSTRA SUCCESS: Found path after exploring {nodes_explored} nodes")
                break

            for nx, ny in self.get_neighbours(x, y):
                if not self.grid[nx, ny].visited:
                    # FIXED: Properly use costmap values for pathfinding
                    terrain_cost = max(1.0, float(self.grid[nx, ny].value))
                    
                    # Add diagonal movement penalty
                    if abs(nx - x) + abs(ny - y) == 2:  # Diagonal move
                        movement_cost = terrain_cost * 1.414  # sqrt(2) penalty
                    else:  # Cardinal move
                        movement_cost = terrain_cost
                    
                    tentative_dist = self.grid[x, y].dist + movement_cost
                    
                    if tentative_dist < self.grid[nx, ny].dist:
                        self.grid[nx, ny].dist = tentative_dist
                        self.grid[nx, ny].backtrack = (x, y)
                        heapq.heappush(self.unvisited, (tentative_dist, (nx, ny)))
        
        if not self.found_path:
            print(f"    DIJKSTRA FAILED: No path found after exploring {nodes_explored} nodes")

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

class StateLatticeOptimizer:
    def __init__(self, costmap, resolution=3):
        self.costmap = costmap
        self.resolution = resolution
        
    def grid_to_world(self, grid_pos, map_origin=[0, 0]):
        world_x = grid_pos[1] * self.resolution + map_origin[0]
        world_y = grid_pos[0] * self.resolution + map_origin[1]
        return [world_x, world_y]

    def world_to_grid(self, world_pos, map_origin=[0, 0]):
        grid_row = int((world_pos[1] - map_origin[1]) / self.resolution)
        grid_col = int((world_pos[0] - map_origin[0]) / self.resolution)
        return [grid_row, grid_col]

    def calculate_trajectory_cost(self, x_traj, y_traj):
        total_cost = 0
        valid_points = 0
        
        for x, y in zip(x_traj, y_traj):
            grid_pos = self.world_to_grid([x, y])
            if (0 <= grid_pos[0] < self.costmap.shape[0] and 
                0 <= grid_pos[1] < self.costmap.shape[1]):
                total_cost += self.costmap[grid_pos[0], grid_pos[1]]
                valid_points += 1
            else:
                total_cost += 1000
                valid_points += 1
        
        return total_cost / valid_points if valid_points > 0 else float('inf')

    def optimize_segment(self, start_world, goal_world):
        dx = goal_world[0] - start_world[0]
        dy = goal_world[1] - start_world[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < 0.5:
            return [start_world, goal_world]
        
        num_points = max(10, int(distance / (self.resolution * 2)))
        t_values = np.linspace(0, 1, num_points)
        
        best_trajectory = []
        best_cost = float('inf')
        
        for curve_factor in [-0.3, -0.1, 0.0, 0.1, 0.3]:
            x_traj = []
            y_traj = []
            
            for t in t_values:
                x = start_world[0] + t * dx
                y = start_world[1] + t * dy
                
                curve_offset = curve_factor * distance * 0.1 * math.sin(math.pi * t)
                perp_x = -dy / distance if distance > 0 else 0
                perp_y = dx / distance if distance > 0 else 0
                
                x += curve_offset * perp_x
                y += curve_offset * perp_y
                
                x_traj.append(x)
                y_traj.append(y)
            
            cost = self.calculate_trajectory_cost(x_traj, y_traj)
            if cost < best_cost:
                best_cost = cost
                best_trajectory = list(zip(x_traj, y_traj))
        
        return best_trajectory if best_trajectory else [start_world, goal_world]

class DynamicHybridPlanner:
    def __init__(self, costmap, resolution=3):
        self.costmap = costmap
        self.resolution = resolution
        self.state_lattice = StateLatticeOptimizer(costmap, resolution)
        
        self.vehicle_speed = 2.0
        self.optimization_lookahead = 25
        
        self.global_dijkstra_path = None
        self.current_optimized_segment = None
        
    def ensure_valid_position(self, pos):
        x = max(0, min(int(round(pos[0])), self.costmap.shape[0] - 1))
        y = max(0, min(int(round(pos[1])), self.costmap.shape[1] - 1))
        return [x, y]
        
    def plan_global_path(self, current_pos, final_goal):
        """Plan complete path from current position to final goal using Dijkstra"""
        current_pos = self.ensure_valid_position(current_pos)
        final_goal = self.ensure_valid_position(final_goal)
        
        print(f"  === RUNNING DIJKSTRA FROM {current_pos} TO {final_goal} ===")
        
        if current_pos == final_goal:
            return [current_pos]
        
        try:
            start_time = time()
            grid = Grid(self.costmap, current_pos)
            dijkstra_solver = Dijkstra(grid, current_pos, final_goal)
            dijkstra_solver.find_path()
            dijkstra_path = dijkstra_solver.backtrack_path()
            planning_time = time() - start_time
            
            if not dijkstra_path or len(dijkstra_path) < 2:
                print("  DIJKSTRA FAILED - using direct path")
                dijkstra_path = [current_pos, final_goal]
            else:
                path_length = len(dijkstra_path)
                actual_distance = sum(
                    math.sqrt((dijkstra_path[i+1][0] - dijkstra_path[i][0])**2 + 
                             (dijkstra_path[i+1][1] - dijkstra_path[i][1])**2) * 3
                    for i in range(len(dijkstra_path)-1)
                )
                print(f"  SUCCESS: {path_length} waypoints, {actual_distance:.0f}m total, {planning_time:.3f}s")
            
            return dijkstra_path
            
        except Exception as e:
            print(f"  ERROR: {e}")
            return [current_pos, final_goal]
    
    def optimize_next_segment(self, dijkstra_path, current_pos_index):
        if not dijkstra_path or current_pos_index >= len(dijkstra_path):
            return None
        
        end_index = min(current_pos_index + self.optimization_lookahead, len(dijkstra_path) - 1)
        segment_to_optimize = dijkstra_path[current_pos_index:end_index + 1]
        
        if len(segment_to_optimize) < 2:
            return None
        
        print(f"  Optimizing segment: waypoints {current_pos_index} to {end_index} ({len(segment_to_optimize)} points)")
        
        optimized_path = []
        segment_length = 8
        
        i = 0
        while i < len(segment_to_optimize):
            segment_end = min(i + segment_length, len(segment_to_optimize) - 1)
            
            start_world = self.state_lattice.grid_to_world(segment_to_optimize[i])
            end_world = self.state_lattice.grid_to_world(segment_to_optimize[segment_end])
            
            optimized_segment = self.state_lattice.optimize_segment(start_world, end_world)
            
            if i == 0:
                optimized_path.extend(optimized_segment)
            else:
                optimized_path.extend(optimized_segment[1:])
            
            i = segment_end
            if i >= len(segment_to_optimize) - 1:
                break
        
        return optimized_path

class DynamicPathVisualizer:
    def __init__(self, costmap, start, goal, planner):
        self.costmap = costmap
        self.start = start
        self.goal = goal
        self.planner = planner
        
        self.current_pos = [float(start[0]), float(start[1])]
        self.current_path_index = 0
        self.step_count = 0
        
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        self.setup_plot()
        
        self.frames_data = []
        
    def setup_plot(self):
        # Use hot colormap to better show cost variations
        self.ax.imshow(self.costmap, cmap='hot', alpha=0.7, origin='upper', vmin=0, vmax=100)
        self.ax.set_title('Dynamic Hybrid Path Planning - Dijkstra Global + State Lattice Local', 
                         fontsize=16, fontweight='bold')
        self.ax.set_xlabel('Grid X (columns)', fontsize=12)
        self.ax.set_ylabel('Grid Y (rows)', fontsize=12)
        
        self.ax.plot(self.start[1], self.start[0], 'go', markersize=15, 
                    label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
        self.ax.plot(self.goal[1], self.goal[0], 'ro', markersize=15, 
                    label='Goal', markeredgecolor='darkred', markeredgewidth=2)
        
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        
    def update_visualization(self, frame_num):
        if frame_num >= len(self.frames_data):
            return []
        
        frame_data = self.frames_data[frame_num]
        
        # Clear previous dynamic elements
        for artist in self.ax.lines[2:]:
            artist.remove()
        for artist in self.ax.patches:
            artist.remove()
        
        # Plot complete global Dijkstra path
        if frame_data['global_dijkstra_path'] and len(frame_data['global_dijkstra_path']) > 1:
            path_array = np.array(frame_data['global_dijkstra_path'])
            self.ax.plot(path_array[:, 1], path_array[:, 0], 'r-', 
                        linewidth=2, alpha=0.7, label='Global Dijkstra Path')
        
        # Highlight current segment being optimized
        if frame_data['current_segment'] and len(frame_data['current_segment']) > 1:
            segment_array = np.array(frame_data['current_segment'])
            self.ax.plot(segment_array[:, 1], segment_array[:, 0], 'r-', 
                        linewidth=4, alpha=0.9, label='Current Dijkstra Segment')
        
        # Plot optimized segment
        if frame_data['optimized_segment'] and len(frame_data['optimized_segment']) > 1:
            opt_grid = []
            for world_point in frame_data['optimized_segment']:
                grid_row = int(world_point[1] / self.planner.resolution)
                grid_col = int(world_point[0] / self.planner.resolution)
                grid_row = max(0, min(grid_row, self.costmap.shape[0] - 1))
                grid_col = max(0, min(grid_col, self.costmap.shape[1] - 1))
                opt_grid.append([grid_col, grid_row])
            
            if len(opt_grid) > 1:
                opt_array = np.array(opt_grid)
                self.ax.plot(opt_array[:, 0], opt_array[:, 1], 'b-', 
                            linewidth=5, alpha=0.9, label='State Lattice Optimized')
        
        # Plot vehicle trajectory
        if len(frame_data['vehicle_history']) > 1:
            history_array = np.array(frame_data['vehicle_history'])
            self.ax.plot(history_array[:, 1], history_array[:, 0], 'g-', 
                        linewidth=3, alpha=0.8, label='Vehicle Trajectory')
        
        # Plot vehicle position
        current_pos = frame_data['current_pos']
        vehicle_circle = Circle((current_pos[1], current_pos[0]), 8, 
                               color='orange', alpha=0.9, zorder=10)
        self.ax.add_patch(vehicle_circle)
        
        # Direction arrow
        if frame_data['optimized_segment'] and len(frame_data['optimized_segment']) > 5:
            next_world = frame_data['optimized_segment'][min(5, len(frame_data['optimized_segment'])-1)]
            next_grid = [next_world[1] / self.planner.resolution, 
                        next_world[0] / self.planner.resolution]
            
            dx = next_grid[0] - current_pos[1]
            dy = next_grid[1] - current_pos[0]
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                arrow = Arrow(current_pos[1], current_pos[0], dx*0.5, dy*0.5, 
                             width=5, color='red', alpha=0.8, zorder=11)
                self.ax.add_patch(arrow)
        
        self.ax.legend(loc='upper right', fontsize=10)
        
        # Show real-world distance
        real_distance = frame_data["distance_to_goal"] * 3
        self.ax.set_title(f'Step {frame_data["step"]} - Distance: {real_distance:.0f}m\n'
                         f'Global Path: {frame_data["total_waypoints"]} waypoints, '
                         f'Optimizing: {frame_data["segment_start"]}-{frame_data["segment_end"]}', 
                         fontsize=12, fontweight='bold')
        
        return []
    
    def simulate_dynamic_planning(self, max_steps=150):  # Reduced for testing
        print("Starting dynamic planning simulation...")
        
        vehicle_history = [list(self.current_pos)]
        replan_frequency = 10  # More frequent replanning to see Dijkstra in action
        
        for step in range(max_steps):
            distance_to_goal = math.sqrt(
                (self.goal[0] - self.current_pos[0])**2 + 
                (self.goal[1] - self.current_pos[1])**2
            )
            
            print(f"Step {step}: Position [{self.current_pos[0]:.1f}, {self.current_pos[1]:.1f}], Distance: {distance_to_goal:.1f} ({distance_to_goal*3:.0f}m)")
            
            if distance_to_goal < 5:
                print("Goal reached!")
                break
            
            # More frequent replanning to see Dijkstra in action
            need_global_replan = (
                self.planner.global_dijkstra_path is None or 
                step % replan_frequency == 0 or
                self.current_path_index >= len(self.planner.global_dijkstra_path) - 5
            )
            
            if need_global_replan:
                print(f"  *** STEP {step}: REPLANNING GLOBAL PATH ***")
                self.planner.global_dijkstra_path = self.planner.plan_global_path(
                    self.current_pos, self.goal
                )
                self.current_path_index = 0
            
            # Optimize next segment
            if self.planner.global_dijkstra_path:
                min_dist = float('inf')
                closest_index = 0
                for i, waypoint in enumerate(self.planner.global_dijkstra_path):
                    dist = math.sqrt((waypoint[0] - self.current_pos[0])**2 + 
                                   (waypoint[1] - self.current_pos[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_index = i
                
                self.current_path_index = closest_index
                
                self.planner.current_optimized_segment = self.planner.optimize_next_segment(
                    self.planner.global_dijkstra_path, self.current_path_index
                )
                
                end_index = min(self.current_path_index + self.planner.optimization_lookahead, 
                              len(self.planner.global_dijkstra_path) - 1)
                current_segment = self.planner.global_dijkstra_path[self.current_path_index:end_index + 1]
            else:
                current_segment = None
                self.planner.current_optimized_segment = None
            
            # Store frame data
            frame_data = {
                'step': step,
                'current_pos': list(self.current_pos),
                'global_dijkstra_path': self.planner.global_dijkstra_path.copy() if self.planner.global_dijkstra_path else None,
                'current_segment': current_segment.copy() if current_segment else None,
                'optimized_segment': self.planner.current_optimized_segment.copy() if self.planner.current_optimized_segment else None,
                'vehicle_history': vehicle_history.copy(),
                'distance_to_goal': distance_to_goal,
                'total_waypoints': len(self.planner.global_dijkstra_path) if self.planner.global_dijkstra_path else 0,
                'segment_start': self.current_path_index,
                'segment_end': min(self.current_path_index + self.planner.optimization_lookahead, 
                                 len(self.planner.global_dijkstra_path) - 1) if self.planner.global_dijkstra_path else 0
            }
            self.frames_data.append(frame_data)
            
            # Move vehicle
            if self.planner.current_optimized_segment and len(self.planner.current_optimized_segment) > 0:
                lookahead_index = min(2, len(self.planner.current_optimized_segment) - 1)
                target_world = self.planner.current_optimized_segment[lookahead_index]
                target_grid = self.planner.state_lattice.world_to_grid(target_world)
                
                dx = target_grid[0] - self.current_pos[0]
                dy = target_grid[1] - self.current_pos[1]
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance > 0.1:
                    move_x = (dx / distance) * self.planner.vehicle_speed
                    move_y = (dy / distance) * self.planner.vehicle_speed
                    
                    self.current_pos[0] += move_x
                    self.current_pos[1] += move_y
                    
                    self.current_pos[0] = max(0, min(self.current_pos[0], self.costmap.shape[0] - 1))
                    self.current_pos[1] = max(0, min(self.current_pos[1], self.costmap.shape[1] - 1))
                
                vehicle_history.append(list(self.current_pos))
        
        print(f"Simulation completed with {len(self.frames_data)} frames")
        
    def create_animation(self, filename='dijkstra_fixed.gif', fps=8):
        if len(self.frames_data) == 0:
            print("No frames to animate!")
            return None
            
        print(f"Creating animation with {len(self.frames_data)} frames...")
        
        anim = animation.FuncAnimation(
            self.fig, self.update_visualization, 
            frames=len(self.frames_data),
            interval=1000//fps,
            blit=False,
            repeat=True
        )
        
        print(f"Saving animation to {filename}...")
        try:
            anim.save(filename, writer='pillow', fps=fps, dpi=100)
            print(f"Animation saved successfully!")
        except Exception as e:
            print(f"Error saving animation: {e}")
        
        return anim

def main():
    print("=== FIXED DIJKSTRA: CURRENT POSITION TO GOAL ===")
    
    print("Loading costmap...")
    try:
        grid_data = np.load('/home/level5_kasi/Downloads/cavasos_costmap_final.npy', allow_pickle=True)
        print(f"Costmap loaded: {grid_data.shape}")
        print(f"Value range: {grid_data.min()} to {grid_data.max()}")
    except FileNotFoundError:
        print("Costmap file not found! Creating a simple test costmap...")
        grid_data = np.ones((600, 600)) * 10
        grid_data[200:400, 200:220] = 100
        grid_data[300:320, 100:500] = 100
    
    start = [1300, 300]
    goal = [1048, 1122]
    
    start[0] = max(0, min(start[0], grid_data.shape[0] - 1))
    start[1] = max(0, min(start[1], grid_data.shape[1] - 1))
    goal[0] = max(0, min(goal[0], grid_data.shape[0] - 1))
    goal[1] = max(0, min(goal[1], grid_data.shape[1] - 1))
    
    real_distance = math.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2) * 3
    print(f"Planning from {start} to {goal}")
    print(f"Real-world distance: {real_distance:.0f} meters")
    print("Dijkstra will now properly run from current position to goal!")
    
    planner = DynamicHybridPlanner(grid_data, resolution=3)
    visualizer = DynamicPathVisualizer(grid_data, start, goal, planner)
    
    visualizer.simulate_dynamic_planning(max_steps=100)
    
    if len(visualizer.frames_data) > 0:
        animation_obj = visualizer.create_animation(
            filename='dijkstra_current_to_goal_fixed.gif', 
            fps=10
        )
        plt.show()
    else:
        print("No frames generated!")
    
    print("Simulation completed!")

if __name__ == "__main__":
    main()
