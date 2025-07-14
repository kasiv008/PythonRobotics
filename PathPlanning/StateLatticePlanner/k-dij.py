import numpy as np
import heapq
from time import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Arrow, Rectangle
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
        # Ensure start and goal are integers
        self.start = [int(start[0]), int(start[1])]
        self.goal = [int(goal[0]), int(goal[1])]
        self.found_path = False
        self.unvisited = []
        heapq.heappush(self.unvisited, (0, tuple(self.start)))

    def find_path(self,alpha,beta):
        while self.unvisited:
            current_dist, current_node = heapq.heappop(self.unvisited)
            x, y = current_node

            # Ensure indices are integers and within bounds
            x, y = int(x), int(y)
            if x < 0 or x >= self.grid.shape[0] or y < 0 or y >= self.grid.shape[1]:
                continue

            if self.grid[x, y].visited:
                continue

            self.grid[x, y].visited = True

            if current_node == tuple(self.goal):
                self.found_path = True
                break

            for nx, ny in self.get_neighbours(x, y):
                if not self.grid[nx, ny].visited:
                    tentative_dist = alpha*self.grid[x, y].dist + beta*self.grid[nx, ny].value ## value that is being optimized
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
        
        # Generate smooth trajectory using spline-like interpolation
        num_points = max(10, int(distance / (self.resolution * 2)))
        t_values = np.linspace(0, 1, num_points)
        
        # Create smooth curve with some optimization
        best_trajectory = []
        best_cost = float('inf')
        
        # Try different curve variations
        for curve_factor in [-0.3, -0.1, 0.0, 0.1, 0.3]:
            x_traj = []
            y_traj = []
            
            for t in t_values:
                # Smooth interpolation with curve
                x = start_world[0] + t * dx
                y = start_world[1] + t * dy
                
                # Add smooth curvature
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
    def __init__(self, costmap, resolution=3, alpha=1.0, beta=1.0):
        self.costmap = costmap
        self.resolution = resolution
        self.state_lattice = StateLatticeOptimizer(costmap, resolution)

        # Vehicle parameters
        self.vehicle_speed = 2.0  # grid units per step
        self.optimization_lookahead = 25  # Only optimize next 25 waypoints
        self.alpha = alpha
        self.beta = beta
        
        # Current global path
        self.global_dijkstra_path = None
        self.current_optimized_segment = None
        
    def ensure_valid_position(self, pos):
        """Ensure position is valid integer coordinates within bounds"""
        x = max(0, min(int(round(pos[0])), self.costmap.shape[0] - 1))
        y = max(0, min(int(round(pos[1])), self.costmap.shape[1] - 1))
        return [x, y]
        
    def plan_global_path(self, current_pos, final_goal):
        """Plan complete path from current position to final goal using Dijkstra"""
        # Ensure positions are valid integers
        current_pos = self.ensure_valid_position(current_pos)
        final_goal = self.ensure_valid_position(final_goal)
        
        print(f"  Planning global Dijkstra path from {current_pos} to {final_goal}")
        
        # Check if start and goal are the same
        if current_pos == final_goal:
            return [current_pos]
        
        try:
            # Plan complete path with Dijkstra to final goal
            grid = Grid(self.costmap, current_pos)
            dijkstra_solver = Dijkstra(grid, current_pos, final_goal)
            dijkstra_solver.find_path(self.alpha,self.beta)
            dijkstra_path = dijkstra_solver.backtrack_path()
            
            if not dijkstra_path or len(dijkstra_path) < 2:
                # Fallback: direct line path
                print("  Dijkstra failed, using direct path")
                dijkstra_path = [current_pos, final_goal]
            else:
                print(f"  Global Dijkstra path found with {len(dijkstra_path)} waypoints")
            
            return dijkstra_path
            
        except Exception as e:
            print(f"  Global planning error: {e}")
            # Fallback path
            return [current_pos, final_goal]
    
    def optimize_next_segment(self, dijkstra_path, current_pos_index):
        """Optimize only the next 25 waypoints from current position"""
        if not dijkstra_path or current_pos_index >= len(dijkstra_path):
            return None
        
        # Get the next segment to optimize (up to 25 waypoints)
        end_index = min(current_pos_index + self.optimization_lookahead, len(dijkstra_path) - 1)
        segment_to_optimize = dijkstra_path[current_pos_index:end_index + 1]
        
        if len(segment_to_optimize) < 2:
            return None
        
        print(f"  Optimizing segment: waypoints {current_pos_index} to {end_index} ({len(segment_to_optimize)} points)")
        
        # Optimize this segment with state lattice
        optimized_path = []
        segment_length = 8  # Process in smaller sub-segments for better optimization
        
        i = 0
        while i < len(segment_to_optimize):
            segment_end = min(i + segment_length, len(segment_to_optimize) - 1)
            
            start_world = self.state_lattice.grid_to_world(segment_to_optimize[i])
            end_world = self.state_lattice.grid_to_world(segment_to_optimize[segment_end])
            
            optimized_segment = self.state_lattice.optimize_segment(start_world, end_world)
            
            if i == 0:
                optimized_path.extend(optimized_segment)
            else:
                optimized_path.extend(optimized_segment[1:])  # Skip first point to avoid duplication
            
            i = segment_end
            if i >= len(segment_to_optimize) - 1:
                break
        
        return optimized_path

class DynamicPathVisualizer:
    def __init__(self, costmap, start, goal, planner, zoom_radius=75):
        self.costmap = costmap
        self.start = start
        self.goal = goal
        self.planner = planner
        self.zoom_radius = zoom_radius  # Radius for zoomed view
        
        # Current state
        self.current_pos = [float(start[0]), float(start[1])]
        self.current_path_index = 0  # Index in the global Dijkstra path
        self.step_count = 0
        
        # Visualization setup
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        self.setup_plot()
        
        # Animation data storage
        self.frames_data = []
        
    def setup_plot(self):
        """Setup the initial plot"""
        self.ax.set_title('Dynamic Hybrid Path Planning - Zoomed View Following Robot', 
                         fontsize=16, fontweight='bold')
        self.ax.set_xlabel('Grid X (columns)', fontsize=12)
        self.ax.set_ylabel('Grid Y (rows)', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        
    def get_zoom_bounds(self, center_pos):
        """Calculate bounds for zoomed view around robot"""
        center_x, center_y = int(center_pos[0]), int(center_pos[1])
        
        x_min = max(0, center_x - self.zoom_radius)
        x_max = min(self.costmap.shape[0], center_x + self.zoom_radius)
        y_min = max(0, center_y - self.zoom_radius)
        y_max = min(self.costmap.shape[1], center_y + self.zoom_radius)
        
        return x_min, x_max, y_min, y_max
        
    def update_visualization(self, frame_num):
        """Update function for animation with zoomed view"""
        if frame_num >= len(self.frames_data):
            return []
        
        frame_data = self.frames_data[frame_num]
        current_pos = frame_data['current_pos']
        
        # Clear the plot
        self.ax.clear()
        
        # Get zoom bounds around robot
        x_min, x_max, y_min, y_max = self.get_zoom_bounds(current_pos)
        
        # Show zoomed costmap
        zoomed_costmap = self.costmap[x_min:x_max, y_min:y_max]
        self.ax.imshow(zoomed_costmap, cmap='Blues', alpha=0.6, origin='upper',
                      extent=[y_min, y_max, x_max, x_min])
        
        # Plot start and goal if they're in view
        if x_min <= self.start[0] <= x_max and y_min <= self.start[1] <= y_max:
            self.ax.plot(self.start[1], self.start[0], 'go', markersize=12, 
                        label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
        
        if x_min <= self.goal[0] <= x_max and y_min <= self.goal[1] <= y_max:
            self.ax.plot(self.goal[1], self.goal[0], 'ro', markersize=12, 
                        label='Goal', markeredgecolor='darkred', markeredgewidth=2)
        
        # Plot global Dijkstra path in view (faded)
        if frame_data['global_dijkstra_path'] and len(frame_data['global_dijkstra_path']) > 1:
            path_in_view = []
            for point in frame_data['global_dijkstra_path']:
                if x_min <= point[0] <= x_max and y_min <= point[1] <= y_max:
                    path_in_view.append(point)
            
            if len(path_in_view) > 1:
                path_array = np.array(path_in_view)
                self.ax.plot(path_array[:, 1], path_array[:, 0], 'r--', 
                            linewidth=2, alpha=0.5, label='Global Dijkstra Path')
        
        # Highlight current Dijkstra segment in view
        if frame_data['current_segment'] and len(frame_data['current_segment']) > 1:
            segment_in_view = []
            for point in frame_data['current_segment']:
                if x_min <= point[0] <= x_max and y_min <= point[1] <= y_max:
                    segment_in_view.append(point)
            
            if len(segment_in_view) > 1:
                segment_array = np.array(segment_in_view)
                self.ax.plot(segment_array[:, 1], segment_array[:, 0], 'r-', 
                            linewidth=4, alpha=0.8, label='Current Dijkstra Segment', 
                            marker='s', markersize=8, markerfacecolor='red', markeredgecolor='darkred')
        
        # Plot state lattice optimized path in view (MAIN FOCUS)
        if frame_data['optimized_segment'] and len(frame_data['optimized_segment']) > 1:
            opt_grid_in_view = []
            for world_point in frame_data['optimized_segment']:
                grid_row = int(world_point[1] / self.planner.resolution)
                grid_col = int(world_point[0] / self.planner.resolution)
                grid_row = max(0, min(grid_row, self.costmap.shape[0] - 1))
                grid_col = max(0, min(grid_col, self.costmap.shape[1] - 1))
                
                if x_min <= grid_row <= x_max and y_min <= grid_col <= y_max:
                    opt_grid_in_view.append([grid_col, grid_row])
            
            if len(opt_grid_in_view) > 1:
                opt_array = np.array(opt_grid_in_view)
                self.ax.plot(opt_array[:, 0], opt_array[:, 1], 'b-', 
                            linewidth=6, alpha=0.9, label='State Lattice Optimized', 
                            marker='o', markersize=6, markerfacecolor='blue', markeredgecolor='darkblue')
        
        # Plot vehicle trajectory history in view
        if len(frame_data['vehicle_history']) > 1:
            history_in_view = []
            for point in frame_data['vehicle_history']:
                if x_min <= point[0] <= x_max and y_min <= point[1] <= y_max:
                    history_in_view.append(point)
            
            if len(history_in_view) > 1:
                history_array = np.array(history_in_view)
                self.ax.plot(history_array[:, 1], history_array[:, 0], 'g-', 
                            linewidth=3, alpha=0.7, label='Vehicle Trajectory')
        
        # Plot current robot position (CENTERED)
        robot_circle = Circle((current_pos[1], current_pos[0]), 4, 
                             color='orange', alpha=0.9, zorder=10, 
                             edgecolor='black', linewidth=2)
        self.ax.add_patch(robot_circle)
        
        # # Add direction arrow
        # if frame_data['optimized_segment'] and len(frame_data['optimized_segment']) > 5:
        #     next_world = frame_data['optimized_segment'][min(5, len(frame_data['optimized_segment'])-1)]
        #     next_grid = [next_world[1] / self.planner.resolution, 
        #                 next_world[0] / self.planner.resolution]
            
        #     dx = next_grid[0] - current_pos[1]
        #     dy = next_grid[1] - current_pos[0]
        #     if abs(dx) > 0.1 or abs(dy) > 0.1:
        #         arrow = Arrow(current_pos[1], current_pos[0], dx*0.4, dy*0.4, 
        #                      width=3, color='red', alpha=0.8, zorder=11)
        #         self.ax.add_patch(arrow)
        
        # Set zoom limits to follow robot
        self.ax.set_xlim(y_min, y_max)
        self.ax.set_ylim(x_max, x_min)  # Inverted for image coordinates
        
        # Update legend and title
        self.ax.legend(loc='upper right', fontsize=10)
        self.ax.set_title(f'Step {frame_data["step"]} - Zoomed View (Radius: {self.zoom_radius})\n'
                         f'Distance to Goal: {frame_data["distance_to_goal"]:.1f} - '
                         f'Robot Position: [{current_pos[0]:.1f}, {current_pos[1]:.1f}]', 
                         fontsize=12, fontweight='bold')
        
        self.ax.grid(True, alpha=0.3)
        
        return []
    
    def simulate_dynamic_planning(self, max_steps=2000):
        """Simulate the dynamic planning process"""
        print("Starting dynamic planning simulation...")
        
        vehicle_history = [list(self.current_pos)]
        replan_frequency = max_steps  # Replan every 30 steps
        
        for step in range(max_steps):
            # Calculate distance to goal
            distance_to_goal = math.sqrt(
                (self.goal[0] - self.current_pos[0])**2 + 
                (self.goal[1] - self.current_pos[1])**2
            )
            
            if step % 20 == 0:  # Print less frequently for zoomed view
                print(f"Step {step}: Position [{self.current_pos[0]:.1f}, {self.current_pos[1]:.1f}], Distance: {distance_to_goal:.1f}")
            
            # Check if reached goal
            if distance_to_goal < 5:
                print("Goal reached!")
                break
            
            # Replan global path periodically or if no path exists
            need_global_replan = (
                self.planner.global_dijkstra_path is None or 
                step % replan_frequency == 0 or
                self.current_path_index >= len(self.planner.global_dijkstra_path) - 5
            )
            
            if need_global_replan:
                self.planner.global_dijkstra_path = self.planner.plan_global_path(
                    self.current_pos, self.goal
                )
                self.current_path_index = 0  # Reset to start of new path
            
            # Always optimize the next segment from current position
            if self.planner.global_dijkstra_path:
                # Find closest point on global path to current position
                min_dist = float('inf')
                closest_index = 0
                for i, waypoint in enumerate(self.planner.global_dijkstra_path):
                    dist = math.sqrt((waypoint[0] - self.current_pos[0])**2 + 
                                   (waypoint[1] - self.current_pos[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_index = i
                
                self.current_path_index = closest_index
                
                # Optimize next segment
                self.planner.current_optimized_segment = self.planner.optimize_next_segment(
                    self.planner.global_dijkstra_path, self.current_path_index
                )
                
                # Get current segment for visualization
                end_index = min(self.current_path_index + self.planner.optimization_lookahead, 
                              len(self.planner.global_dijkstra_path) - 1)
                current_segment = self.planner.global_dijkstra_path[self.current_path_index:end_index + 1]
            else:
                current_segment = None
                self.planner.current_optimized_segment = None
            
            # Store frame data (every few steps to reduce file size)
            if step % 2 == 0:  # Store every 2nd frame for smoother zoomed animation
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
            
            # Move vehicle along optimized path
            if self.planner.current_optimized_segment and len(self.planner.current_optimized_segment) > 0:
                # Move towards next waypoint in optimized segment
                lookahead_index = min(3, len(self.planner.current_optimized_segment) - 1)
                target_world = self.planner.current_optimized_segment[lookahead_index]
                target_grid = self.planner.state_lattice.world_to_grid(target_world)
                
                # Calculate movement direction
                dx = target_grid[0] - self.current_pos[0]
                dy = target_grid[1] - self.current_pos[1]
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance > 0.1:
                    # Normalize and scale by vehicle speed
                    move_x = (dx / distance) * self.planner.vehicle_speed
                    move_y = (dy / distance) * self.planner.vehicle_speed
                    
                    # Update position
                    self.current_pos[0] += move_x
                    self.current_pos[1] += move_y
                    
                    # Keep within bounds
                    self.current_pos[0] = max(0, min(self.current_pos[0], self.costmap.shape[0] - 1))
                    self.current_pos[1] = max(0, min(self.current_pos[1], self.costmap.shape[1] - 1))
                
                vehicle_history.append(list(self.current_pos))
        
        print(f"Simulation completed with {len(self.frames_data)} frames")
        
    def create_animation(self, filename='zoomed_state_lattice_planning.gif', fps=10):
        """Create and save zoomed animation"""
        if len(self.frames_data) == 0:
            print("No frames to animate!")
            return None
            
        print(f"Creating zoomed animation with {len(self.frames_data)} frames...")
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, self.update_visualization, 
            frames=len(self.frames_data),
            interval=1000//fps,
            blit=False,
            repeat=True
        )
        
        # Save as GIF
        print(f"Saving zoomed animation to {filename}...")
        try:
            anim.save(filename, writer='pillow', fps=fps, dpi=100)
            print(f"Zoomed animation saved successfully!")
        except Exception as e:
            print(f"Error saving animation: {e}")
        
        return anim

def generate_k_alternative_paths(costmap, start, goal, k=3, alpha=1.0, beta=1.0):
    """Generate K alternative paths using Dijkstra with same alpha/beta parameters"""
    print(f"=== GENERATING {k} ALTERNATIVE DIJKSTRA PATHS ===")
    print(f"Using parameters: α={alpha}, β={beta}")
    
    # Create a copy of the original costmap that we can modify
    original_costmap = costmap.copy()
    
    # Create figure for visualization
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(costmap, cmap='Blues', alpha=0.6, origin='upper')
    
    # Plot start and goal
    ax.plot(start[1], start[0], 'go', markersize=12, label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
    ax.plot(goal[1], goal[0], 'ro', markersize=12, label='Goal', markeredgecolor='darkred', markeredgewidth=2)
    
    paths = []
    path_colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    
    # Find first path with original costmap
    print(f"Finding path 1 of {k}...")
    grid = Grid(original_costmap, start)
    dijkstra_solver = Dijkstra(grid, start, goal)
    dijkstra_solver.find_path(alpha, beta)
    path1 = dijkstra_solver.backtrack_path()
    
    if not path1 or len(path1) < 2:
        print("Failed to find first path!")
        return None
    
    # Calculate first path cost for reference
    path1_cost = calculate_path_cost(original_costmap, path1, alpha, beta)
    
    paths.append({
        "path": path1,
        "label": f"Path 1 (Cost: {path1_cost:.1f})",
        "color": path_colors[0],
        "cost": path1_cost
    })
    
    # Plot first path
    path_array = np.array(path1)
    ax.plot(path_array[:, 1], path_array[:, 0], 
           color=path_colors[0], linewidth=3, alpha=0.8,
           label=f"Path 1 (Cost: {path1_cost:.1f})", marker='o', markersize=4)
    
    # Create a working copy of the costmap to modify for subsequent paths
    working_costmap = original_costmap.copy()
    
    # For subsequent paths, penalize cells along previous paths
    for i in range(1, k):
        print(f"Finding path {i+1} of {k}...")
        
        # Modify costmap to discourage using the same path
        # We'll apply an increasing penalty to cells that were part of previous paths
        for j, path_data in enumerate(paths):
            path = path_data["path"]
            
            # Calculate path width based on which alternative we're searching for
            path_width = 3 if i <= 2 else 2
            
            # Apply penalty to this path's cells and nearby cells
            for point in path:
                r, c = point
                r_min = max(0, r - path_width)
                r_max = min(working_costmap.shape[0] - 1, r + path_width)
                c_min = max(0, c - path_width)
                c_max = min(working_costmap.shape[1] - 1, c + path_width)
                
                # Penalty decreases with distance from path
                for dr in range(r_min, r_max + 1):
                    for dc in range(c_min, c_max + 1):
                        dist = abs(dr - r) + abs(dc - c)
                        if dist == 0:  # Exact path point
                            working_costmap[dr, dc] += 50.0 * (1 + 0.5 * j)
                        else:  # Near path point
                            # Penalty decreases with distance from path
                            penalty = 50.0 * (1 + 0.5 * j) / (1 + dist)
                            working_costmap[dr, dc] += penalty
        
        # Find a new path with the modified costmap
        grid = Grid(working_costmap, start)
        dijkstra_solver = Dijkstra(grid, start, goal)
        dijkstra_solver.find_path(alpha, beta)
        new_path = dijkstra_solver.backtrack_path()
        
        if not new_path or len(new_path) < 2:
            print(f"Failed to find path {i+1}!")
            continue
        
        # Calculate path cost based on ORIGINAL costmap values
        new_path_cost = calculate_path_cost(original_costmap, new_path, alpha, beta)
        
        # Check path uniqueness - must differ from previous paths by at least 15% of points
        is_unique = True
        for prev_path_data in paths:
            prev_path = prev_path_data["path"]
            
            # Convert paths to sets of points for comparison
            new_points = set(tuple(point) for point in new_path)
            prev_points = set(tuple(point) for point in prev_path)
            
            # Calculate overlap
            common_points = new_points.intersection(prev_points)
            overlap_ratio = len(common_points) / min(len(new_points), len(prev_points))
            
            if overlap_ratio > 0.85:  # More than 85% overlap
                is_unique = False
                print(f"  Path {i+1} is too similar to a previous path ({overlap_ratio:.2%} overlap)")
                break
        
        if not is_unique:
            # Try to increase the penalty and find another path
            continue
        
        # Add this unique path to our collection
        paths.append({
            "path": new_path,
            "label": f"Path {i+1} (Cost: {new_path_cost:.1f})",
            "color": path_colors[i % len(path_colors)],
            "cost": new_path_cost
        })
        
        # Plot this path
        path_array = np.array(new_path)
        ax.plot(path_array[:, 1], path_array[:, 0], 
               color=path_colors[i % len(path_colors)], linewidth=3, alpha=0.8,
               label=f"Path {i+1} (Cost: {new_path_cost:.1f})", marker='o', markersize=4)
    
    # Add legend and title
    ax.legend(loc='upper right', fontsize=12)
    ax.set_title(f'{len(paths)} Alternative Dijkstra Paths (α={alpha}, β={beta})', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Show the plot
    plt.tight_layout()
    plt.savefig(f'alternative_paths_alpha{alpha}_beta{beta}.png')
    plt.show()
    
    # Ask user to select a path
    while True:
        print("\nSelect a path to use for dynamic planning:")
        for i, path_data in enumerate(paths):
            print(f"{i+1}: {path_data['label']}")
        
        try:
            choice = int(input(f"Enter your choice (1-{len(paths)}): "))
            if 1 <= choice <= len(paths):
                selected_path = paths[choice-1]
                print(f"\nYou selected: {selected_path['label']}")
                return selected_path["path"], alpha, beta
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(paths)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def calculate_path_cost(costmap, path, alpha, beta):
    """Calculate the actual path cost using the original costmap values"""
    if not path or len(path) < 2:
        return float('inf')
    
    total_cost = 0
    
    for i in range(1, len(path)):
        prev_point = path[i-1]
        curr_point = path[i]
        
        # Calculate distance cost component
        distance = math.sqrt((curr_point[0] - prev_point[0])**2 + 
                           (curr_point[1] - prev_point[1])**2)
        
        # Calculate costmap value component (use the current point's cost)
        costmap_value = costmap[curr_point[0], curr_point[1]]
        
        # Combine with alpha and beta
        segment_cost = alpha * distance + beta * costmap_value
        total_cost += segment_cost
    
    return total_cost

def add_circular_obstacle(self, center, radius):
    """Add a circular obstacle with high cost to the costmap"""
    # Ensure center is valid
    center_x = max(0, min(center[0], self.costmap.shape[0] - 1))
    center_y = max(0, min(center[1], self.costmap.shape[1] - 1))
    
    # Calculate bounds of the affected region
    min_x = max(0, center_x - radius)
    max_x = min(self.costmap.shape[0] - 1, center_x + radius)
    min_y = max(0, center_y - radius)
    max_y = min(self.costmap.shape[1] - 1, center_y + radius)
    
    # Update costmap with high cost in circular region
    for x in range(int(min_x), int(max_x) + 1):
        for y in range(int(min_y), int(max_y) + 1):
            # Calculate distance to center
            dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            if dist <= radius:
                # Set high cost (1000) for obstacle
                self.costmap[x, y] = 1000
                self.planner.costmap[x, y] = 1000
    print(f"Added circular obstacle at ({center_x}, {center_y}) with radius {radius}")

def main():
    print("=== DYNAMIC HYBRID PATH PLANNING WITH K-ALTERNATIVE PATHS ===")
    
    # Load costmap
    print("Loading costmap...")
    try:
        grid_data = np.load('/home/kasi/Desktop/PythonRobotics/PathPlanning/StateLatticePlanner/cavasos_costmap_final.npy', allow_pickle=True)
    except FileNotFoundError:
        print("Costmap file not found! Creating a simple test costmap...")
        grid_data = np.ones((600, 600)) * 10
        grid_data[200:400, 200:220] = 100
        grid_data[300:320, 100:500] = 100
    
    # Define start and goal
    start = [1211, 281]
    goal = [1258, 1657]
    # Ensure start and goal are within bounds
    start[0] = max(0, min(start[0], grid_data.shape[0] - 1))
    start[1] = max(0, min(start[1], grid_data.shape[1] - 1))
    goal[0] = max(0, min(goal[0], grid_data.shape[0] - 1))
    goal[1] = max(0, min(goal[1], grid_data.shape[1] - 1))
    
    print(f"Costmap shape: {grid_data.shape}")
    print(f"Planning from {start} to {goal}")
    
    # Set parameters for path planning
    alpha = 1.0
    beta = 1.0
    
    # Generate K alternative paths and let user select one
    selected_path, alpha, beta = generate_k_alternative_paths(
        grid_data, start, goal, k=5, alpha=alpha, beta=beta)
    
    if selected_path is None:
        print("Path selection failed. Exiting.")
        return
    
    # Create dynamic planner with the parameters
    print(f"\nCreating dynamic planner with α={alpha}, β={beta}")
    planner = DynamicHybridPlanner(grid_data, resolution=3, alpha=alpha, beta=beta)
    
    # Set the selected path as the global path
    planner.global_dijkstra_path = selected_path
    
    # Create zoomed visualizer
    print(f"Creating zoomed view that follows the robot")
    visualizer = DynamicPathVisualizer(grid_data, start, goal, planner, zoom_radius=75)
    
    # Run simulation
    visualizer.simulate_dynamic_planning(max_steps=1500)
    
    # Create animation
    if len(visualizer.frames_data) > 0:
        animation_obj = visualizer.create_animation(
            filename=f'zoomed_planning_path{len(visualizer.frames_data)}.gif', 
            fps=12
        )
        
        # Show final plot
        plt.show()
    else:
        print("No frames generated for animation!")
    
    print("Zoomed dynamic planning simulation completed!")

if __name__ == "__main__":
    main()