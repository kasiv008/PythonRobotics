"""
State lattice planner with model predictive trajectory generator
Modified to include costmap integration for start-to-goal planning
"""
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
import math
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import ModelPredictiveTrajectoryGenerator.trajectory_generator as planner
import ModelPredictiveTrajectoryGenerator.motion_model as motion_model

TABLE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/lookup_table.csv"
show_animation = True

# Load your costmap
COSTMAP_PATH = "/home/level5_kasi/Downloads/cavasos_costmap_final.npy"  # Replace with your actual file path
costmap = np.load(COSTMAP_PATH)

# Costmap parameters (adjust these to match your costmap)
MAP_RESOLUTION = 3  # meters per pixel
MAP_ORIGIN = [0.0, 0.0]  # world coordinates of costmap[0,0]

def world_to_map(x, y, map_resolution, map_origin):
    """Convert world coordinates to map indices"""
    ix = int((x - map_origin[0]) / map_resolution)
    iy = int((y - map_origin[1]) / map_resolution)
    return ix, iy

def calculate_trajectory_cost(x_traj, y_traj, costmap, map_resolution, map_origin):
    """Calculate the total cost of a trajectory using the costmap"""
    total_cost = 0
    valid_points = 0
    
    for x, y in zip(x_traj, y_traj):
        ix, iy = world_to_map(x, y, map_resolution, map_origin)
        
        # Check if point is within costmap bounds
        if 0 <= ix < costmap.shape[1] and 0 <= iy < costmap.shape[0]:
            total_cost += costmap[iy, ix]
            valid_points += 1
        else:
            # Heavy penalty for out-of-bounds points
            total_cost += 1000
            valid_points += 1
    
    # Return average cost per point
    return total_cost / valid_points if valid_points > 0 else float('inf')

def search_nearest_one_from_lookup_table(t_x, t_y, t_yaw, lookup_table):
    mind = float("inf")
    minid = -1

    for (i, table) in enumerate(lookup_table):
        dx = t_x - table[0]
        dy = t_y - table[1]
        dyaw = t_yaw - table[2]
        d = math.sqrt(dx ** 2 + dy ** 2 + dyaw ** 2)
        if d <= mind:
            minid = i
            mind = d

    return lookup_table[minid]

def get_lookup_table(table_path):
    return np.loadtxt(table_path, delimiter=',', skiprows=1)

def generate_path_with_costmap(target_states, k0, costmap, map_resolution, map_origin):
    """Generate paths and evaluate them using costmap"""
    lookup_table = get_lookup_table(TABLE_PATH)
    candidates = []

    for state in target_states:
        bestp = search_nearest_one_from_lookup_table(
            state[0], state[1], state[2], lookup_table)

        target = motion_model.State(x=state[0], y=state[1], yaw=state[2])
        init_p = np.array(
            [np.hypot(state[0], state[1]), bestp[4], bestp[5]]).reshape(3, 1)

        x, y, yaw, p = planner.optimize_trajectory(target, k0, init_p)

        if x is not None:
            # Calculate cost using costmap
            cost = calculate_trajectory_cost(x, y, costmap, map_resolution, map_origin)
            
            candidates.append({
                'trajectory': [x, y, yaw],
                'params': [float(p[0, 0]), float(p[1, 0]), float(p[2, 0])],
                'cost': cost,
                'end_state': [x[-1], y[-1], yaw[-1]]
            })

    # Sort by cost and return best candidates
    candidates.sort(key=lambda x: x['cost'])
    print(f"Generated {len(candidates)} valid paths")
    return candidates

def plan_start_to_goal(start_pos, goal_pos, costmap, map_resolution, map_origin):
    """
    Plan path from start to goal using state lattice approach
    
    Parameters:
    start_pos: [x, y, yaw] - starting position and orientation
    goal_pos: [x, y, yaw] - goal position and orientation
    """
    
    # Calculate relative goal position from start
    dx = goal_pos[0] - start_pos[0]
    dy = goal_pos[1] - start_pos[1]
    distance = math.sqrt(dx**2 + dy**2)
    
    # Generate target states around the goal
    k0 = 0.0  # Initial curvature
    
    # Method 1: Direct goal targeting with some variations
    target_states = []
    
    # Add the exact goal
    target_states.append([dx, dy, goal_pos[2] - start_pos[2]])
    
    # Add variations around the goal for robustness
    for angle_offset in [-0.2, -0.1, 0.1, 0.2]:  # radians
        for dist_offset in [-1.0, -0.5, 0.5, 1.0]:  # meters
            if distance + dist_offset > 0:
                scale = (distance + dist_offset) / distance
                new_dx = dx * scale
                new_dy = dy * scale
                new_yaw = goal_pos[2] - start_pos[2] + angle_offset
                target_states.append([new_dx, new_dy, new_yaw])
    
    # Generate and evaluate paths
    candidates = generate_path_with_costmap(target_states, k0, costmap, map_resolution, map_origin)
    
    if not candidates:
        print("No valid path found!")
        return None
    
    # Transform best path back to world coordinates
    best_candidate = candidates[0]
    x_rel, y_rel, yaw_rel = best_candidate['trajectory']
    
    # Transform relative coordinates to world coordinates
    cos_start = math.cos(start_pos[2])
    sin_start = math.sin(start_pos[2])
    
    x_world = []
    y_world = []
    yaw_world = []
    
    for i in range(len(x_rel)):
        # Rotate and translate
        x_w = start_pos[0] + x_rel[i] * cos_start - y_rel[i] * sin_start
        y_w = start_pos[1] + x_rel[i] * sin_start + y_rel[i] * cos_start
        yaw_w = start_pos[2] + yaw_rel[i]
        
        x_world.append(x_w)
        y_world.append(y_w)
        yaw_world.append(yaw_w)
    
    return {
        'path': [x_world, y_world, yaw_world],
        'cost': best_candidate['cost'],
        'all_candidates': candidates
    }

def visualize_planning_result(start_pos, goal_pos, result, costmap):
    """Visualize the planning result"""
    if show_animation:
        plt.figure(figsize=(12, 8))
        
        # Show costmap as background
        plt.imshow(costmap, cmap='hot', alpha=0.6, origin='lower')
        plt.colorbar(label='Cost')
        
        # Convert world coordinates to map coordinates for visualization
        start_map = world_to_map(start_pos[0], start_pos[1], MAP_RESOLUTION, MAP_ORIGIN)
        goal_map = world_to_map(goal_pos[0], goal_pos[1], MAP_RESOLUTION, MAP_ORIGIN)
        
        # Plot start and goal
        plt.plot(start_map[0], start_map[1], 'go', markersize=10, label='Start')
        plt.plot(goal_map[0], goal_map[1], 'ro', markersize=10, label='Goal')
        
        if result:
            # Plot best path
            x_world, y_world, yaw_world = result['path']
            x_map = [(x - MAP_ORIGIN[0]) / MAP_RESOLUTION for x in x_world]
            y_map = [(y - MAP_ORIGIN[1]) / MAP_RESOLUTION for y in y_world]
            plt.plot(x_map, y_map, 'b-', linewidth=3, label=f'Best Path (Cost: {result["cost"]:.2f})')
            
            # Plot alternative paths with lower alpha
            for i, candidate in enumerate(result['all_candidates'][1:6]):  # Show top 5 alternatives
                x_rel, y_rel, yaw_rel = candidate['trajectory']
                # Transform to world then to map coordinates
                x_world_alt = []
                y_world_alt = []
                cos_start = math.cos(start_pos[2])
                sin_start = math.sin(start_pos[2])
                
                for j in range(len(x_rel)):
                    x_w = start_pos[0] + x_rel[j] * cos_start - y_rel[j] * sin_start
                    y_w = start_pos[1] + x_rel[j] * sin_start + y_rel[j] * cos_start
                    x_world_alt.append(x_w)
                    y_world_alt.append(y_w)
                
                x_map_alt = [(x - MAP_ORIGIN[0]) / MAP_RESOLUTION for x in x_world_alt]
                y_map_alt = [(y - MAP_ORIGIN[1]) / MAP_RESOLUTION for y in y_world_alt]
                plt.plot(x_map_alt, y_map_alt, 'c--', alpha=0.5, linewidth=1)
        
        plt.legend()
        plt.title('State Lattice Path Planning with Costmap')
        plt.xlabel('Map X (pixels)')
        plt.ylabel('Map Y (pixels)')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()

def main():
    """Main function demonstrating start-to-goal planning"""
    planner.show_animation = show_animation
    
    # Define start and goal positions [x, y, yaw]
    start_position = [200.0, 100.0, 0.0]  # Adjust to your coordinate system
    goal_position = [300.0, 250.0, math.pi/4]  # Adjust to your coordinate system
    
    print(f"Planning path from {start_position} to {goal_position}")
    print(f"Costmap shape: {costmap.shape}")
    print(f"Map resolution: {MAP_RESOLUTION} m/pixel")
    
    # Plan the path
    result = plan_start_to_goal(start_position, goal_position, costmap, MAP_RESOLUTION, MAP_ORIGIN)
    
    if result:
        print(f"Path found with cost: {result['cost']:.2f}")
        print(f"Path length: {len(result['path'][0])} points")
        
        # Visualize result
        visualize_planning_result(start_position, goal_position, result, costmap)
    else:
        print("No path found!")

if __name__ == '__main__':
    main()
