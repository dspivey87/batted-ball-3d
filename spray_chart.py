import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from typing import Tuple, List, Dict, Optional
import colorsys
import trimesh
import os
from pathlib import Path

# Constants
G = 9.81  # gravity (m/s^2)
MPS_TO_MPH = 2.23694
M_TO_FT = 3.28084

@st.cache_data
def load_data(filepath: str) -> pd.DataFrame:
    """Load and cache the data."""
    return pd.read_csv(filepath)

@st.cache_resource
def load_stadium_model(filepath: str) -> Dict:
    """Load and cache the stadium 3D model."""
    try:
        # Load the stadium model using trimesh
        stadium_mesh = trimesh.load(filepath)
        stadium_data = {}
        
        # Process each mesh geometry in the model
        for name, geometry in stadium_mesh.geometry.items():
            vertices = geometry.vertices
            faces = geometry.faces
            
            # Scale vertices to match our coordinate system (feet)
            vertices_ft = vertices * M_TO_FT
            
            # Center the stadium at home plate
            # Adjust these offsets based on your specific model
            x_offset = 0
            y_offset = 0
            z_offset = 0
            
            # Store the processed mesh
            stadium_data[name] = {
                'vertices': vertices_ft,
                'faces': faces,
                'x_offset': x_offset,
                'y_offset': y_offset,
                'z_offset': z_offset
            }
        
        return stadium_data
    except Exception as e:
        st.error(f"Error loading stadium model: {str(e)}")
        return {}

def create_stadium_traces(stadium_data: Dict) -> List[go.Mesh3d]:
    """Convert stadium data to Plotly Mesh3d traces."""
    traces = []
    
    # Define stadium component colors
    color_map = {
        'field': 'green',
        'seats': 'blue',
        'walls': 'brown',
        'stands': 'gray',
        'roof': 'silver',
        'default': 'lightblue'
    }
    
    # Process each mesh component
    for name, mesh in stadium_data.items():
        # Determine color based on component name
        color = 'lightblue'
        for key, col in color_map.items():
            if key in name.lower():
                color = col
                break
        
        # Create mesh trace
        trace = go.Mesh3d(
            x=mesh['vertices'][:, 0] + mesh['x_offset'],
            y=mesh['vertices'][:, 1] + mesh['y_offset'],
            z=mesh['vertices'][:, 2] + mesh['z_offset'],
            i=mesh['faces'][:, 0],
            j=mesh['faces'][:, 1],
            k=mesh['faces'][:, 2],
            color=color,
            opacity=0.7,
            name=f"Stadium: {name}",
            showscale=False,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2)
        )
        traces.append(trace)
    
    return traces

@st.cache_data
def generate_color_palette(n: int) -> List[str]:
    """Generate n distinct colors using HSV color space."""
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors

@st.cache_data
def calculate_trajectory(
    exit_velocity: float, 
    launch_angle: float, 
    spray_angle: float,
    num_points: int = 50
) -> Dict[str, np.ndarray]:
    """Calculate ball trajectory with optimized computation."""
    # Convert to SI units
    v0 = exit_velocity / MPS_TO_MPH
    la_rad = np.radians(launch_angle)
    sa_rad = np.radians(spray_angle)
    
    # Velocity components
    vx = v0 * np.cos(la_rad) * np.cos(sa_rad)
    vy = v0 * np.cos(la_rad) * np.sin(sa_rad)
    vz = v0 * np.sin(la_rad)
    
    # Time of flight
    t_flight = 2 * vz / G
    t = np.linspace(0, t_flight, num_points)
    
    # Positions in feet
    x = vx * t * M_TO_FT
    y = vy * t * M_TO_FT
    z = np.maximum(0, (vz * t - 0.5 * G * t**2) * M_TO_FT)
    
    # Calculate additional metrics
    max_height = (vz**2 / (2 * G)) * M_TO_FT
    distance = np.sqrt(x[-1]**2 + y[-1]**2)
    hang_time = t_flight
    
    return {
        'x': x,
        'y': y,
        'z': z,
        'max_height': max_height,
        'distance': distance,
        'hang_time': hang_time
    }

def create_simplified_stadium() -> List[go.Mesh3d]:
    """Create a simplified stadium if no 3D model is available."""
    traces = []
    
    # Field (infield + outfield)
    theta = np.linspace(-45, 45, 50)
    outfield_r = 350
    x_field = np.concatenate([[0], outfield_r * np.cos(np.radians(theta))])
    y_field = np.concatenate([[0], outfield_r * np.sin(np.radians(theta))])
    z_field = np.zeros_like(x_field)
    
    # Triangulate the field
    triangles = []
    for i in range(1, len(x_field) - 1):
        triangles.append([0, i, i+1])
    
    # Create field mesh
    traces.append(go.Mesh3d(
        x=x_field,
        y=y_field,
        z=z_field,
        i=[t[0] for t in triangles],
        j=[t[1] for t in triangles],
        k=[t[2] for t in triangles],
        color='green',
        opacity=0.6,
        name='Field',
        showscale=False
    ))
    
    # Outfield wall
    wall_height = 8
    x_wall = outfield_r * np.cos(np.radians(theta))
    y_wall = outfield_r * np.sin(np.radians(theta))
    z_wall_bottom = np.zeros_like(x_wall)
    z_wall_top = np.ones_like(x_wall) * wall_height
    
    x_wall_all = np.concatenate([x_wall, x_wall])
    y_wall_all = np.concatenate([y_wall, y_wall])
    z_wall_all = np.concatenate([z_wall_bottom, z_wall_top])
    
    wall_triangles = []
    n = len(x_wall)
    for i in range(n-1):
        wall_triangles.append([i, i+1, i+n])
        wall_triangles.append([i+n, i+1, i+n+1])
    
    traces.append(go.Mesh3d(
        x=x_wall_all,
        y=y_wall_all,
        z=z_wall_all,
        i=[t[0] for t in wall_triangles],
        j=[t[1] for t in wall_triangles],
        k=[t[2] for t in wall_triangles],
        color='blue',
        opacity=0.5,
        name='Outfield Wall',
        showscale=False
    ))
    
    # Create bases
    bases = [
        {'name': 'Home', 'x': 0, 'y': 0, 'color': 'white'},
        {'name': 'First', 'x': 90, 'y': 90, 'color': 'white'},
        {'name': 'Second', 'x': 0, 'y': 127, 'color': 'white'},
        {'name': 'Third', 'x': -90, 'y': 90, 'color': 'white'},
        {'name': 'Pitcher', 'x': 0, 'y': 60, 'color': 'brown'},
    ]
    
    for base in bases:
        traces.append(go.Scatter3d(
            x=[base['x']],
            y=[base['y']],
            z=[0.1],  # Slightly above ground
            mode='markers',
            marker=dict(
                size=8,
                color=base['color'],
                symbol='square',
                line=dict(color='black', width=1)
            ),
            name=f"{base['name']} Base",
            showlegend=False
        ))
    
    return traces

def create_field_overlay() -> List[go.Scatter3d]:
    """Create baseball field reference lines."""
    traces = []
    
    # Foul lines
    foul_line_dist = 350
    traces.append(go.Scatter3d(
        x=[0, foul_line_dist],
        y=[0, 0],
        z=[0, 0],
        mode='lines',
        line=dict(color='white', width=2, dash='dash'),
        name='Center Field',
        showlegend=False
    ))
    
    # Left field line
    traces.append(go.Scatter3d(
        x=[0, foul_line_dist * np.cos(np.radians(45))],
        y=[0, foul_line_dist * np.sin(np.radians(45))],
        z=[0, 0],
        mode='lines',
        line=dict(color='white', width=2, dash='dash'),
        name='Left Field Line',
        showlegend=False
    ))
    
    # Right field line
    traces.append(go.Scatter3d(
        x=[0, foul_line_dist * np.cos(np.radians(-45))],
        y=[0, foul_line_dist * np.sin(np.radians(-45))],
        z=[0, 0],
        mode='lines',
        line=dict(color='white', width=2, dash='dash'),
        name='Right Field Line',
        showlegend=False
    ))
    
    # Diamond
    diamond_x = [0, 90, 0, -90, 0]
    diamond_y = [0, 90, 127, 90, 0]
    diamond_z = [0, 0, 0, 0, 0]
    
    traces.append(go.Scatter3d(
        x=diamond_x,
        y=diamond_y,
        z=diamond_z,
        mode='lines',
        line=dict(color='white', width=3),
        name='Diamond',
        showlegend=False
    ))
    
    # Distance markers
    for dist in [100, 200, 300]:
        theta = np.linspace(-45, 45, 20)
        x = dist * np.cos(np.radians(theta))
        y = dist * np.sin(np.radians(theta))
        z = np.zeros_like(x)
        
        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            name=f'{dist}ft',
            showlegend=False
        ))
    
    return traces

def create_trajectory_trace(
    trajectory: Dict[str, np.ndarray],
    color: str,
    name: str,
    show_markers: bool = True
) -> go.Scatter3d:
    """Create a single trajectory trace."""
    return go.Scatter3d(
        x=trajectory['x'],
        y=trajectory['y'],
        z=trajectory['z'],
        mode='lines+markers' if show_markers else 'lines',
        line=dict(color=color, width=3),
        marker=dict(size=2, color=color) if show_markers else None,
        name=name,
        hovertemplate=(
            f"<b>{name}</b><br>" +
            "Distance: %{text}<br>" +
            "Height: %{z:.1f} ft<br>" +
            "<extra></extra>"
        ),
        text=[f"{trajectory['distance']:.0f} ft"] * len(trajectory['x'])
    )

def create_landing_markers(trajectories: List[Dict], colors: List[str], names: List[str]) -> go.Scatter3d:
    """Create landing point markers."""
    x_lands = [traj['x'][-1] for traj in trajectories]
    y_lands = [traj['y'][-1] for traj in trajectories]
    distances = [f"{traj['distance']:.0f} ft" for traj in trajectories]
    
    return go.Scatter3d(
        x=x_lands,
        y=y_lands,
        z=[0] * len(x_lands),
        mode='markers+text',
        marker=dict(
            size=8,
            color=colors,
            symbol='circle',
            line=dict(color='white', width=2)
        ),
        text=distances,
        textposition='top center',
        name='Landing Points',
        showlegend=False
    )

def main():
    st.set_page_config(page_title="3D Spray Chart", layout="wide")
    st.title("⚾ Advanced 3D Stadium Spray Chart")
    
    # Load data
    df = load_data("data/hittrax_batted_balls.csv")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        
        # Player selection
        player = st.selectbox("Select Player", df["player_name"].unique())
        player_df = df[df["player_name"] == player].reset_index(drop=True)
        
        # Stadium options
        st.subheader("Stadium Options")
        stadium_type = st.radio(
            "Stadium Visualization Type",
            ["3D Model", "Simplified", "Field Lines Only", "None"],
            index=1
        )
        
        if stadium_type == "3D Model":
            stadium_path = st.text_input(
                "Stadium Model Path",
                value="assets/Stadium.glb"
            )
        
        # Display options
        st.subheader("Display Options")
        show_animation = st.checkbox("Enable Animation", value=True)
        show_stats = st.checkbox("Show Statistics", value=True)
        
        # Camera angles
        st.subheader("Camera Presets")
        camera_preset = st.radio(
            "Camera Angle",
            ["Behind Home", "Outfield View", "Overhead", "Side View", "Custom"],
            index=0
        )
        
        # Trajectory options
        st.subheader("Trajectory Options")
        num_points = st.slider("Trajectory Points", 20, 100, 50)
        show_all = st.checkbox("Show All Trajectories", value=True)
        
        if not show_all:
            selected_hits = st.multiselect(
                "Select Hits",
                options=list(range(len(player_df))),
                default=list(range(min(5, len(player_df)))),
                format_func=lambda x: f"Hit {x+1}: {player_df.iloc[x]['hit_type']}"
            )
        else:
            selected_hits = list(range(len(player_df)))
    
    # Generate colors
    colors = generate_color_palette(len(selected_hits))
    
    # Calculate trajectories
    trajectories = []
    for idx in selected_hits:
        row = player_df.iloc[idx]
        traj = calculate_trajectory(
            row['exit_velocity'],
            row['launch_angle'],
            row['spray_angle'],
            num_points
        )
        trajectories.append(traj)
    
    # Create plot
    fig = go.Figure()
    
    # Add stadium based on selected type
    if stadium_type == "3D Model":
        if os.path.exists(stadium_path):
            stadium_data = load_stadium_model(stadium_path)
            for trace in create_stadium_traces(stadium_data):
                fig.add_trace(trace)
        else:
            st.warning(f"Stadium model not found at: {stadium_path}")
            for trace in create_simplified_stadium():
                fig.add_trace(trace)
    elif stadium_type == "Simplified":
        for trace in create_simplified_stadium():
            fig.add_trace(trace)
    elif stadium_type == "Field Lines Only":
        for trace in create_field_overlay():
            fig.add_trace(trace)
    
    # Add trajectories
    if show_animation:
        # Create animated version
        frames = []
        for i, (idx, traj, color) in enumerate(zip(selected_hits, trajectories, colors)):
            row = player_df.iloc[idx]
            name = f"Hit {idx+1}: {row['hit_type']}"
            
            # Create frames for this trajectory
            for j in range(2, len(traj['x']) + 1):
                frame_data = []
                
                # Add stadium to each frame if using 3D model
                if stadium_type == "3D Model" and os.path.exists(stadium_path):
                    stadium_data = load_stadium_model(stadium_path)
                    for trace in create_stadium_traces(stadium_data):
                        frame_data.append(trace)
                elif stadium_type == "Simplified":
                    for trace in create_simplified_stadium():
                        frame_data.append(trace)
                elif stadium_type == "Field Lines Only":
                    for trace in create_field_overlay():
                        frame_data.append(trace)
                
                # Add all previous complete trajectories
                for k in range(i):
                    prev_traj = trajectories[k]
                    prev_row = player_df.iloc[selected_hits[k]]
                    prev_name = f"Hit {selected_hits[k]+1}: {prev_row['hit_type']}"
                    frame_data.append(create_trajectory_trace(prev_traj, colors[k], prev_name, False))
                
                # Add current partial trajectory
                partial_traj = {
                    'x': traj['x'][:j],
                    'y': traj['y'][:j],
                    'z': traj['z'][:j],
                    'distance': traj['distance']
                }
                frame_data.append(create_trajectory_trace(partial_traj, color, name, True))
                
                frames.append(go.Frame(data=frame_data, name=f"frame_{i}_{j}"))
        
        fig.frames = frames
        
        # Add play/pause buttons
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': '▶ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 50, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 0}
                        }]
                    },
                    {
                        'label': '⏸ Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }]
        )
    else:
        # Static version
        for idx, traj, color in zip(selected_hits, trajectories, colors):
            row = player_df.iloc[idx]
            name = f"Hit {idx+1}: {row['hit_type']}"
            fig.add_trace(create_trajectory_trace(traj, color, name, False))
    
    # Add landing markers
    if trajectories:
        names = [f"Hit {idx+1}" for idx in selected_hits]
        fig.add_trace(create_landing_markers(trajectories, colors, names))
    
    # Set camera angle based on preset
    if camera_preset == "Behind Home":
        camera = dict(
            eye=dict(x=-1.5, y=-1.5, z=0.8),
            center=dict(x=90, y=90, z=0),
            up=dict(x=0, y=0, z=1)
        )
    elif camera_preset == "Outfield View":
        camera = dict(
            eye=dict(x=300, y=0, z=50),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        )
    elif camera_preset == "Overhead":
        camera = dict(
            eye=dict(x=0, y=0, z=400),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=1, z=0)
        )
    elif camera_preset == "Side View":
        camera = dict(
            eye=dict(x=0, y=-200, z=100),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        )
    else:  # Custom
        camera = dict(
            eye=dict(x=1.5, y=1.5, z=0.8),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        )
    
    # Update layout
    fig.update_layout(
        title=f"{player} - 3D Stadium Spray Chart",
        scene=dict(
            xaxis=dict(title="Distance (ft)", range=[-50, 400]),
            yaxis=dict(title="Spray (ft)", range=[-250, 250]),
            zaxis=dict(title="Height (ft)", range=[0, 150]),
            aspectratio=dict(x=2, y=1.5, z=0.75),
            camera=camera
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Display plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics
    if show_stats and trajectories:
        st.subheader("📊 Hit Statistics")
        
        stats_df = pd.DataFrame({
            'Hit': [f"Hit {idx+1}" for idx in selected_hits],
            'Type': [player_df.iloc[idx]['hit_type'] for idx in selected_hits],
            'Exit Velocity': [f"{player_df.iloc[idx]['exit_velocity']:.1f} mph" for idx in selected_hits],
            'Launch Angle': [f"{player_df.iloc[idx]['launch_angle']:.1f}°" for idx in selected_hits],
            'Spray Angle': [f"{player_df.iloc[idx]['spray_angle']:.1f}°" for idx in selected_hits],
            'Distance': [f"{traj['distance']:.0f} ft" for traj in trajectories],
            'Max Height': [f"{traj['max_height']:.0f} ft" for traj in trajectories],
            'Hang Time': [f"{traj['hang_time']:.1f} s" for traj in trajectories]
        })
        
        st.dataframe(stats_df, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_distance = np.mean([traj['distance'] for traj in trajectories])
            st.metric("Avg Distance", f"{avg_distance:.0f} ft")
        with col2:
            max_distance = np.max([traj['distance'] for traj in trajectories])
            st.metric("Max Distance", f"{max_distance:.0f} ft")
        with col3:
            avg_ev = player_df.iloc[selected_hits]['exit_velocity'].mean()
            st.metric("Avg Exit Velocity", f"{avg_ev:.1f} mph")
        with col4:
            avg_la = player_df.iloc[selected_hits]['launch_angle'].mean()
            st.metric("Avg Launch Angle", f"{avg_la:.1f}°")

if __name__ == "__main__":
    main()fig.write_html(
    "spray_chart.html",
    include_plotlyjs="cdn",
    full_html=True
)
print("Wrote spray_chart.html — open this in any browser, or host it statically.")
