import pandas as pd
import numpy as np
import plotly.graph_objects as go

# === Load HitTrax Data ===
df = pd.read_csv("data/hittrax_batted_balls.csv")

# Constants
g = 9.81  # gravity (m/s^2)

# Storage
flight_arcs = []
landing_markers = []

def create_flight(x_vel, y_vel, z_vel):
    t_total = 2 * z_vel / g
    t = np.linspace(0, t_total, 40)
    x = x_vel * t
    y = y_vel * t
    z = z_vel * t - 0.5 * g * t**2
    return x * 3.28084, y * 3.28084, z * 3.28084  # Convert to feet

# Build arcs and landing points
for _, row in df.iterrows():
    ev = row['exit_velocity'] * 0.44704
    la = np.radians(row['launch_angle'])
    sa = np.radians(row['spray_angle'])

    vx = ev * np.cos(la) * np.cos(sa)
    vy = ev * np.cos(la) * np.sin(sa)
    vz = ev * np.sin(la)

    x_ft, y_ft, z_ft = create_flight(vx, vy, vz)

    arc = go.Scatter3d(
        x=x_ft,
        y=y_ft,
        z=z_ft,
        mode='lines',
        name=row['player_name'],
        line=dict(width=4),
        hovertext=f"{row['player_name']} - {row['hit_type']}"
    )
    flight_arcs.append(arc)

    dist = np.sqrt(x_ft[-1]**2 + y_ft[-1]**2)
    marker = go.Scatter3d(
        x=[x_ft[-1]],
        y=[y_ft[-1]],
        z=[0],
        mode='markers+text',
        marker=dict(size=6, color='red'),
        text=[f"{int(dist)} ft"],
        textposition='top center',
        showlegend=False
    )
    landing_markers.append(marker)

# === Placeholder Stadium Mesh (green field box) ===
stadium_box = go.Mesh3d(
    x=[0, 300, 0, 300, 0, 300, 0, 300],
    y=[-200, -200, 200, 200, -200, -200, 200, 200],
    z=[0, 0, 0, 0, 40, 40, 40, 40],
    i=[0, 0, 0, 4, 5, 6],
    j=[1, 2, 3, 5, 6, 7],
    k=[5, 6, 7, 6, 7, 4],
    color='green',
    opacity=0.2,
    name='Placeholder Field',
    showscale=False
)

# === Layout ===
layout = go.Layout(
    title="HTL HitTrax: Placeholder Stadium + Flight Arcs",
    scene=dict(
        xaxis=dict(title='X (ft)', range=[-100, 300]),
        yaxis=dict(title='Y (ft)', range=[-200, 200]),
        zaxis=dict(title='Z (ft)', range=[0, 100]),
        aspectratio=dict(x=2, y=1.2, z=0.6)
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    showlegend=True
)

# === Combine & Export ===
fig = go.Figure(data=[stadium_box] + flight_arcs + landing_markers, layout=layout)
fig.write_html("stadium_placeholder_visual.html", auto_open=True)
