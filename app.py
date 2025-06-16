import pandas as pd
import numpy as np
import plotly.graph_objects as go
import trimesh
import colorsys
from typing import List, Dict

# === Constants ===
G = 9.81
MPS_TO_MPH = 2.23694
M_TO_FT = 3.28084

# === Helpers ===
def generate_color_palette(n: int) -> List[str]:
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors

def calculate_trajectory(ev, la, sa, pts=50) -> Dict[str,np.ndarray]:
    v0 = ev / MPS_TO_MPH
    la_r, sa_r = np.radians(la), np.radians(sa)
    vx = v0*np.cos(la_r)*np.cos(sa_r)
    vy = v0*np.cos(la_r)*np.sin(sa_r)
    vz = v0*np.sin(la_r)
    t_f = 2*vz/G
    t = np.linspace(0, t_f, pts)
    x = vx*t*M_TO_FT
    y = vy*t*M_TO_FT
    z = np.maximum(0,(vz*t - 0.5*G*t**2)*M_TO_FT)
    return {
        "x": x, "y": y, "z": z,
        "distance": float(np.sqrt(x[-1]**2 + y[-1]**2)),
        "max_height": float((vz**2/(2*G))*M_TO_FT),
        "hang_time": float(t_f)
    }

# === Main ===
def main():
    # 1) Load your mock data
    df = pd.read_csv("data/hittrax_batted_balls.csv")
    # Here we pick one batter for demo; remove `.iloc[...]` to show all
    batter = df["player_name"].unique()[0]
    player_df = df[df.player_name == batter].reset_index(drop=True)

    # 2) Prepare colors & stadium mesh
    colors = generate_color_palette(len(player_df))
    mesh = trimesh.load("assets/stadium.glb")
    geom = list(mesh.geometry.values())[0]
    V, F = geom.vertices, geom.faces

    # 3) Build animation frames
    frames = []
    for idx, row in player_df.iterrows():
        traj = calculate_trajectory(
            row.exit_velocity, row.launch_angle, row.spray_angle, pts=50
        )
        x,y,z = traj["x"], traj["y"], traj["z"]
        c = colors[idx]
        name = f"Hit {idx+1} ({row.hit_type})"
        for j in range(2, len(x)+1):
            traces = []
            # add all completed hits
            for k in range(idx):
                prev = calculate_trajectory(
                    player_df.iloc[k].exit_velocity,
                    player_df.iloc[k].launch_angle,
                    player_df.iloc[k].spray_angle,
                    pts=50
                )
                traces.append(go.Scatter3d(
                    x=prev["x"], y=prev["y"], z=prev["z"],
                    mode="lines", line=dict(color=colors[k], width=2),
                    showlegend=False
                ))
            # current partial flight
            traces.append(go.Scatter3d(
                x=x[:j], y=y[:j], z=z[:j],
                mode="lines+markers",
                line=dict(color=c, width=4),
                marker=dict(size=3, color=c),
                name=name, hoverinfo="none"
            ))
            frames.append(go.Frame(data=traces, name=f"{idx}_{j}"))

    # 4) Create base figure
    fig = go.Figure(
        data=[
            # stadium mesh
            go.Mesh3d(
                x=V[:,0], y=V[:,1], z=V[:,2],
                i=F[:,0], j=F[:,1], k=F[:,2],
                color="lightgray", opacity=0.5, name="Stadium", showscale=False
            )
        ],
        layout=go.Layout(
            title=f"{batter} — 3D Animated Spray Chart",
            scene=dict(
                xaxis=dict(title="X (ft)", range=[-50,400]),
                yaxis=dict(title="Y (ft)", range=[-250,250]),
                zaxis=dict(title="Z (ft)", range=[0,150]),
                aspectratio=dict(x=2,y=1.5,z=0.75)
            ),
            margin=dict(l=0,r=0,b=0,t=40),
            updatemenus=[dict(
                type="buttons", showactive=False, y=1.05, x=0,
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration":50,"redraw":True},"fromcurrent":True}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame": {"duration":0,"redraw":False},"mode":"immediate"}])
                ]
            )]
        ),
        frames=frames
    )

    # 5) Export standalone HTML
    fig.write_html(
        "spray_chart.html",
        include_plotlyjs="cdn",
        full_html=True
    )
    print("✅ spray_chart.html has been written in your project folder.")

if __name__ == "__main__":
    main()
