#!/usr/bin/env python3
"""
Program 1: Setup and Verification (01_verify_setup.py)
------------------------------------------------------
Newtonian Aerodynamics Solver:
- Loads configuration (config.yaml) with clean Body Axes and Flight Conditions:
    * Body Axes: Roll (Nose forward), Yaw (Dorsal upward), Pitch (Starboard right)
    * Flight Condition: Airspeed [m/s], Angle of Attack alpha [deg], Sideslip beta [deg]
- Automatically computes orthonormal body frame and right-handed wind frame
  (Drag along v_inf, Lift in vertical symmetry plane, Side along lateral axis).
- Directly opens a hardware-accelerated interactive 3D PyVista desktop window:
    * Trigonometrically-proportioned arrows originating from vehicle centroid,
      each extending beyond the STL bounding box boundary into free space.
    * Single Drag Axis arrow (+D) along v_inf (no redundant flow arrows).
    * Large, high-contrast, prominent 3D typography (22 pt bold) for all axis labels.
    * Interactive on-screen checkboxes and keyboard shortcuts:
        - [A] Body coordinate axes (Roll: Red, Pitch: Green, Yaw: Blue)
        - [D] Drag Axis (+D) (Pink)
        - [L] Lift Axis (+L) (Purple)
        - [W] Mesh wireframe overlay
    * Instant camera view presets:
        - [1] Isometric view
        - [2] Top view (+Z)
        - [3] Front view (+Y)
        - [4] Side view (+X)
        - [5] Windward view (looking along incoming freestream flow)
        - [R] Reset camera view
    * Surface colored by geometric flow-facing condition (Windward vs. Leeward).
    * On-screen HUD displaying body axes, alpha, beta, and reference metrics.
"""

import os
import sys
import argparse
import numpy as np
import pyvista as pv

from solver_utils import (
    load_config,
    compute_flow_vectors,
    load_mesh_geometry,
    resolve_center_of_gravity,
    print_table,
)


def compute_ray_box_protrusion_length(direction, center, bounds, diag):
    """
    Trigonometric calculation of the arrow length along a 3D unit direction vector
    from the centroid to pierce and extend comfortably outside the STL bounding box.

    For ray r(t) = center + t * u:
      t_k = (bounds[k] - center[k]) / u[k] for each axis k.
      t_exit = min(t_k) is the exact distance where the ray pierces the bounding box.
      L = t_exit + protrusion ensures the arrow head and text emerge into free space.
    """
    u = np.array(direction, dtype=np.float64)
    norm_u = np.linalg.norm(u)
    if norm_u < 1e-12:
        return 0.5 * diag, 0.5 * diag
    u = u / norm_u

    t_candidates = []
    for k in range(3):
        if u[k] > 1e-6:
            t_candidates.append((bounds[1, k] - center[k]) / u[k])
        elif u[k] < -1e-6:
            t_candidates.append((bounds[0, k] - center[k]) / u[k])

    t_box = min(t_candidates) if t_candidates else 0.5 * diag
    # Ensure arrow head and text extend prominently past the bounding box face
    protrusion = max(0.22 * t_box, 0.16 * diag)
    arrow_len = t_box + protrusion
    return arrow_len, t_box


def make_sleek_arrow(plotter, start, direction, length, color, diag, opacity=1.0):
    """
    Generate a slender, well-proportioned aerodynamic arrow mesh.
    Ensures the shaft and tip maintain physically slender dimensions in meters:
    - Shaft radius ~ 0.3% of model diagonal (~0.38 m for Starship)
    - Tip radius ~ 1.0% of model diagonal (~1.25 m for Starship)
    - Tip length ~ 4.0% of model diagonal (~5.0 m for Starship)
    """
    u_dir = np.array(direction, dtype=np.float64)
    norm_d = np.linalg.norm(u_dir)
    if norm_d < 1e-12 or length < 1e-6:
        return None
    u_dir = u_dir / norm_d

    # Physical target dimensions in model units (meters)
    target_shaft_r = max(0.003 * diag, 0.05)
    target_tip_r = max(0.010 * diag, 0.15)
    target_tip_l = max(0.040 * diag, 0.50)

    # pv.Arrow requires dimensionless fractions relative to total arrow length (scale)
    frac_shaft = min(target_shaft_r / length, 0.02)
    frac_tip_r = min(target_tip_r / length, 0.06)
    frac_tip_l = min(target_tip_l / length, 0.20)

    arrow_geom = pv.Arrow(
        start=start,
        direction=u_dir,
        scale=length,
        tip_length=frac_tip_l,
        tip_radius=frac_tip_r,
        shaft_radius=frac_shaft,
        tip_resolution=24,
        shaft_resolution=24,
    )
    return plotter.add_mesh(arrow_geom, color=color, opacity=opacity, smooth_shading=True)


def launch_interactive_pyvista_setup(mesh, flow_data, cfg, off_screen=False):
    """
    Launch interactive hardware-accelerated 3D setup visualizer using PyVista.
    Provides on-screen interactive checkboxes and keyboard view shortcuts.
    """
    # Flow and body vectors
    v_inf = flow_data["v_inf"]
    u_roll = flow_data["roll_axis"]
    u_pitch = flow_data["pitch_axis"]
    u_yaw = flow_data["yaw_axis"]
    lift_axis = flow_data["lift_axis"]

    alpha_deg = flow_data.get("alpha_deg", 0.0)
    beta_deg = flow_data.get("beta_deg", 0.0)

    cg, cg_mode = resolve_center_of_gravity(cfg.get("aerodynamics", {}), mesh)
    center = cg
    bounds = mesh.bounds
    extents = mesh.extents
    diag = float(np.linalg.norm(extents))

    # Initialize PyVista Plotter
    plotter = pv.Plotter(
        title="Hypersonic Newtonian Aerodynamics - 3D Setup Verification",
        off_screen=off_screen,
    )
    plotter.set_background("#0b0e14")

    # 1. Mesh with Flow-Facing Windward / Leeward coloring
    poly = pv.wrap(mesh)
    dots = np.dot(mesh.face_normals, v_inf)
    facing = (dots < 0.0).astype(int)  # 1 = Windward, 0 = Leeward
    poly.cell_data["Flow_Facing"] = facing

    mesh_actor = plotter.add_mesh(
        poly,
        scalars="Flow_Facing",
        cmap=["#475569", "#10b981"],  # Leeward: Slate Gray, Windward: Emerald
        style="surface",
        show_edges=False,
        edge_color="#334155",
        opacity=0.92,
        scalar_bar_args={
            "title": "Surface Orientation (0: Leeward, 1: Windward)",
            "title_font_size": 16,
            "label_font_size": 14,
            "color": "white",
            "n_labels": 2,
            "vertical": False,
            "position_x": 0.30,
            "position_y": 0.03,
            "width": 0.40,
            "height": 0.07,
        },
    )

    # 2. Body Coordinate Axes (Start at center, extend beyond STL bounding box)
    plotter.add_axes(color="white")

    # Compute trig-based protrusion lengths beyond bounding box
    L_roll, _ = compute_ray_box_protrusion_length(u_roll, center, bounds, diag)
    L_pitch, _ = compute_ray_box_protrusion_length(u_pitch, center, bounds, diag)
    L_yaw, _ = compute_ray_box_protrusion_length(u_yaw, center, bounds, diag)

    roll_arrow = make_sleek_arrow(plotter, center, u_roll, L_roll, color="#ef4444", diag=diag)
    pitch_arrow = make_sleek_arrow(plotter, center, u_pitch, L_pitch, color="#22c55e", diag=diag)
    yaw_arrow = make_sleek_arrow(plotter, center, u_yaw, L_yaw, color="#3b82f6", diag=diag)

    # Large, bold, high-contrast 3D labels positioned outside the bounding box
    lbl_offset = 0.04 * diag
    roll_lbl = plotter.add_point_labels(
        np.array([center + (L_roll + lbl_offset) * u_roll]),
        ["+Roll (Nose)"],
        font_size=22,
        bold=True,
        shadow=True,
        show_points=False,
        always_visible=True,
        text_color="#ef4444",
    )
    pitch_lbl = plotter.add_point_labels(
        np.array([center + (L_pitch + lbl_offset) * u_pitch]),
        ["+Pitch (Starboard)"],
        font_size=22,
        bold=True,
        shadow=True,
        show_points=False,
        always_visible=True,
        text_color="#22c55e",
    )
    yaw_lbl = plotter.add_point_labels(
        np.array([center + (L_yaw + lbl_offset) * u_yaw]),
        ["+Yaw (Dorsal)"],
        font_size=22,
        bold=True,
        shadow=True,
        show_points=False,
        always_visible=True,
        text_color="#3b82f6",
    )

    body_actors = [roll_arrow, pitch_arrow, yaw_arrow, roll_lbl, pitch_lbl, yaw_lbl]

    # 3. Aerodynamic Wind Frame: Drag Axis (+D) and Lift Axis (+L)
    # Both start from centroid and extend outside the bounding box
    L_drag, _ = compute_ray_box_protrusion_length(v_inf, center, bounds, diag)
    L_lift, _ = compute_ray_box_protrusion_length(lift_axis, center, bounds, diag)

    drag_arrow = make_sleek_arrow(plotter, center, v_inf, L_drag, color="#ec4899", diag=diag)
    drag_lbl = plotter.add_point_labels(
        np.array([center + (L_drag + lbl_offset) * v_inf]),
        ["Drag Axis (+D)"],
        font_size=22,
        bold=True,
        shadow=True,
        show_points=False,
        always_visible=True,
        text_color="#f472b6",
    )
    drag_actors = [drag_arrow, drag_lbl]

    lift_arrow = make_sleek_arrow(plotter, center, lift_axis, L_lift, color="#8b5cf6", diag=diag)
    lift_lbl = plotter.add_point_labels(
        np.array([center + (L_lift + lbl_offset) * lift_axis]),
        ["Lift Axis (+L)"],
        font_size=22,
        bold=True,
        shadow=True,
        show_points=False,
        always_visible=True,
        text_color="#c084fc",
    )
    lift_actors = [lift_arrow, lift_lbl]

    # Helper to toggle visibility of actor list
    def set_actor_list_visibility(actor_list, visible):
        for a in actor_list:
            if a is None:
                continue
            if hasattr(a, "visibility"):
                a.visibility = visible
            elif hasattr(a, "SetVisibility"):
                a.SetVisibility(visible)

    # 4. Interactive Checkbox Callbacks
    def toggle_body(state):
        set_actor_list_visibility(body_actors, state)
        if state:
            plotter.show_axes()
        else:
            plotter.hide_axes()
        plotter.render()

    def toggle_drag(state):
        set_actor_list_visibility(drag_actors, state)
        plotter.render()

    def toggle_lift(state):
        set_actor_list_visibility(lift_actors, state)
        plotter.render()

    def toggle_wire(state=None):
        if state is None:
            # Case-insensitive toggle between surface and wireframe
            is_wire = mesh_actor.prop.style.lower() == "wireframe"
            new_style = "surface" if is_wire else "wireframe"
        else:
            new_style = "wireframe" if state else "surface"
        mesh_actor.prop.style = new_style
        plotter.render()

    # Place interactive clickable checkboxes at bottom-left
    plotter.add_checkbox_button_widget(
        toggle_body,
        value=True,
        position=(20.0, 145.0),
        size=28,
        color_on="#22c55e",
        color_off="#64748b",
    )
    plotter.add_text("Body Axes [A]", position=(60, 147), font_size=13, color="white")

    plotter.add_checkbox_button_widget(
        toggle_drag,
        value=True,
        position=(20.0, 105.0),
        size=28,
        color_on="#ec4899",
        color_off="#64748b",
    )
    plotter.add_text("Drag Axis (+D) [D]", position=(60, 107), font_size=13, color="white")

    plotter.add_checkbox_button_widget(
        toggle_lift,
        value=True,
        position=(20.0, 65.0),
        size=28,
        color_on="#8b5cf6",
        color_off="#64748b",
    )
    plotter.add_text("Lift Axis (+L) [L]", position=(60, 67), font_size=13, color="white")

    plotter.add_checkbox_button_widget(
        toggle_wire,
        value=False,
        position=(20.0, 25.0),
        size=28,
        color_on="#3b82f6",
        color_off="#64748b",
    )
    plotter.add_text("Wireframe [W]", position=(60, 27), font_size=13, color="white")

    # State tracker for keyboard shortcuts
    states = {
        "body": True,
        "drag": True,
        "lift": True,
    }

    def key_body():
        states["body"] = not states["body"]
        toggle_body(states["body"])

    def key_drag():
        states["drag"] = not states["drag"]
        toggle_drag(states["drag"])

    def key_lift():
        states["lift"] = not states["lift"]
        toggle_lift(states["lift"])

    def key_wire():
        toggle_wire()

    plotter.add_key_event("a", key_body)
    plotter.add_key_event("A", key_body)
    plotter.add_key_event("d", key_drag)
    plotter.add_key_event("D", key_drag)
    plotter.add_key_event("f", key_drag)  # Flow alias
    plotter.add_key_event("F", key_drag)
    plotter.add_key_event("l", key_lift)
    plotter.add_key_event("L", key_lift)
    plotter.add_key_event("w", key_wire)
    plotter.add_key_event("W", key_wire)

    # Suppress VTK's un-toggled default CharEvent so 'w' toggles reliably
    try:
        iren = plotter.render_window.GetInteractor()
        if hasattr(iren, "RemoveObservers"):
            iren.RemoveObservers("CharEvent")
    except Exception:
        pass

    # 5. Camera View Functions and Shortcuts
    def set_isometric():
        plotter.view_isometric()
        plotter.render()

    def set_top():
        plotter.view_xy()
        plotter.render()

    def set_front():
        plotter.view_xz()
        plotter.render()

    def set_side():
        plotter.view_yz()
        plotter.render()

    def set_windward():
        dist = 2.4 * diag
        cam_pos = center - dist * v_inf
        plotter.camera_position = [cam_pos, center, lift_axis]
        plotter.render()

    def reset_view():
        plotter.reset_camera()
        plotter.render()

    plotter.add_key_event("1", set_isometric)
    plotter.add_key_event("2", set_top)
    plotter.add_key_event("3", set_front)
    plotter.add_key_event("4", set_side)
    plotter.add_key_event("5", set_windward)
    plotter.add_key_event("r", reset_view)
    plotter.add_key_event("R", reset_view)

    set_isometric()

    # 6. On-Screen Information HUD (Enlarged, Clear Typography)
    hud_text = (
        f"SETUP & FLIGHT CONDITIONS (Program 1)\n"
        f"-----------------------------------------\n"
        f"Model:       {os.path.basename(cfg['geometry']['stl_file'])}\n"
        f"Faces/Area:  {len(mesh.faces):,} faces | {mesh.area:.1f} m²\n"
        f"AoA (α):     {alpha_deg:.1f}°\n"
        f"Sideslip (β):{beta_deg:.1f}°\n"
        f"Roll Axis:   [{u_roll[0]:+.2f}, {u_roll[1]:+.2f}, {u_roll[2]:+.2f}] (Nose)\n"
        f"Pitch Axis:  [{u_pitch[0]:+.2f}, {u_pitch[1]:+.2f}, {u_pitch[2]:+.2f}] (Right)\n"
        f"Yaw Axis:    [{u_yaw[0]:+.2f}, {u_yaw[1]:+.2f}, {u_yaw[2]:+.2f}] (Dorsal)\n"
        f"Drag Axis:   [{v_inf[0]:+.3f}, {v_inf[1]:+.3f}, {v_inf[2]:+.3f}] (+D)\n"
        f"Lift Axis:   [{lift_axis[0]:+.3f}, {lift_axis[1]:+.3f}, {lift_axis[2]:+.3f}] (+L)\n"
        f"Ref Area:    {cfg['aerodynamics']['reference_area']:.1f} m²"
    )
    plotter.add_text(hud_text, position="upper_left", font_size=15, color="white", font="courier")

    controls_text = (
        f"CAMERA PRESETS (Keys):\n"
        f" [1] Isometric\n"
        f" [2] Top (+Z)\n"
        f" [3] Front (+Y)\n"
        f" [4] Side (+X)\n"
        f" [5] Windward (Flow-Facing)\n"
        f" [R] Reset Camera\n\n"
        f"TOGGLES (Buttons / Keys):\n"
        f" [A] Body Axes (XYZ)\n"
        f" [D] Drag Axis (+D)\n"
        f" [L] Lift Axis (+L)\n"
        f" [W] Mesh Wireframe"
    )
    plotter.add_text(controls_text, position="upper_right", font_size=13, color="#cbd5e1", font="courier")

    print("\n[✓] PyVista 3D interactive setup window launched.")
    print("    - Click checkboxes at bottom-left or press [A], [D], [L], [W] to toggle axes/layers.")
    print("    - Press keys [1]-[5] to switch camera views; [R] to reset view.")
    print("    - Close window when visual verification is complete.\n")

    if not off_screen:
        plotter.show()
    else:
        plotter.render()
        plotter.close()


def main():
    parser = argparse.ArgumentParser(
        description="Program 1: Setup and Verification for Newtonian Aerodynamics"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--off-screen", action="store_true", help="Run headlessly (for testing or automation)"
    )
    args = parser.parse_args()

    # 1. Load config
    cfg = load_config(args.config)

    # 2. Load Mesh
    stl_file = cfg["geometry"]["stl_file"]
    scale = float(cfg["geometry"].get("scale", 1.0))
    mesh = load_mesh_geometry(stl_file, scale=scale)

    # 3. Compute and normalize flow vectors
    flow_data = compute_flow_vectors(cfg)
    v_inf = flow_data["v_inf"]
    u_roll = flow_data["roll_axis"]
    u_pitch = flow_data["pitch_axis"]
    u_yaw = flow_data["yaw_axis"]
    lift_axis = flow_data["lift_axis"]
    side_axis = flow_data["side_axis"]

    # 4. Print summary tables
    bounds = mesh.bounds
    extents = mesh.extents
    centroid = mesh.centroid
    cg, cg_mode = resolve_center_of_gravity(cfg.get("aerodynamics", {}), mesh)

    geom_rows = [
        ("STL File", stl_file),
        ("Number of Triangles (Faces)", f"{len(mesh.faces):,}"),
        ("Number of Vertices", f"{len(mesh.vertices):,}"),
        ("Total Surface Area", f"{mesh.area:.3f} m²"),
        ("Watertight (Closed Solid)", str(mesh.is_watertight)),
        ("Bounding Box X [min, max]", f"[{bounds[0,0]:.2f}, {bounds[1,0]:.2f}] m  (ΔX = {extents[0]:.2f} m)"),
        ("Bounding Box Y [min, max]", f"[{bounds[0,1]:.2f}, {bounds[1,1]:.2f}] m  (ΔY = {extents[1]:.2f} m)"),
        ("Bounding Box Z [min, max]", f"[{bounds[0,2]:.2f}, {bounds[1,2]:.2f}] m  (ΔZ = {extents[2]:.2f} m)"),
        ("Geometric Centroid", f"[{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}] m"),
        ("Center of Gravity (CG)", f"[{cg[0]:.2f}, {cg[1]:.2f}, {cg[2]:.2f}] m ({cg_mode})"),
    ]
    print("\n--- GEOMETRY PROPERTIES ---")
    print_table(geom_rows)

    mach_inf = flow_data.get("mach_infinity")
    gamma = flow_data.get("gamma", 1.4)
    aero_rows = [
        ("Angle of Attack (alpha)", f"{flow_data['alpha_deg']:.2f}°"),
        ("Sideslip Angle (beta)", f"{flow_data['beta_deg']:.2f}°"),
    ]
    if mach_inf is not None:
        aero_rows.extend([
            ("Freestream Mach (M_inf)", f"{mach_inf:.2f}"),
            ("Specific Heat Ratio (gamma)", f"{gamma:.2f}"),
            ("Stagnation Cp_max (Auto)", f"{cfg['aerodynamics']['cp_max']:.4f}"),
        ])
    else:
        aero_rows.append(("Cp_max (Stagnation Pressure)", f"{cfg['aerodynamics']['cp_max']:.3f}"))

    aero_rows.extend([
        ("Body Roll Axis (Nose)", f"[{u_roll[0]:+.4f}, {u_roll[1]:+.4f}, {u_roll[2]:+.4f}]"),
        ("Body Pitch Axis (Starboard)", f"[{u_pitch[0]:+.4f}, {u_pitch[1]:+.4f}, {u_pitch[2]:+.4f}]"),
        ("Body Yaw Axis (Dorsal)", f"[{u_yaw[0]:+.4f}, {u_yaw[1]:+.4f}, {u_yaw[2]:+.4f}]"),
        ("Drag Axis (+D)", f"[{v_inf[0]:+.4f}, {v_inf[1]:+.4f}, {v_inf[2]:+.4f}]"),
        ("Lift Axis (+L)", f"[{lift_axis[0]:+.4f}, {lift_axis[1]:+.4f}, {lift_axis[2]:+.4f}]"),
        ("Side Force Axis", f"[{side_axis[0]:+.4f}, {side_axis[1]:+.4f}, {side_axis[2]:+.4f}]"),
        ("Aerodynamic Reference Area S_ref", f"{cfg['aerodynamics']['reference_area']:.3f} m²"),
    ])

    print("\n--- AERODYNAMIC & FLIGHT CONDITIONS ---")
    print_table(aero_rows)

    # 5. Launch interactive PyVista desktop visualizer
    launch_interactive_pyvista_setup(mesh, flow_data, cfg, off_screen=args.off_screen)




if __name__ == "__main__":
    main()
