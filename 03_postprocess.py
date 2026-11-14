#!/usr/bin/env python3
"""
Program 3: Post-Processing & ParaView Integration (03_postprocess.py)
---------------------------------------------------------------------
Newtonian Aerodynamics Solver:
- Loads simulation results from the output folder (aerodynamic_results.npz & aerodynamic_summary.json).
- Displays comprehensive terminal reports:
    * Non-dimensional aerodynamic coefficients (Cl, Cd, Cy, L/D)
    * Pitching, rolling, and yawing moment coefficients
    * Static stability derivatives (Cm_alpha, Cn_beta, Cl_beta, static margin)
    * Effective drag and lift areas (Cd*S_ref, Cl*S_ref)
    * Center of Pressure (CoP) coordinates
- Optional hardware-accelerated interactive 3D viewer (--interactive)
  with instant toggling between Pressure Coefficient (Cp) and Local Mach Number (M_local).
"""

import os
import sys
import json
import argparse
import numpy as np

from solver_utils import (
    load_config,
    print_table,
)


def load_simulation_results(output_dir: str):
    """
    Load saved NumPy array archive and JSON summary metrics.
    """
    npz_path = os.path.join(output_dir, "aerodynamic_results.npz")
    json_path = os.path.join(output_dir, "aerodynamic_summary.json")

    if not os.path.exists(npz_path) or not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Simulation results not found in '{output_dir}'. "
            "Please run 'python 02_compute_forces.py' first."
        )

    data = np.load(npz_path)
    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    return data, summary


def launch_interactive_pyvista_viewer(vtu_path, summary):
    """
    Open an interactive hardware-accelerated 3D VTK viewer window using PyVista.
    Allows real-time orbiting, panning, zooming, scalar probing, wireframe toggle ('w'),
    and field switching between Pressure Coefficient (Cp) and Local Mach Number ('c').
    """
    try:
        import pyvista as pv
        if not os.path.exists(vtu_path):
            print(f"[!] VTU file not found: {vtu_path}")
            return

        print("\n[*] Initializing desktop interactive 3D VTK viewer (PyVista)...")
        mesh = pv.read(vtu_path)

        plotter = pv.Plotter(title="Hypersonic Newtonian Aerodynamics - Interactive 3D Viewer")
        plotter.set_background("#0b0e14")

        # Active scalar tracker
        current_scalar = ["Cp"]

        mesh_actor = plotter.add_mesh(
            mesh,
            scalars="Cp",
            cmap="turbo",
            show_edges=False,
            scalar_bar_args={
                "title": "Pressure Coefficient (Cp)",
                "title_font_size": 16,
                "label_font_size": 14,
                "color": "white",
                "vertical": True,
                "position_x": 0.88,
                "position_y": 0.25,
            },
        )

        def toggle_wire():
            is_wire = mesh_actor.prop.style.lower() == "wireframe"
            mesh_actor.prop.style = "surface" if is_wire else "wireframe"
            plotter.render()

        def toggle_scalar():
            if current_scalar[0] == "Cp" and "Local_Mach" in mesh.point_data:
                current_scalar[0] = "Local_Mach"
                title = "Local Mach Number (M_local)"
                cmap = "coolwarm"
            else:
                current_scalar[0] = "Cp"
                title = "Pressure Coefficient (Cp)"
                cmap = "turbo"

            plotter.clear()
            plotter.set_background("#0b0e14")
            plotter.add_mesh(
                mesh,
                scalars=current_scalar[0],
                cmap=cmap,
                show_edges=mesh_actor.prop.style.lower() == "wireframe",
                scalar_bar_args={
                    "title": title,
                    "title_font_size": 16,
                    "label_font_size": 14,
                    "color": "white",
                    "vertical": True,
                    "position_x": 0.88,
                    "position_y": 0.25,
                },
            )
            plotter.add_axes(color="white")
            plotter.add_text(info_text, position="lower_left", font_size=12, color="white")
            plotter.render()

        plotter.add_key_event("w", toggle_wire)
        plotter.add_key_event("W", toggle_wire)
        plotter.add_key_event("c", toggle_scalar)
        plotter.add_key_event("C", toggle_scalar)

        try:
            iren = plotter.render_window.GetInteractor()
            if hasattr(iren, "RemoveObservers"):
                iren.RemoveObservers("CharEvent")
        except Exception:
            pass

        plotter.add_axes(color="white")
        mach_str = f"M_inf: {summary['mach_infinity']:.1f} | " if summary.get("mach_infinity") is not None else ""
        info_text = (
            f"{mach_str}Cl: {summary['Cl']:.3f} | Cd: {summary['Cd']:.3f} | L/D: {summary['L_over_D']:.3f}\n"
            f"Controls: Left-Click: Orbit | Right-Click: Zoom | Middle-Click: Pan | 'w': Wireframe | 'c': Cycle (Cp / Mach)"
        )
        plotter.add_text(info_text, position="lower_left", font_size=12, color="white")

        print("[✓] Window open! Inspect pressure or press 'c' for Mach number. Close window when finished.")
        plotter.show()
    except Exception as e:
        print(f"[!] PyVista desktop window could not be opened: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Program 3: Post-Processing & ParaView Export for Newtonian Aerodynamics"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Launch desktop interactive 3D PyVista/VTK viewer window"
    )
    args = parser.parse_args()

    # 1. Load config
    cfg = load_config(args.config)
    out_dir = cfg["output"].get("output_dir", "results")

    # 2. Load simulation results
    data, summary = load_simulation_results(out_dir)

    # 3. Print Comprehensive Non-Dimensional Performance Metrics
    print("\n--- NON-DIMENSIONAL AERODYNAMIC COEFFICIENTS ---")
    rows = [
        ("Flight Orientation", f"α = {summary.get('alpha_deg', 0.0):.1f}° | β = {summary.get('beta_deg', 0.0):.1f}°"),
    ]
    if summary.get("mach_infinity") is not None:
        wake_str = f"{summary.get('wake_cp', 0.0):+.4f} ({summary.get('wake_cp_mode', 'auto')})"
        rows.extend([
            ("Freestream Mach (M_inf)", f"{summary['mach_infinity']:.2f}"),
            ("Specific Heat Ratio (gamma)", f"{summary.get('gamma', 1.4):.2f}"),
            ("Stagnation Cp_max (Rayleigh)", f"{summary['cp_max']:.4f}"),
            ("Stagnation Pressure Ratio", f"Pt2 / P_inf = {summary.get('stagnation_pressure_ratio_Pt2_Pinf', 0.0):.2f}"),
            ("Wake Pressure Coeff (Cp_wake)", wake_str),
            ("Local Mach Range [min, max]", f"[{summary.get('min_local_mach', 0.0):.2f}, {summary.get('max_local_mach', 0.0):.2f}]"),
        ])
    else:
        wake_str = f"{summary.get('wake_cp', 0.0):+.4f} ({summary.get('wake_cp_mode', 'auto')})"
        rows.extend([
            ("Stagnation Cp_max", f"{summary['cp_max']:.4f}"),
            ("Wake Pressure Coeff (Cp_wake)", wake_str),
        ])

    rows.extend([
        ("Lift Coefficient (Cl)", f"{summary['Cl']:.4f}"),
        ("Drag Coefficient (Cd)", f"{summary['Cd']:.4f}"),
        ("Side Force Coefficient (Cy)", f"{summary.get('Cy', 0.0):.4f}"),
        ("Lift-to-Drag Ratio (L/D)", f"{summary['L_over_D']:.4f}"),
        ("Pitching Moment (Cm)", f"{summary.get('Cm', 0.0):+.4f}"),
        ("Rolling Moment (Cl_roll)", f"{summary.get('C_roll', 0.0):+.4f}"),
        ("Yawing Moment (Cn)", f"{summary.get('Cn', 0.0):+.4f}"),
        ("Drag Area (Cd * S_ref)", f"{summary.get('Cd_S_ref_m2', 0.0):.2f} m²"),
        ("Lift Area (Cl * S_ref)", f"{summary.get('Cl_S_ref_m2', 0.0):.2f} m²"),
        ("Reference Area (S_ref)", f"{summary['reference_area_m2']:.2f} m²"),
        ("Reference Length (L_ref)", f"{summary.get('reference_length_m', 0.0):.2f} m"),
        ("Projected Frontal Area", f"{summary['projected_frontal_area_m2']:.2f} m²"),
        ("Active Windward Area", f"{summary['active_windward_area_m2']:.2f} m²"),
        ("Center of Gravity (CG) [X,Y,Z]", f"[{summary['moment_center_m'][0]:.2f}, {summary['moment_center_m'][1]:.2f}, {summary['moment_center_m'][2]:.2f}] m ({summary.get('moment_center_mode', 'auto')})"),
        ("Center of Pressure (CP) [X,Y,Z]", f"[{summary['center_of_pressure_m'][0]:.2f}, {summary['center_of_pressure_m'][1]:.2f}, {summary['center_of_pressure_m'][2]:.2f}] m"),
    ])
    print_table(rows)

    # 4. Print Aerodynamic Stability Derivatives Summary
    if "stability_derivatives" in summary:
        print("\n--- STATIC STABILITY DERIVATIVES ---")
        stab = summary["stability_derivatives"]
        cm_a = stab["Cm_alpha_per_rad"]
        cn_b = stab["Cn_beta_per_rad"]
        cl_b = stab["Cl_beta_per_rad"]

        pitch_stat = "Stable (Cm_alpha < 0)" if cm_a < 0 else ("Unstable (Cm_alpha > 0)" if cm_a > 0 else "Neutral")
        yaw_stat = "Stable (Cn_beta > 0)" if cn_b > 0 else ("Unstable (Cn_beta < 0)" if cn_b < 0 else "Neutral")
        roll_stat = "Stable (Cl_beta < 0)" if cl_b < 0 else ("Unstable (Cl_beta > 0)" if cl_b > 0 else "Neutral")

        stab_rows = [
            ("Lift Curve Slope (Cl_alpha)", f"{stab['Cl_alpha_per_rad']:+.4f} /rad  ({stab['Cl_alpha_per_deg']:+.4f} /deg)"),
            ("Drag Polar Slope (Cd_alpha)", f"{stab['Cd_alpha_per_rad']:+.4f} /rad  ({stab['Cd_alpha_per_deg']:+.4f} /deg)"),
            ("Pitching Moment Slope (Cm_alpha)", f"{stab['Cm_alpha_per_rad']:+.4f} /rad  ({stab['Cm_alpha_per_deg']:+.4f} /deg)"),
            ("Side Force Slope (Cy_beta)", f"{stab['Cy_beta_per_rad']:+.4f} /rad  ({stab['Cy_beta_per_deg']:+.4f} /deg)"),
            ("Dihedral Effect (Cl_beta)", f"{stab['Cl_beta_per_rad']:+.4f} /rad  ({stab['Cl_beta_per_deg']:+.4f} /deg)"),
            ("Directional Stability (Cn_beta)", f"{stab['Cn_beta_per_rad']:+.4f} /rad  ({stab['Cn_beta_per_deg']:+.4f} /deg)"),
            ("Longitudinal Static Margin", f"{stab['static_margin_percent']:+.2f}% L_ref ({pitch_stat.split()[0]})"),
            ("Directional Static Stability", yaw_stat),
            ("Lateral Dihedral Stability", roll_stat),
        ]
        print_table(stab_rows)

    # 5. Mesh & Occlusion Breakdown
    print("\n--- MESH & SHADOW OCCLUSION BREAKDOWN ---")
    mesh_rows = [
        ("Total Surface Triangles", f"{summary['total_faces']:,}"),
        ("Flow-Facing Faces", f"{summary['facing_faces']:,} ({summary['facing_faces']/summary['total_faces']*100:.1f}%)"),
        ("Shadowed / Occluded Faces", f"{summary['shadowed_faces']:,} ({summary['shadowed_faces']/max(1,summary['facing_faces'])*100:.1f}% of facing)"),
        ("Active Windward Faces", f"{summary['active_faces']:,} ({summary['active_faces']/summary['total_faces']*100:.1f}%)"),
    ]
    print_table(mesh_rows)

    vtu_file = os.path.join(out_dir, "newtonian_solution.vtu")

    # Launch desktop viewer if requested
    if args.interactive:
        launch_interactive_pyvista_viewer(vtu_file, summary)


if __name__ == "__main__":
    main()
