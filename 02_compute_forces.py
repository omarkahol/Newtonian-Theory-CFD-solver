#!/usr/bin/env python3
"""
Program 2: Computation Engine (02_compute_forces.py)
---------------------------------------------------
Newtonian Aerodynamics Solver:
- Loads configuration (config.yaml) and 3D STL mesh.
- Vectorized calculation of non-dimensional Aerodynamic fields:
    1. Flow-facing geometric check: (n_i · v_inf) < 0  ->  sin(theta_i) = -(n_i · v_inf)
    2. Ray-traced upstream shadow/occlusion detection via trimesh.ray.intersects_any
    3. Pressure Coefficient: Cp_i = Cp_max * sin^2(theta_i)
    4. Aerodynamic Coefficients:
         - Drag Coefficient: Cd = (1/S_ref) * sum( Cp_i * A_i * (n_i · v_inf) )
         - Lift Coefficient: Cl = (1/S_ref) * sum( -Cp_i * A_i * (n_i · lift_axis) )
         - Side Force Coefficient: Cy
         - Lift-to-Drag Ratio: L/D = Cl / Cd
    5. Center of Pressure (r_cp) in geometry coordinates.
- Saves results as .npz (arrays), .json (summary), and .vtu (solution mesh for ParaView/F3D).
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import trimesh

from solver_utils import (
    load_config,
    compute_flow_vectors,
    load_mesh_geometry,
    print_table,
    cell_data_to_point_data,
    compute_rayleigh_pitot,
    compute_local_mach,
    resolve_wake_cp,
    resolve_center_of_gravity,
    compute_stability_derivatives,
)


def compute_newtonian_aerodynamics(mesh, flow_data, cfg):
    """
    Core vectorized Newtonian solver with raycast shadow occlusion detection.
    """
    t_start = time.time()
    timings = {}

    # Extract flow orientation vectors
    v_inf = flow_data["v_inf"]
    lift_axis = flow_data["lift_axis"]
    side_axis = flow_data["side_axis"]

    aero_cfg = cfg["aerodynamics"]
    mach_inf = flow_data.get("mach_infinity")
    gamma = float(flow_data.get("gamma", 1.4))

    # Automatic Rayleigh Pitot calculation of Cp_max if Mach number is specified
    if mach_inf is not None and mach_inf > 0:
        pt2_pinf, cp_max = compute_rayleigh_pitot(mach_inf, gamma)
    else:
        cp_max = float(aero_cfg.get("cp_max", 2.0))
        pt2_pinf = None

    # Resolve wake Cp (-1/M_inf^2 or user-defined value)
    wake_cp, wake_cp_mode = resolve_wake_cp(aero_cfg, mach_inf)

    s_ref = float(aero_cfg.get("reference_area", 100.0))

    shad_cfg = cfg.get("shadowing", {})
    shadow_enabled = bool(shad_cfg.get("enabled", True))
    ray_epsilon = float(shad_cfg.get("ray_epsilon", 1.0e-4))

    # Mesh geometric properties
    t0 = time.time()
    normals = mesh.face_normals          # (N, 3) unit outward normals
    areas = mesh.area_faces              # (N,) triangle face areas
    centroids = mesh.triangles_center    # (N, 3) face centroids
    n_faces = len(mesh.faces)
    timings["geometry_extraction_s"] = time.time() - t0

    # 1. Flow-facing geometric condition: (n_i · v_inf) < 0
    t0 = time.time()
    dot_products = np.dot(normals, v_inf)  # cos(angle between normal and flow)
    facing_mask = dot_products < 0.0
    n_facing = int(np.sum(facing_mask))
    timings["flow_facing_check_s"] = time.time() - t0

    # Local inclination sin(theta) = max(0, -dot)
    sin_theta = np.clip(-dot_products, 0.0, 1.0)
    inclination_deg = np.degrees(np.arcsin(sin_theta)).astype(np.float32)

    # 2. Ray-traced shadow occlusion detection
    t0 = time.time()
    shadow_mask = np.zeros(n_faces, dtype=bool)

    if shadow_enabled and n_facing > 0:
        facing_indices = np.where(facing_mask)[0]
        c_facing = centroids[facing_indices]

        # Offset ray origins upstream along -v_inf to prevent self-intersection at t=0
        ray_origins = c_facing - ray_epsilon * v_inf
        ray_directions = np.tile(-v_inf, (len(facing_indices), 1))

        # Perform ray-mesh intersection query
        hits = mesh.ray.intersects_any(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
        )
        shadow_mask[facing_indices[hits]] = True

    n_shadowed = int(np.sum(shadow_mask))
    timings["raycast_shadow_s"] = time.time() - t0

    # 3. Active windward faces: geometrically flow-facing AND unshadowed
    active_mask = facing_mask & (~shadow_mask)
    n_active = int(np.sum(active_mask))

    # 4. Pressure Coefficient (Cp)
    t0 = time.time()
    cp = np.full(n_faces, wake_cp, dtype=np.float64)

    if n_active > 0:
        # Newtonian pressure: Cp = Cp_max * sin^2(theta)
        cp[active_mask] = cp_max * (sin_theta[active_mask] ** 2)

    # Local Mach Number & Static Pressure Ratio via Isentropic Expansion from Pt2
    if mach_inf is not None and mach_inf > 0:
        plocal_over_pinf, local_mach = compute_local_mach(
            cp=cp,
            active_mask=active_mask,
            mach_inf=mach_inf,
            gamma=gamma,
            pt2_pinf=pt2_pinf,
            wake_cp=wake_cp,
        )
    else:
        plocal_over_pinf = np.ones(n_faces, dtype=np.float64)
        local_mach = np.zeros(n_faces, dtype=np.float64)

    # 5. Non-Dimensional Aerodynamic Force Integration
    # Force coefficient area vector: d(CF * S_ref) = - Cp_i * A_i * n_i  [m²]
    face_force_areas = - (cp * areas)[:, None] * normals  # Shape (N, 3)
    total_force_area = np.sum(face_force_areas, axis=0)   # Vector in m²

    # Aerodynamic Force Coefficients (non-dimensionalized by S_ref)
    cd_S = float(np.dot(total_force_area, v_inf))         # Drag Area [m²]
    cl_S = float(np.dot(total_force_area, lift_axis))     # Lift Area [m²]
    cy_S = float(np.dot(total_force_area, side_axis))     # Side Area [m²]

    if s_ref > 0:
        cd = cd_S / s_ref
        cl = cl_S / s_ref
        cy = cy_S / s_ref
    else:
        cd = cl = cy = 0.0

    ld_ratio = (cl / cd) if abs(cd) > 1e-12 else 0.0

    # 6. Projected Areas & Aerodynamic Center of Pressure
    total_wetted_area = float(np.sum(areas))
    facing_area = float(np.sum(areas[facing_mask]))
    shadowed_area = float(np.sum(areas[shadow_mask]))
    active_area = float(np.sum(areas[active_mask]))
    projected_frontal_area = float(np.sum(areas[active_mask] * sin_theta[active_mask]))

    # Center of Pressure (r_cp): weighted by local normal pressure force (Cp * Area)
    force_weights = cp * areas
    total_weight = np.sum(force_weights)
    if total_weight > 1e-12:
        r_cp = np.sum(centroids * force_weights[:, None], axis=0) / total_weight
    else:
        r_cp = mesh.centroid

    # 7. Moment Reference Parameters and Baseline Moment Coefficients:
    l_ref_cfg = aero_cfg.get("reference_length")
    l_ref = float(l_ref_cfg) if l_ref_cfg is not None else float(np.max(mesh.extents))

    cg, cg_mode = resolve_center_of_gravity(aero_cfg, mesh)

    u_roll = flow_data["roll_axis"]
    u_pitch = flow_data["pitch_axis"]
    u_yaw = flow_data["yaw_axis"]

    # Aerodynamic Moments about moment center cg:
    # dM_i = (r_i - cg) x dF_i
    r_arm = centroids - cg
    face_moments = np.cross(r_arm, face_force_areas)  # (N, 3) in m^3
    total_moment_area = np.sum(face_moments, axis=0)  # (3,) in m^3

    denom = (s_ref * l_ref) if (s_ref * l_ref) > 0 else 1.0
    c_roll = float(np.dot(total_moment_area, u_roll)) / denom
    c_pitch = float(np.dot(total_moment_area, u_pitch)) / denom
    c_yaw = float(np.dot(total_moment_area, u_yaw)) / denom

    timings["force_integration_s"] = time.time() - t0

    # 9. Aerodynamic Stability Derivatives (Central Differences):
    t_stab = time.time()
    step_deg = float(aero_cfg.get("stability_step_deg", 1.0))
    stability_data = compute_stability_derivatives(
        mesh=mesh,
        flow_data=flow_data,
        cfg=cfg,
        cg=cg,
        l_ref=l_ref,
        s_ref=s_ref,
        cp_max=cp_max,
        wake_cp=wake_cp,
        d_deg=step_deg,
    )
    timings["stability_derivatives_s"] = time.time() - t_stab
    timings["total_solver_time_s"] = time.time() - t_start

    results = {
        "arrays": {
            "cp": cp,
            "local_mach": local_mach,
            "plocal_over_pinf": plocal_over_pinf,
            "face_force_areas": face_force_areas,
            "facing_mask": facing_mask,
            "shadow_mask": shadow_mask,
            "active_mask": active_mask,
            "dot_products": dot_products,
            "sin_theta": sin_theta,
            "inclination_deg": inclination_deg,
            "face_areas": areas,
            "face_normals": normals,
            "face_centroids": centroids,
            "v_inf": v_inf,
            "lift_axis": lift_axis,
            "side_axis": side_axis,
            "r_cp": r_cp,
        },
        "summary": {
            "stl_file": cfg["geometry"]["stl_file"],
            "total_faces": n_faces,
            "facing_faces": n_facing,
            "shadowed_faces": n_shadowed,
            "active_faces": n_active,
            "mach_infinity": mach_inf,
            "gamma": gamma,
            "cp_max": cp_max,
            "stagnation_pressure_ratio_Pt2_Pinf": pt2_pinf,
            "wake_cp": wake_cp,
            "wake_cp_mode": wake_cp_mode,
            "min_local_mach": float(np.min(local_mach)) if mach_inf is not None else None,
            "max_local_mach": float(np.max(local_mach)) if mach_inf is not None else None,
            "avg_windward_mach": float(np.mean(local_mach[active_mask])) if (mach_inf is not None and n_active > 0) else None,
            "reference_area_m2": s_ref,
            "reference_length_m": l_ref,
            "moment_center_m": cg.tolist(),
            "moment_center_mode": cg_mode,
            "alpha_deg": flow_data["alpha_deg"],
            "beta_deg": flow_data["beta_deg"],
            "lift_axis": lift_axis.tolist(),
            "side_axis": side_axis.tolist(),
            "Cm": c_pitch,
            "C_roll": c_roll,
            "Cn": c_yaw,
            "stability_derivatives": stability_data,
            "Cd": cd,
            "Cl": cl,
            "Cy": cy,
            "L_over_D": ld_ratio,
            "Cd_S_ref_m2": cd_S,
            "Cl_S_ref_m2": cl_S,
            "Cy_S_ref_m2": cy_S,
            "total_wetted_area_m2": total_wetted_area,
            "flow_facing_area_m2": facing_area,
            "shadowed_area_m2": shadowed_area,
            "active_windward_area_m2": active_area,
            "projected_frontal_area_m2": projected_frontal_area,
            "center_of_pressure_m": r_cp.tolist(),
            "timings": timings,
        },
    }
    return results


def save_results(mesh, results, output_dir):
    """
    Save non-dimensional solver output data into:
    - .npz (NumPy arrays: cp, masks, geometry)
    - .json (summary aerodynamic metrics)
    - .vtu (ParaView/F3D solution mesh with Cp and flow fields)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. NumPy compressed arrays
    npz_path = os.path.join(output_dir, "aerodynamic_results.npz")
    np.savez_compressed(
        npz_path,
        cp=results["arrays"]["cp"],
        local_mach=results["arrays"]["local_mach"],
        plocal_over_pinf=results["arrays"]["plocal_over_pinf"],
        face_force_areas=results["arrays"]["face_force_areas"],
        facing_mask=results["arrays"]["facing_mask"],
        shadow_mask=results["arrays"]["shadow_mask"],
        active_mask=results["arrays"]["active_mask"],
        dot_products=results["arrays"]["dot_products"],
        sin_theta=results["arrays"]["sin_theta"],
        inclination_deg=results["arrays"]["inclination_deg"],
        face_areas=results["arrays"]["face_areas"],
        face_normals=results["arrays"]["face_normals"],
        face_centroids=results["arrays"]["face_centroids"],
        v_inf=results["arrays"]["v_inf"],
        lift_axis=results["arrays"]["lift_axis"],
        side_axis=results["arrays"]["side_axis"],
        r_cp=results["arrays"]["r_cp"],
    )

    # 2. JSON summary metrics
    json_path = os.path.join(output_dir, "aerodynamic_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results["summary"], f, indent=4)

    # 3. ParaView VTK / VTU Export
    try:
        import meshio

        cp = results["arrays"]["cp"]
        m_loc = results["arrays"]["local_mach"]
        p_ratio = results["arrays"]["plocal_over_pinf"]
        normals = results["arrays"]["face_normals"]
        areas = results["arrays"]["face_areas"]
        dots = results["arrays"]["dot_products"]
        facing = results["arrays"]["facing_mask"]
        shadow = results["arrays"]["shadow_mask"]
        inclination_deg = results["arrays"]["inclination_deg"]

        # Facing status: 0 = Leeward, 1 = Shadowed, 2 = Active Windward
        status = np.zeros(len(mesh.faces), dtype=np.int32)
        status[facing & shadow] = 1
        status[facing & ~shadow] = 2

        raw_cell_dict = {
            "Cp": cp,
            "Local_Mach": m_loc,
            "Pressure_Ratio": p_ratio,
            "Face_Normal": normals,
            "Face_Area": areas,
            "Facing_Status": status,
            "Dot_Product": dots,
            "Local_Inclination_Deg": inclination_deg,
        }

        cell_data = {
            "Cp": [cp.astype(np.float32)],
            "Local_Mach": [m_loc.astype(np.float32)],
            "Pressure_Ratio": [p_ratio.astype(np.float32)],
            "Face_Normal": [normals.astype(np.float32)],
            "Face_Area": [areas.astype(np.float32)],
            "Facing_Status": [status],
            "Dot_Product": [dots.astype(np.float32)],
            "Local_Inclination_Deg": [inclination_deg],
        }

        # Interpolate to points for automatic coloring in F3D and smooth ParaView contouring
        point_data = cell_data_to_point_data(
            mesh.faces, len(mesh.vertices), raw_cell_dict, face_areas=areas
        )

        m = meshio.Mesh(
            points=mesh.vertices,
            cells=[("triangle", mesh.faces)],
            cell_data=cell_data,
            point_data=point_data,
        )

        vtu_path = os.path.join(output_dir, "newtonian_solution.vtu")
        m.write(vtu_path)
    except Exception as e:
        print(f"[!] Warning: Could not export VTU file: {e}")

    print(f"\nSaved results to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Program 2: Newtonian Aerodynamic & Aerothermodynamic Computation Engine"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config.yaml"
    )
    args = parser.parse_args()

    # 1. Load config
    cfg = load_config(args.config)

    # 2. Load Mesh
    stl_file = cfg["geometry"]["stl_file"]
    scale = float(cfg["geometry"].get("scale", 1.0))
    mesh = load_mesh_geometry(stl_file, scale=scale)

    # 3. Compute flow vectors
    flow_data = compute_flow_vectors(cfg)

    # 4. Run solver
    results = compute_newtonian_aerodynamics(mesh, flow_data, cfg)
    summary = results["summary"]
    timings = summary["timings"]

    # 5. Display Aerodynamic Performance Summary
    print("\n--- NON-DIMENSIONAL AERODYNAMIC COEFFICIENTS ---")
    perf_rows = [
        ("Flight Orientation", f"α = {flow_data['alpha_deg']:.1f}° | β = {flow_data['beta_deg']:.1f}°"),
    ]
    if summary.get("mach_infinity") is not None:
        wake_str = f"{summary.get('wake_cp', 0.0):+.4f} ({summary.get('wake_cp_mode', 'auto')})"
        perf_rows.extend([
            ("Freestream Mach (M_inf)", f"{summary['mach_infinity']:.2f}"),
            ("Specific Heat Ratio (gamma)", f"{summary['gamma']:.2f}"),
            ("Stagnation Cp_max (Rayleigh)", f"{summary['cp_max']:.4f}"),
            ("Stagnation Pressure Ratio", f"Pt2 / P_inf = {summary.get('stagnation_pressure_ratio_Pt2_Pinf', 0.0):.2f}"),
            ("Wake Pressure Coeff (Cp_wake)", wake_str),
            ("Local Mach Range [min, max]", f"[{summary.get('min_local_mach', 0.0):.2f}, {summary.get('max_local_mach', 0.0):.2f}]"),
        ])
    else:
        wake_str = f"{summary.get('wake_cp', 0.0):+.4f} ({summary.get('wake_cp_mode', 'auto')})"
        perf_rows.extend([
            ("Stagnation Cp_max", f"{summary['cp_max']:.4f}"),
            ("Wake Pressure Coeff (Cp_wake)", wake_str),
        ])

    perf_rows.extend([
        ("Lift Coefficient (Cl)", f"{summary['Cl']:.4f}"),
        ("Drag Coefficient (Cd)", f"{summary['Cd']:.4f}"),
        ("Side Force Coefficient (Cy)", f"{summary['Cy']:.4f}"),
        ("Lift-to-Drag Ratio (L/D)", f"{summary['L_over_D']:.4f}"),
        ("Pitching Moment (Cm)", f"{summary.get('Cm', 0.0):+.4f}"),
        ("Rolling Moment (Cl_roll)", f"{summary.get('C_roll', 0.0):+.4f}"),
        ("Yawing Moment (Cn)", f"{summary.get('Cn', 0.0):+.4f}"),
        ("Drag Area (Cd * S_ref)", f"{summary['Cd_S_ref_m2']:.2f} m²"),
        ("Lift Area (Cl * S_ref)", f"{summary['Cl_S_ref_m2']:.2f} m²"),
        ("Reference Area (S_ref)", f"{summary['reference_area_m2']:.2f} m²"),
        ("Reference Length (L_ref)", f"{summary.get('reference_length_m', 0.0):.2f} m"),
        ("Projected Frontal Area", f"{summary['projected_frontal_area_m2']:.2f} m²"),
        ("Active Windward Area", f"{summary['active_windward_area_m2']:.2f} m²"),
        ("Center of Gravity (CG) [X,Y,Z]", f"[{summary['moment_center_m'][0]:.2f}, {summary['moment_center_m'][1]:.2f}, {summary['moment_center_m'][2]:.2f}] m ({summary.get('moment_center_mode', 'auto')})"),
        ("Center of Pressure (CP) [X,Y,Z]", f"[{summary['center_of_pressure_m'][0]:.2f}, {summary['center_of_pressure_m'][1]:.2f}, {summary['center_of_pressure_m'][2]:.2f}] m"),
    ])
    print_table(perf_rows)

    # 6. Display Aerodynamic Stability Derivatives Summary
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

    # 7. Mesh and timing summary
    print("\n--- MESH & SHADOW OCCLUSION BREAKDOWN ---")
    mesh_rows = [
        ("Total Mesh Triangles", f"{summary['total_faces']:,}"),
        ("Flow-Facing Faces", f"{summary['facing_faces']:,} ({summary['facing_faces']/summary['total_faces']*100:.1f}%)"),
        ("Shadowed / Occluded Faces", f"{summary['shadowed_faces']:,} ({summary['shadowed_faces']/max(1,summary['facing_faces'])*100:.1f}% of facing)"),
        ("Active Windward Faces", f"{summary['active_faces']:,} ({summary['active_faces']/summary['total_faces']*100:.1f}%)"),
    ]
    print_table(mesh_rows)

    print("\n--- EXECUTION TIMING BENCHMARK ---")
    time_rows = [
        ("Flow-Facing Dot Products", f"{timings['flow_facing_check_s']*1000:.2f} ms"),
        ("Raycast Shadow Detection", f"{timings['raycast_shadow_s']*1000:.2f} ms"),
        ("Aerodynamic Force Integration", f"{timings['force_integration_s']*1000:.2f} ms"),
        ("Stability Derivatives (4 pts)", f"{timings.get('stability_derivatives_s', 0.0)*1000:.2f} ms"),
        ("Total Solver Time", f"{timings['total_solver_time_s']:.3f} s"),
    ]
    print_table(time_rows)

    # 8. Save results
    out_dir = cfg["output"].get("output_dir", "results")
    save_results(mesh, results, out_dir)


if __name__ == "__main__":
    main()
