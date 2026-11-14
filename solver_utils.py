"""
Newtonian Aerodynamics Solver - Utility Functions
Provides configuration loading, geometry parsing, vector orientation kinematics,
and formatting utilities for the hypersonic Newtonian solver suite.
"""

import os
import sys
import yaml
import numpy as np
import trimesh


def compute_rayleigh_pitot(mach_inf: float, gamma: float = 1.4):
    """
    Calculate the stagnation pressure ratio Pt2/Pinf and modified Newtonian Cp_max
    across a normal shock using the Rayleigh Pitot tube formula:

        P_t2 / P_inf = [ ((gamma + 1)^2 * M_inf^2) / (4*gamma*M_inf^2 - 2*(gamma - 1)) ]^(gamma / (gamma - 1))
                       * [ (2*gamma*M_inf^2 - (gamma - 1)) / (gamma + 1) ]

        Cp_max = (Pt2 / P_inf - 1) / (0.5 * gamma * M_inf^2)

    Returns:
        (pt2_pinf, cp_max)
    """
    m_inf = float(mach_inf)
    g = float(gamma)
    if m_inf <= 1.0:
        pt2_pinf = (1.0 + 0.5 * (g - 1.0) * m_inf**2) ** (g / (g - 1.0))
        cp_max = (pt2_pinf - 1.0) / (0.5 * g * m_inf**2) if m_inf > 0 else 1.0
        return float(pt2_pinf), float(cp_max)

    m2 = m_inf ** 2
    term1 = (((g + 1.0)**2 * m2) / (4.0 * g * m2 - 2.0 * (g - 1.0))) ** (g / (g - 1.0))
    term2 = (2.0 * g * m2 - (g - 1.0)) / (g + 1.0)
    pt2_pinf = float(term1 * term2)
    cp_max = float((pt2_pinf - 1.0) / (0.5 * g * m2))
    return pt2_pinf, cp_max


def resolve_wake_cp(aero_cfg: dict, mach_inf: float = None):
    """
    Resolve wake Cp value and mode from aerodynamics configuration.
    Supports:
      - "auto" (default): -1.0 / (mach_inf ** 2) if mach_inf > 0 else 0.0
      - float / int: explicit numerical value (e.g. 0.0, -0.01)
      - None / omitted: defaults to "auto"
    Returns:
      (wake_cp_value: float, mode_str: str)
    """
    raw_val = aero_cfg.get("wake_cp", "auto")
    if raw_val is None or (isinstance(raw_val, str) and raw_val.strip().lower() in ["auto", "-1/m_inf^2", "-1/mach^2"]):
        mode = "auto"
        if mach_inf is not None and mach_inf > 0:
            val = -1.0 / (float(mach_inf) ** 2)
        else:
            val = 0.0
    else:
        try:
            val = float(raw_val)
            mode = "custom"
        except (ValueError, TypeError):
            val = 0.0
            mode = "auto"
    return float(val), mode


def resolve_center_of_gravity(aero_cfg: dict, mesh) -> tuple:
    """
    Resolve Center of Gravity (moment reference center) coordinates.
    Supports configuration keys:
      - 'center_of_gravity': [X, Y, Z]
      - 'moment_center': [X, Y, Z]
      - 'cg': [X, Y, Z]
    If omitted or set to "auto", defaults to mesh.centroid.
    Returns:
      (cg_array: np.ndarray, mode_description: str)
    """
    raw_cg = aero_cfg.get("center_of_gravity")
    if raw_cg is None:
        raw_cg = aero_cfg.get("moment_center")
    if raw_cg is None:
        raw_cg = aero_cfg.get("cg")

    if isinstance(raw_cg, (list, tuple)) and len(raw_cg) == 3:
        try:
            cg_arr = np.array([float(x) for x in raw_cg], dtype=np.float64)
            return cg_arr, "user-defined"
        except (ValueError, TypeError):
            pass

    return np.array(mesh.centroid, dtype=np.float64), "auto (mesh centroid)"


def compute_local_mach(
    cp: np.ndarray,
    active_mask: np.ndarray,
    mach_inf: float,
    gamma: float = 1.4,
    pt2_pinf: float = None,
    wake_cp: float = 0.0,
):
    """
    Calculate local static pressure ratio Plocal/Pinf and local Mach number M_local
    via isentropic expansion from the normal shock stagnation point (Pt2).

    Returns:
        (plocal_pinf, local_mach)
    """
    g = float(gamma)
    m_inf = float(mach_inf)

    if pt2_pinf is None:
        pt2_pinf, _ = compute_rayleigh_pitot(m_inf, g)

    # 1. Local static pressure ratio (Step 1: Plocal/Pinf = 1 + 0.5 * gamma * Minf^2 * Cp)
    plocal_pinf = np.maximum(1e-4, 1.0 + 0.5 * g * (m_inf ** 2) * cp)

    # 2. Ratio of local pressure to stagnation pressure (Plocal / Pt2)
    pratio = np.clip(plocal_pinf / pt2_pinf, 1e-12, 1.0)

    # 3. Isentropic expansion:
    # M_local = sqrt( 2/(gamma-1) * [ (Plocal / Pt2)^((1 - gamma)/gamma) - 1 ] )
    mach_sq = (2.0 / (g - 1.0)) * (pratio ** ((1.0 - g) / g) - 1.0)
    local_mach = np.sqrt(np.maximum(0.0, mach_sq))

    # If wake_cp is exactly 0.0, preserve undisturbed freestream baseline on wake
    if abs(wake_cp) < 1e-12:
        local_mach[~active_mask] = m_inf
        plocal_pinf[~active_mask] = 1.0

    return plocal_pinf, local_mach


def compute_stability_derivatives(mesh, flow_data, cfg, cg, l_ref, s_ref, cp_max, wake_cp=0.0, d_deg=1.0):
    """
    Compute 2nd-order central-difference aerodynamic and moment stability derivatives
    with respect to angle of attack alpha and sideslip angle beta:
      - Cl_alpha, Cd_alpha, Cm_alpha (pitch)
      - Cy_beta, Cl_beta (roll), Cn_beta (yaw)
      - Static margin and neutral point
    """
    alpha0 = flow_data["alpha_deg"]
    beta0 = flow_data["beta_deg"]
    normals = mesh.face_normals
    areas = mesh.area_faces
    centroids = mesh.triangles_center
    shad_cfg = cfg.get("shadowing", {})
    shadow_enabled = bool(shad_cfg.get("enabled", True))
    ray_epsilon = float(shad_cfg.get("ray_epsilon", 1.0e-4))
    u_roll = flow_data["roll_axis"]
    u_pitch = flow_data["pitch_axis"]
    u_yaw = flow_data["yaw_axis"]

    def eval_at(alpha, beta):
        pert_cfg = dict(cfg)
        pert_cfg["flight_condition"] = dict(cfg.get("flight_condition", {}))
        pert_cfg["flight_condition"]["alpha_deg"] = alpha
        pert_cfg["flight_condition"]["beta_deg"] = beta
        pert_flow = compute_flow_vectors(pert_cfg)
        v_inf_p = pert_flow["v_inf"]
        lift_axis_p = pert_flow["lift_axis"]
        side_axis_p = pert_flow["side_axis"]

        dots = np.dot(normals, v_inf_p)
        facing = dots < 0.0

        if shadow_enabled and np.any(facing):
            c_f = centroids[facing]
            origins = c_f - ray_epsilon * v_inf_p
            dirs = np.tile(-v_inf_p, (len(c_f), 1))
            hits = mesh.ray.intersects_any(origins, dirs)
            active = np.zeros(len(mesh.faces), dtype=bool)
            active[facing] = ~hits
        else:
            active = facing

        sin_t = np.zeros(len(mesh.faces))
        sin_t[active] = -dots[active]
        cp_p = np.full(len(mesh.faces), wake_cp, dtype=np.float64)
        cp_p[active] = cp_max * (sin_t[active] ** 2)

        f_areas = - (cp_p * areas)[:, None] * normals
        tot_f = np.sum(f_areas, axis=0)

        cd_val = float(np.dot(tot_f, v_inf_p)) / s_ref if s_ref > 0 else 0.0
        cl_val = float(np.dot(tot_f, lift_axis_p)) / s_ref if s_ref > 0 else 0.0
        cy_val = float(np.dot(tot_f, side_axis_p)) / s_ref if s_ref > 0 else 0.0

        m_areas = np.cross(centroids - cg, f_areas)
        tot_m = np.sum(m_areas, axis=0)
        denom = s_ref * l_ref if (s_ref * l_ref) > 0 else 1.0
        c_roll_val = float(np.dot(tot_m, u_roll)) / denom
        c_pitch_val = float(np.dot(tot_m, u_pitch)) / denom
        c_yaw_val = float(np.dot(tot_m, u_yaw)) / denom

        return {
            "Cl": cl_val, "Cd": cd_val, "Cy": cy_val,
            "C_roll": c_roll_val, "Cm": c_pitch_val, "Cn": c_yaw_val
        }

    # Central differences
    d_rad = np.radians(2.0 * d_deg)
    p_a = eval_at(alpha0 + d_deg, beta0)
    m_a = eval_at(alpha0 - d_deg, beta0)
    p_b = eval_at(alpha0, beta0 + d_deg)
    m_b = eval_at(alpha0, beta0 - d_deg)

    cl_alpha = (p_a["Cl"] - m_a["Cl"]) / d_rad
    cd_alpha = (p_a["Cd"] - m_a["Cd"]) / d_rad
    cm_alpha = (p_a["Cm"] - m_a["Cm"]) / d_rad

    cy_beta = (p_b["Cy"] - m_b["Cy"]) / d_rad
    cl_beta = (p_b["C_roll"] - m_b["C_roll"]) / d_rad
    cn_beta = (p_b["Cn"] - m_b["Cn"]) / d_rad

    # Longitudinal static margin: -Cm_alpha / Cl_alpha
    # In standard flight dynamics, SM > 0 (stable) implies the neutral point is aft of CG
    # (i.e. smaller coordinate along forward-pointing roll axis).
    if abs(cl_alpha) > 1e-6:
        static_margin = - cm_alpha / cl_alpha
        neutral_point_roll = float(np.dot(cg, u_roll) - static_margin * l_ref)
    else:
        static_margin = 0.0
        neutral_point_roll = float(np.dot(cg, u_roll))

    deg_factor = np.pi / 180.0

    return {
        "step_size_deg": d_deg,
        "Cl_alpha_per_rad": cl_alpha,
        "Cl_alpha_per_deg": cl_alpha * deg_factor,
        "Cd_alpha_per_rad": cd_alpha,
        "Cd_alpha_per_deg": cd_alpha * deg_factor,
        "Cm_alpha_per_rad": cm_alpha,
        "Cm_alpha_per_deg": cm_alpha * deg_factor,
        "Cy_beta_per_rad": cy_beta,
        "Cy_beta_per_deg": cy_beta * deg_factor,
        "Cl_beta_per_rad": cl_beta,
        "Cl_beta_per_deg": cl_beta * deg_factor,
        "Cn_beta_per_rad": cn_beta,
        "Cn_beta_per_deg": cn_beta * deg_factor,
        "static_margin_fraction": static_margin,
        "static_margin_percent": static_margin * 100.0,
        "neutral_point_roll_m": neutral_point_roll,
    }


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load and validate the YAML configuration file.
    Provides fallback defaults for any unspecified optional parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Defaults for geometry
    geom = cfg.setdefault("geometry", {})
    geom.setdefault("stl_file", "starshipv21.stl")
    geom.setdefault("scale", 1.0)

    # Flight condition defaults
    flight = cfg.setdefault("flight_condition", {})
    flight.setdefault("alpha_deg", 0.0)
    flight.setdefault("beta_deg", 0.0)

    # Resolve Mach number and gamma
    mach_inf = flight.get("mach_infinity")
    if mach_inf is None:
        mach_inf = cfg.get("aerodynamics", {}).get("mach_infinity")
    gamma = flight.get("gamma")
    if gamma is None:
        gamma = cfg.get("aerodynamics", {}).get("gamma", 1.4)
    gamma = float(gamma)

    aero = cfg.setdefault("aerodynamics", {})
    aero.setdefault("reference_area", 1.0)

    if mach_inf is not None:
        mach_inf = float(mach_inf)
        flight["mach_infinity"] = mach_inf
        flight["gamma"] = gamma
        pt2_pinf, cp_max_auto = compute_rayleigh_pitot(mach_inf, gamma)
        aero["cp_max"] = cp_max_auto
        aero["pt2_pinf"] = pt2_pinf
    else:
        flight["mach_infinity"] = None
        flight["gamma"] = gamma
        aero.setdefault("cp_max", 2.0)

    # Defaults for shadowing
    shad = cfg.setdefault("shadowing", {})
    shad.setdefault("enabled", True)
    shad.setdefault("ray_epsilon", 1.0e-4)

    # Defaults for output
    out = cfg.setdefault("output", {})
    out.setdefault("output_dir", "results")

    return cfg


def compute_flow_vectors(cfg: dict) -> dict:
    """
    Compute vehicle body axes and aerodynamic wind triad:
      - v_inf: unit freestream flow direction (drag axis)
      - lift_axis: aerodynamic lift axis (perpendicular to v_inf in vertical symmetry plane)
      - side_axis: aerodynamic side force axis (completing right-handed wind frame)

    Parameters are defined in config.yaml under:
      - body_axes: roll_axis (forward), yaw_axis (up), pitch_axis (starboard)
      - flight_condition: airspeed [m/s], alpha_deg (AoA), beta_deg (sideslip)
    """
    # 1. Parse and orthonormalize vehicle body axes
    body_cfg = cfg.get("body_axes", {})
    roll_entry = body_cfg.get("roll_axis", [0.0, 1.0, 0.0])
    yaw_entry = body_cfg.get("yaw_axis", [1.0, 0.0, 0.0])
    pitch_entry = body_cfg.get("pitch_axis", "auto")

    roll_raw = np.array(roll_entry, dtype=np.float64)
    yaw_raw = np.array(yaw_entry, dtype=np.float64)

    norm_roll = np.linalg.norm(roll_raw)
    if norm_roll < 1e-12:
        raise ValueError("body_axes.roll_axis cannot be a zero vector.")
    u_roll = roll_raw / norm_roll

    # Ensure yaw is strictly orthogonal to roll
    yaw_proj = yaw_raw - np.dot(yaw_raw, u_roll) * u_roll
    norm_yaw = np.linalg.norm(yaw_proj)
    if norm_yaw < 1e-12:
        raise ValueError("body_axes.yaw_axis cannot be collinear with roll_axis.")
    u_yaw = yaw_proj / norm_yaw

    # Pitch axis (Starboard / Lateral): Up x Forward
    if isinstance(pitch_entry, str) and pitch_entry.lower() == "auto":
        u_pitch = np.cross(u_yaw, u_roll)
        u_pitch = u_pitch / np.linalg.norm(u_pitch)
    else:
        pitch_raw = np.array(pitch_entry, dtype=np.float64)
        p_proj = pitch_raw - np.dot(pitch_raw, u_roll) * u_roll - np.dot(pitch_raw, u_yaw) * u_yaw
        norm_p = np.linalg.norm(p_proj)
        if norm_p < 1e-12:
            u_pitch = np.cross(u_yaw, u_roll)
            u_pitch = u_pitch / np.linalg.norm(u_pitch)
        else:
            u_pitch = p_proj / norm_p

    # 2. Parse flight conditions
    flight_cfg = cfg.get("flight_condition", {})
    alpha_deg = float(flight_cfg.get("alpha_deg", 0.0))
    beta_deg = float(flight_cfg.get("beta_deg", 0.0))

    alpha_rad = np.radians(alpha_deg)
    beta_rad = np.radians(beta_deg)

    # 3. Incoming flow velocity direction (v_inf):
    # Standard aerospace convention:
    # Flow comes from ahead (-roll), pitch up (alpha) tilts flow towards dorsal (+yaw),
    # sideslip (beta) tilts flow towards starboard (+pitch).
    v_dir = (
        -np.cos(alpha_rad) * np.cos(beta_rad) * u_roll
        + np.sin(beta_rad) * u_pitch
        + np.sin(alpha_rad) * np.cos(beta_rad) * u_yaw
    )
    norm_v = np.linalg.norm(v_dir)
    v_inf = v_dir / norm_v

    # 4. Aerodynamic Wind Frame:
    # Lift axis lies in the vertical symmetry plane (v_inf and u_yaw), perpendicular to v_inf,
    # pointing in the dorsal/upward hemisphere:
    lift_proj = u_yaw - np.dot(u_yaw, v_inf) * v_inf
    norm_lift = np.linalg.norm(lift_proj)
    if norm_lift < 1e-7:
        # Flow is exactly aligned with yaw axis: fallback perpendicular to pitch
        lift_proj = -u_roll - np.dot(-u_roll, v_inf) * v_inf
        norm_lift = np.linalg.norm(lift_proj)
    lift_axis = lift_proj / norm_lift

    # Side force axis completes right-handed wind frame [v_inf, side_axis, lift_axis]
    side_axis = np.cross(v_inf, lift_axis)
    norm_side = np.linalg.norm(side_axis)
    if norm_side > 0:
        side_axis = side_axis / norm_side

    return {
        "v_inf": v_inf,
        "alpha_deg": alpha_deg,
        "beta_deg": beta_deg,
        "mach_infinity": cfg.get("flight_condition", {}).get("mach_infinity"),
        "gamma": cfg.get("flight_condition", {}).get("gamma", 1.4),
        "roll_axis": u_roll,
        "pitch_axis": u_pitch,
        "yaw_axis": u_yaw,
        "lift_axis": lift_axis,
        "side_axis": side_axis,
        "reference_up": u_yaw,
        "mode": "angles",
    }


def load_mesh_geometry(stl_path: str, scale: float = 1.0) -> trimesh.Trimesh:
    """
    Load an STL file using trimesh, apply scaling if required,
    and verify geometric integrity.
    """
    if not os.path.exists(stl_path):
        raise FileNotFoundError(
            f"STL file '{stl_path}' not found. Please verify the path in config.yaml."
        )

    mesh = trimesh.load_mesh(stl_path)
    if not isinstance(mesh, trimesh.Trimesh):
        # In case a Scene was returned (e.g. multi-body STL)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
        else:
            raise ValueError(f"Unable to parse '{stl_path}' into a triangular surface mesh.")

    if scale != 1.0:
        mesh.apply_scale(scale)

    return mesh


def print_banner(title: str, width: int = 70):
    """Print an attractive ANSI terminal header banner."""
    print("\n" + "=" * width)
    print(f" {title.upper()} ".center(width, " "))
    print("=" * width)


def print_table(rows: list, headers: tuple = ("Property", "Value"), width: int = 70):
    """Print a clean formatted key-value summary table."""
    col1_w = max(len(headers[0]), max(len(str(r[0])) for r in rows)) + 2
    col2_w = width - col1_w - 5

    line = "+" + "-" * (col1_w + 2) + "+" + "-" * (col2_w + 2) + "+"
    print(line)
    print(f"| {headers[0]:<{col1_w}} | {headers[1]:<{col2_w}} |")
    print(line)
    for k, v in rows:
        val_str = str(v)
        if len(val_str) > col2_w:
            val_str = val_str[: col2_w - 3] + "..."
        print(f"| {k:<{col1_w}} | {val_str:<{col2_w}} |")
    print(line)


def cell_data_to_point_data(faces: np.ndarray, n_points: int, cell_dict: dict, face_areas: np.ndarray = None) -> dict:
    """
    Interpolate face/cell data to vertex/point data using area-weighted averaging.
    This guarantees automatic, beautiful surface coloring in lightweight viewers like F3D
    (which look for PointData by default) while preserving exact CellData for ParaView.
    """
    if face_areas is None:
        face_areas = np.ones(len(faces), dtype=np.float64)
    else:
        face_areas = np.asarray(face_areas, dtype=np.float64)

    # Accumulate vertex weights based on incident triangle areas
    vertex_weights = np.zeros(n_points, dtype=np.float64)
    np.add.at(vertex_weights, faces.flatten(), np.repeat(face_areas, 3))
    valid = vertex_weights > 0

    point_dict = {}
    for name, arr in cell_dict.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            weighted = arr * face_areas
            pt_vals = np.zeros(n_points, dtype=np.float64)
            np.add.at(pt_vals, faces.flatten(), np.repeat(weighted, 3))
            pt_vals[valid] /= vertex_weights[valid]
            point_dict[name] = pt_vals.astype(np.float32)
        elif arr.ndim == 2:
            dim = arr.shape[1]
            weighted = arr * face_areas[:, None]
            pt_vals = np.zeros((n_points, dim), dtype=np.float64)
            for d in range(dim):
                np.add.at(pt_vals[:, d], faces.flatten(), np.repeat(weighted[:, d], 3))
            pt_vals[valid] /= vertex_weights[valid, None]
            point_dict[name] = pt_vals.astype(np.float32)

    return point_dict
