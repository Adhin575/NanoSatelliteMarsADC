import numpy as np

# ─────────────────────────────────────────────
# Mars mission constants
# ─────────────────────────────────────────────
MU_MARS   = 42828.3        # km³/s²  — Mars gravitational parameter
R_MARS    = 3396.19        # km      — Mars mean radius

# LMO parameters
H_LMO     = 400.0          # km      — LMO altitude
R_LMO     = R_MARS + H_LMO          # km      — LMO orbit radius  (3796.19 km)
OMEGA_LMO = np.radians(20.0)        # rad     — LMO RAAN
I_LMO     = np.radians(30.0)        # rad     — LMO inclination
THETA0_LMO = np.radians(60.0)       # rad     — LMO initial true latitude
THETA_DOT_LMO = np.sqrt(MU_MARS / R_LMO**3)   # rad/s  (≈ 0.000884797)

# GMO parameters
R_GMO     = 20424.2        # km      — GMO orbit radius
OMEGA_GMO = np.radians(0.0)         # rad
I_GMO     = np.radians(0.0)         # rad
THETA0_GMO = np.radians(250.0)      # rad     — GMO initial true latitude
THETA_DOT_GMO = np.sqrt(MU_MARS / R_GMO**3)   # rad/s  (≈ 0.0000709003)


# ─────────────────────────────────────────────
# Helper: elementary rotation matrices
# ─────────────────────────────────────────────
def R1(angle):
    """Principal rotation matrix about axis 1 (x-axis)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1,  0,  0],
                     [0,  c,  s],
                     [0, -s,  c]])

def R3(angle):
    """Principal rotation matrix about axis 3 (z-axis)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[ c,  s,  0],
                     [-s,  c,  0],
                     [ 0,  0,  1]])


# ─────────────────────────────────────────────
# Core function: DCM from Hill frame H → Inertial frame N
# ─────────────────────────────────────────────
def dcm_HN(Omega, inc, theta):
    """
    Compute [HN] via the (3-1-3) Euler angle sequence.
    [HN] = R3(theta) @ R1(inc) @ R3(Omega)

    Parameters
    ----------
    Omega : float  — right ascension of ascending node (rad)
    inc   : float  — inclination (rad)
    theta : float  — true latitude angle (rad)

    Returns
    -------
    HN : (3,3) ndarray  — DCM such that h_vec = HN @ n_vec
    """
    return R3(theta) @ R1(inc) @ R3(Omega)


# ─────────────────────────────────────────────
# Task 1 main function
# ─────────────────────────────────────────────
def orbit_position_velocity(r, Omega, inc, theta, theta_dot):
    """
    Compute the inertial position and velocity vectors of a circular orbit.

    Derivation
    ----------
    In the Hill frame:
        r_H  = r * [1, 0, 0]^T          (along î_r)
        ṙ_H  = r*θ̇ * [0, 1, 0]^T        (along î_θ, from ṙ = r θ̇ î_θ for circular orbit)

    Transforming to inertial:
        N_r  = [NH] @ r_H  = [HN]^T @ r_H
        N_ṙ  = [NH] @ ṙ_H  = [HN]^T @ ṙ_H

    Parameters
    ----------
    r         : float  — orbit radius (km)
    Omega     : float  — RAAN (rad)
    inc       : float  — inclination (rad)
    theta     : float  — true latitude (rad)
    theta_dot : float  — orbit rate (rad/s)

    Returns
    -------
    N_r  : (3,) ndarray  — inertial position vector (km)
    N_rdot : (3,) ndarray  — inertial velocity vector (km/s)
    """
    HN = dcm_HN(Omega, inc, theta)
    NH = HN.T  # [NH] = [HN]^T  (orthogonal matrix)

    # Hill-frame position and velocity
    r_H    = np.array([r,              0.0, 0.0])
    rdot_H = np.array([0.0,  r * theta_dot, 0.0])

    # Rotate to inertial frame
    N_r    = NH @ r_H
    N_rdot = NH @ rdot_H

    return N_r, N_rdot


# ─────────────────────────────────────────────
# Validation runs
# ─────────────────────────────────────────────
def validate():
    print("=" * 60)
    print("ASEN 5010  —  Task 1: Orbit Simulation Validation")
    print("=" * 60)

    # ── LMO at t = 450 s ──────────────────────────────────────
    t_LMO = 450.0   # s
    theta_LMO = THETA0_LMO + THETA_DOT_LMO * t_LMO

    N_r_LMO, N_rdot_LMO = orbit_position_velocity(
        R_LMO, OMEGA_LMO, I_LMO, theta_LMO, THETA_DOT_LMO
    )

    print(f"\n{'─'*60}")
    print(f"LMO Satellite   (t = {t_LMO:.0f} s)")
    print(f"{'─'*60}")
    print(f"  r_LMO          = {R_LMO:.4f} km")
    print(f"  θ_dot_LMO      = {THETA_DOT_LMO:.9f} rad/s")
    print(f"  θ_LMO(450 s)   = {np.degrees(theta_LMO):.6f} deg"
          f"  ({theta_LMO:.6f} rad)")
    print(f"\n  N_r_LMO  [km]   =")
    print(f"    n1: {N_r_LMO[0]:>14.6f}")
    print(f"    n2: {N_r_LMO[1]:>14.6f}")
    print(f"    n3: {N_r_LMO[2]:>14.6f}")
    print(f"  |N_r_LMO|        = {np.linalg.norm(N_r_LMO):.6f} km  "
          f"(should be {R_LMO:.4f} km)")
    print(f"\n  N_rdot_LMO [km/s] =")
    print(f"    n1: {N_rdot_LMO[0]:>14.6f}")
    print(f"    n2: {N_rdot_LMO[1]:>14.6f}")
    print(f"    n3: {N_rdot_LMO[2]:>14.6f}")
    print(f"  |N_rdot_LMO|     = {np.linalg.norm(N_rdot_LMO):.6f} km/s  "
          f"(should be {R_LMO * THETA_DOT_LMO:.6f} km/s)")

    # ── GMO at t = 1150 s ─────────────────────────────────────
    t_GMO = 1150.0  # s
    theta_GMO = THETA0_GMO + THETA_DOT_GMO * t_GMO

    N_r_GMO, N_rdot_GMO = orbit_position_velocity(
        R_GMO, OMEGA_GMO, I_GMO, theta_GMO, THETA_DOT_GMO
    )

    print(f"\n{'─'*60}")
    print(f"GMO Satellite   (t = {t_GMO:.0f} s)")
    print(f"{'─'*60}")
    print(f"  r_GMO          = {R_GMO:.4f} km")
    print(f"  θ_dot_GMO      = {THETA_DOT_GMO:.10f} rad/s")
    print(f"  θ_GMO(1150 s)  = {np.degrees(theta_GMO):.6f} deg"
          f"  ({theta_GMO:.6f} rad)")
    print(f"\n  N_r_GMO  [km]   =")
    print(f"    n1: {N_r_GMO[0]:>14.6f}")
    print(f"    n2: {N_r_GMO[1]:>14.6f}")
    print(f"    n3: {N_r_GMO[2]:>14.6f}")
    print(f"  |N_r_GMO|        = {np.linalg.norm(N_r_GMO):.6f} km  "
          f"(should be {R_GMO:.4f} km)")
    print(f"\n  N_rdot_GMO [km/s] =")
    print(f"    n1: {N_rdot_GMO[0]:>14.6f}")
    print(f"    n2: {N_rdot_GMO[1]:>14.6f}")
    print(f"    n3: {N_rdot_GMO[2]:>14.6f}")
    print(f"  |N_rdot_GMO|     = {np.linalg.norm(N_rdot_GMO):.6f} km/s  "
          f"(should be {R_GMO * THETA_DOT_GMO:.6f} km/s)")

    print(f"\n{'─'*60}")
    print("Sanity checks  (magnitude must equal orbit radius / speed)")
    lmo_pos_ok  = np.isclose(np.linalg.norm(N_r_LMO),    R_LMO,               rtol=1e-9)
    lmo_vel_ok  = np.isclose(np.linalg.norm(N_rdot_LMO), R_LMO*THETA_DOT_LMO, rtol=1e-9)
    gmo_pos_ok  = np.isclose(np.linalg.norm(N_r_GMO),    R_GMO,               rtol=1e-9)
    gmo_vel_ok  = np.isclose(np.linalg.norm(N_rdot_GMO), R_GMO*THETA_DOT_GMO, rtol=1e-9)
    print(f"  LMO |r| correct  : {lmo_pos_ok}")
    print(f"  LMO |ṙ| correct  : {lmo_vel_ok}")
    print(f"  GMO |r| correct  : {gmo_pos_ok}")
    print(f"  GMO |ṙ| correct  : {gmo_vel_ok}")
    print("=" * 60)

    return (N_r_LMO, N_rdot_LMO), (N_r_GMO, N_rdot_GMO)


if __name__ == "__main__":
    validate()
