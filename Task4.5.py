import numpy as np

# ── Mission constants ────────────────────────────────────────────────
MU_MARS        = 42828.3
R_MARS         = 3396.19

# LMO
R_LMO          = R_MARS + 400.0
OMEGA_LMO      = np.radians(20.0)
I_LMO          = np.radians(30.0)
THETA0_LMO     = np.radians(60.0)
THETA_DOT_LMO  = np.sqrt(MU_MARS / R_LMO**3)   # 0.000884797 rad/s

# GMO
R_GMO          = 20424.2
OMEGA_GMO      = np.radians(0.0)
I_GMO          = np.radians(0.0)
THETA0_GMO     = np.radians(250.0)
THETA_DOT_GMO  = np.sqrt(MU_MARS / R_GMO**3)   # 0.0000709003 rad/s

# Inertial n̂₃ direction (fixed)
N3 = np.array([0.0, 0.0, 1.0])

# ── Elementary rotations ─────────────────────────────────────────────
def R1(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,s],[0,-s,c]])

def R3(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,s,0],[-s,c,0],[0,0,1]])

# ── Orbit position functions (Task 1) ────────────────────────────────
def orbit_pos(r_orb, Omega, inc, theta0, theta_dot, t):
    """Inertial position vector of a circular orbit at time t."""
    theta = theta0 + theta_dot * t
    HN    = R3(theta) @ R1(inc) @ R3(Omega)
    return HN.T @ np.array([r_orb, 0.0, 0.0])

def pos_LMO(t):
    return orbit_pos(R_LMO, OMEGA_LMO, I_LMO, THETA0_LMO, THETA_DOT_LMO, t)

def pos_GMO(t):
    return orbit_pos(R_GMO, OMEGA_GMO, I_GMO, THETA0_GMO, THETA_DOT_GMO, t)

# ════════════════════════════════════════════════════════════════════
#  ANALYTIC DCM  [RcN](t)
# ════════════════════════════════════════════════════════════════════
def dcm_RcN(t):
    """
    GMO-pointing reference frame DCM [RcN](t).

    Construction:
        Δr  = r_GMO(t) − r_LMO(t)
        r̂₁  = −Δr / |Δr|                      (−r̂₁ points at GMO)
        r̂₂  = (Δr × n̂₃) / |Δr × n̂₃|
        r̂₃  = r̂₁ × r̂₂

    Parameters
    ----------
    t : float — time (s)

    Returns
    -------
    RcN : (3,3) ndarray — proper DCM (det = +1)
    """
    r_lmo = pos_LMO(t)
    r_gmo = pos_GMO(t)
    delta_r = r_gmo - r_lmo                     # LMO → GMO vector

    # r̂₁: negative of unit Δr (so −r̂₁ points at GMO)
    r1 = -delta_r / np.linalg.norm(delta_r)

    # r̂₂: in-plane axis perpendicular to Δr in the equatorial-ish sense
    cross2 = np.cross(delta_r, N3)
    r2 = cross2 / np.linalg.norm(cross2)

    # r̂₃: completes right-handed triad
    r3 = np.cross(r1, r2)

    return np.array([r1, r2, r3])               # rows = basis vectors in N

# ════════════════════════════════════════════════════════════════════
#  ANGULAR VELOCITY  NωRc/N   (numerical finite-difference)
# ════════════════════════════════════════════════════════════════════
def omega_RcN_numerical(t, dt=1.0):
    """
    NωRc/N in N-frame components via kinematic finite difference.

    Kinematic relation (transport theorem form):
        [RcN_dot] = −[ω̃_in_Rc] [RcN]

    Equivalently, working with [NcR] = [RcN].T:
        [NcR_dot] = [ω̃_N] [NcR]
    → [ω̃_N] = [NcR_dot] [NcR].T = [NcR_dot] [RcN]

    Finite-difference estimate of [NcR_dot] (central difference):
        [NcR_dot] ≈ ([RcN(t+dt)] − [RcN(t−dt)]).T / (2dt)

    Extract ω from upper-diagonal elements of skew-sym [ω̃_N]:
        Ω = [ω̃_N],  ω₁ = Ω₃₂,  ω₂ = Ω₁₃,  ω₃ = Ω₂₁

    Parameters
    ----------
    t  : float — time (s)
    dt : float — finite-difference step (s), default 1.0 s

    Returns
    -------
    omega_N : (3,) ndarray — rad/s in N-frame
    """
    RcN_dot = (dcm_RcN(t + dt) - dcm_RcN(t - dt)) / (2.0 * dt)
    NcR_dot = RcN_dot.T
    RcN     = dcm_RcN(t)

    omega_tilde_N = NcR_dot @ RcN          # skew-symmetric in N-frame

    # Pull upper-diagonal elements (as instructed)
    omega_N = np.array([
        omega_tilde_N[2, 1],               # ω₁ = Ω₃₂
        omega_tilde_N[0, 2],               # ω₂ = Ω₁₃
        omega_tilde_N[1, 0]                # ω₃ = Ω₂₁
    ])
    return omega_N

# ════════════════════════════════════════════════════════════════════
#  VALIDATION
# ════════════════════════════════════════════════════════════════════
def validate(t=330.0):
    print("="*66)
    print("ASEN 5010  —  Task 5: GMO-Pointing Reference Frame")
    print("="*66)

    # ── Derivation summary ────────────────────────────────────
    print("\nAnalytic construction:")
    print("  Δr  = r_GMO(t) − r_LMO(t)")
    print("  r̂₁  = −Δr/|Δr|              (−r̂₁ → GMO)")
    print("  r̂₂  = (Δr × n̂₃)/|Δr × n̂₃|  (given)")
    print("  r̂₃  = r̂₁ × r̂₂               (RH completion)")
    print()
    print("  NωRc/N computed via numerical finite difference:")
    print("  [ω̃_N] = [NcR_dot] @ [RcN],  ω pulled from upper diagonal")

    # ── Intermediate vectors at t ─────────────────────────────
    r_lmo   = pos_LMO(t)
    r_gmo   = pos_GMO(t)
    delta_r = r_gmo - r_lmo
    print(f"\n{'─'*66}")
    print(f"Intermediate vectors at t = {t:.0f} s")
    print(f"{'─'*66}")
    print(f"  r_LMO  = [{r_lmo[0]:12.4f}, {r_lmo[1]:12.4f}, {r_lmo[2]:12.4f}]  km")
    print(f"  r_GMO  = [{r_gmo[0]:12.4f}, {r_gmo[1]:12.4f}, {r_gmo[2]:12.4f}]  km")
    print(f"  Δr     = [{delta_r[0]:12.4f}, {delta_r[1]:12.4f}, {delta_r[2]:12.4f}]  km")
    print(f"  |Δr|   = {np.linalg.norm(delta_r):.4f} km")

    # ── [RcN] ─────────────────────────────────────────────────
    RcN = dcm_RcN(t)
    print(f"\n{'─'*66}")
    print(f"[RcN] at t = {t:.0f} s")
    print(f"{'─'*66}")
    hdr = f"  {'':12s}  {'n̂₁':>12s}  {'n̂₂':>12s}  {'n̂₃':>12s}"
    print(hdr)
    lbl = ["r̂₁ = −Δr̂", "r̂₂ (given)", "r̂₃ = r̂₁×r̂₂"]
    for i in range(3):
        r = RcN[i]
        print(f"  {lbl[i]:12s}  {r[0]:12.8f}  {r[1]:12.8f}  {r[2]:12.8f}")

    # ── Angular velocity ──────────────────────────────────────
    omega_N = omega_RcN_numerical(t, dt=1.0)

    print(f"\n{'─'*66}")
    print(f"NωRc/N at t = {t:.0f} s  (finite-difference, dt = 1.0 s)")
    print(f"{'─'*66}")
    print(f"  NωRc/N = [{omega_N[0]:14.9f},"
          f" {omega_N[1]:14.9f},"
          f" {omega_N[2]:14.9f}]  rad/s")
    print(f"  |NωRc/N| = {np.linalg.norm(omega_N):.9f} rad/s")

    # Convergence check: compare dt=1.0 vs dt=0.1
    omega_fine = omega_RcN_numerical(t, dt=0.1)
    diff_conv  = np.max(np.abs(omega_N - omega_fine))
    print(f"\n  Convergence check (dt=1.0 vs dt=0.1):")
    print(f"    dt=0.1  result: [{omega_fine[0]:14.9f},"
          f" {omega_fine[1]:14.9f},"
          f" {omega_fine[2]:14.9f}]")
    print(f"    Max difference : {diff_conv:.2e}  " +
          ("✓" if diff_conv < 1e-8 else "✗"))

    # ── Sanity checks ─────────────────────────────────────────
    print(f"\n{'─'*66}")
    print("Sanity checks:")

    # −r̂₁ should point from LMO toward GMO
    delta_r_hat = delta_r / np.linalg.norm(delta_r)
    r1_points_gmo = np.allclose(-RcN[0], delta_r_hat, atol=1e-10)
    print(f"  −r̂₁ ‖ Δr̂ (points at GMO)   : " +
          ("True ✓" if r1_points_gmo else "FAIL ✗"))

    # r̂₂ definition
    cross2 = np.cross(delta_r, N3)
    r2_check = np.allclose(RcN[1], cross2/np.linalg.norm(cross2), atol=1e-10)
    print(f"  r̂₂ = (Δr×n̂₃)/|Δr×n̂₃|      : " +
          ("True ✓" if r2_check else "FAIL ✗"))

    # r̂₃ = r̂₁ × r̂₂ (right-hand rule)
    rh_check = np.allclose(np.cross(RcN[0], RcN[1]), RcN[2], atol=1e-10)
    print(f"  r̂₃ = r̂₁ × r̂₂ (RH rule)     : " +
          ("True ✓" if rh_check else "FAIL ✗"))

    # Orthogonality
    I_err = np.max(np.abs(RcN @ RcN.T - np.eye(3)))
    print(f"  Orthogonality error          : {I_err:.2e}  " +
          ("✓" if I_err < 1e-12 else "✗"))

    # Determinant = +1
    det_val = np.linalg.det(RcN)
    print(f"  det([RcN])                   : {det_val:+.10f}  " +
          ("✓" if np.isclose(det_val, 1.0) else "✗"))

    # Angular velocity magnitude sanity
    # (should be roughly between LMO and GMO rates)
    mag = np.linalg.norm(omega_N)
    in_range = THETA_DOT_GMO * 0.1 < mag < THETA_DOT_LMO * 10
    print(f"  |ω| in plausible range       : {mag:.6e} rad/s  " +
          ("✓" if in_range else "✗"))

    print("="*66)

if __name__ == "__main__":
    validate(t=330.0)
