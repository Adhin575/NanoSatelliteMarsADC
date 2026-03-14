import numpy as np

# ── Mission constants ────────────────────────────────────────────────
MU_MARS       = 42828.3
R_MARS        = 3396.19
R_LMO         = R_MARS + 400.0
OMEGA_LMO     = np.radians(20.0)
I_LMO         = np.radians(30.0)
THETA0_LMO    = np.radians(60.0)
THETA_DOT_LMO = np.sqrt(MU_MARS / R_LMO**3)   # 0.000884797 rad/s

# ── Elementary rotations ─────────────────────────────────────────────
def R1(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,s],[0,-s,c]])

def R3(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,s,0],[-s,c,0],[0,0,1]])

def dcm_HN(t):
    """[HN](t) via (3-1-3) Euler sequence for LMO."""
    theta = THETA0_LMO + THETA_DOT_LMO * t
    return R3(theta) @ R1(I_LMO) @ R3(OMEGA_LMO)

# ════════════════════════════════════════════════════════════════════
#  ANALYTIC DCM  [RnN](t)
# ════════════════════════════════════════════════════════════════════
_FLIP = np.diag([-1.0, 1.0, -1.0])

def dcm_RnN(t):
    """
    Nadir-pointing reference frame DCM [RnN](t).

        [RnN] = diag(−1, 1, −1) @ [HN](t)

    Rows:
        row 0 : r̂₁ = −î_r   (nadir — toward Mars center)
        row 1 : r̂₂ =  î_θ   (along-track)
        row 2 : r̂₃ = −î_h   (completes right-handed triad)

    Parameters
    ----------
    t : float — time (s)

    Returns
    -------
    RnN : (3,3) ndarray — proper DCM (det = +1)
    """
    return _FLIP @ dcm_HN(t)

# ════════════════════════════════════════════════════════════════════
#  ANGULAR VELOCITY  NωRn/N
# ════════════════════════════════════════════════════════════════════
def omega_RnN_inertial(t):
    """
    NωRn/N expressed in N-frame components.

    Rn co-rotates with the Hill frame at rate θ̇ about î_h:
        NωRn/N = θ̇ · î_h = θ̇ · [HN][2, :]

    Parameters
    ----------
    t : float — time (s)

    Returns
    -------
    omega_N : (3,) ndarray — rad/s in N-frame
    """
    return THETA_DOT_LMO * dcm_HN(t)[2, :]

def omega_RnN_body(t=None):
    """
    NωRn/N expressed in the Rn body frame.
    Since r̂₃ = −î_h:  î_h = −r̂₃  →  RnωRn/N = [0, 0, −θ̇]
    """
    return np.array([0.0, 0.0, -THETA_DOT_LMO])

def omega_RnN_numerical(t, dt=1e-5):
    """
    Finite-difference verification.
    Uses kinematic relation:  [NH_dot] = [ω̃_N] [NH]
    → [ω̃_N] = [NH_dot] @ [NH]ᵀ = [NH_dot] @ [RnN]
    Extract ω from skew-symmetric tilde matrix.
    """
    NH_dot = (dcm_RnN(t + dt) - dcm_RnN(t - dt)).T / (2.0 * dt)
    RnN    = dcm_RnN(t)
    omega_tilde_N = NH_dot @ RnN
    return np.array([omega_tilde_N[2,1],
                     omega_tilde_N[0,2],
                     omega_tilde_N[1,0]])

# ════════════════════════════════════════════════════════════════════
#  VALIDATION
# ════════════════════════════════════════════════════════════════════
def validate(t=330.0):
    print("="*64)
    print("ASEN 5010  —  Task 4: Nadir-Pointing Reference Frame")
    print("="*64)

    print("\nAnalytic construction:")
    print("  r̂₁ = −î_r                         (nadir → Mars center)")
    print("  r̂₂ =  î_θ                         (along-track, given)")
    print("  r̂₃ = r̂₁ × r̂₂ = (−î_r)×î_θ = −î_h  (RH triad)")
    print()
    print("  [RnN](t) = diag(−1, 1, −1) · [HN](t)")
    print()
    print("  NωRn/N = θ̇ · î_h    [N-frame]")
    print("  RnωRn/N = [0, 0, −θ̇]  [Rn body frame]")
    print(f"  θ̇ = {THETA_DOT_LMO:.9f} rad/s")

    # ── [RnN] at t ────────────────────────────────────────────
    RnN     = dcm_RnN(t)
    HN_t    = dcm_HN(t)
    theta_t = np.degrees(THETA0_LMO + THETA_DOT_LMO * t)

    print(f"\n{'─'*64}")
    print(f"[RnN] at t = {t:.0f} s   (θ = {theta_t:.6f}°)")
    print(f"{'─'*64}")
    hdr = f"  {'':14s}  {'n̂₁':>12s}  {'n̂₂':>12s}  {'n̂₃':>12s}"
    print(hdr)
    lbl = ["r̂₁ = −î_r", "r̂₂ =  î_θ", "r̂₃ = −î_h"]
    for i in range(3):
        r = RnN[i]
        print(f"  {lbl[i]:14s}  {r[0]:12.8f}  {r[1]:12.8f}  {r[2]:12.8f}")

    # ── Angular velocity ──────────────────────────────────────
    omega_N   = omega_RnN_inertial(t)
    omega_B   = omega_RnN_body(t)
    omega_num = omega_RnN_numerical(t)

    print(f"\n{'─'*64}")
    print(f"NωRn/N at t = {t:.0f} s")
    print(f"{'─'*64}")
    print(f"  Analytic  (N-frame) : [{omega_N[0]:12.9f},"
          f" {omega_N[1]:12.9f}, {omega_N[2]:12.9f}]  rad/s")
    print(f"  Numerical (N-frame) : [{omega_num[0]:12.9f},"
          f" {omega_num[1]:12.9f}, {omega_num[2]:12.9f}]  rad/s")
    print(f"  Body-frame  (Rn)    : [{omega_B[0]:12.9f},"
          f" {omega_B[1]:12.9f}, {omega_B[2]:12.9f}]  rad/s")
    print(f"  |NωRn/N|            = {np.linalg.norm(omega_N):.9f} rad/s")
    print(f"  (should equal θ̇  =  {THETA_DOT_LMO:.9f} rad/s)")

    diff_omega = np.max(np.abs(omega_N - omega_num))
    print(f"\n  Analytic vs numeric diff : {diff_omega:.2e}  " +
          ("✓" if diff_omega < 1e-9 else "✗"))

    # ── Sanity checks ─────────────────────────────────────────
    print(f"\n{'─'*64}")
    print("Sanity checks:")
    i_r_N, i_th_N, i_h_N = HN_t[0], HN_t[1], HN_t[2]

    checks = [
        ("r̂₁ = −î_r",               np.allclose(RnN[0], -i_r_N)),
        ("r̂₂ =  î_θ",               np.allclose(RnN[1],  i_th_N)),
        ("r̂₃ = −î_h",               np.allclose(RnN[2], -i_h_N)),
        ("r̂₃ = r̂₁×r̂₂  (RH rule)", np.allclose(np.cross(RnN[0], RnN[1]), RnN[2])),
        (f"Orthog. error {np.max(np.abs(RnN@RnN.T-np.eye(3))):.1e}",
                                     np.max(np.abs(RnN@RnN.T-np.eye(3))) < 1e-14),
        (f"det([RnN]) = {np.linalg.det(RnN):+.8f}",
                                     np.isclose(np.linalg.det(RnN), 1.0)),
        ("|NωRn/N| = θ̇",            np.isclose(np.linalg.norm(omega_N), THETA_DOT_LMO)),
    ]
    for name, ok in checks:
        print(f"  {name:38s} : " + ("✓" if ok else "✗ FAIL"))

    print("="*64)

if __name__ == "__main__":
    validate(t=330.0)
