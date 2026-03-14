import numpy as np

# ── Constants (from Task 1) ──────────────────────────────────────────
MU_MARS       = 42828.3
R_MARS        = 3396.19
R_LMO         = R_MARS + 400.0          # 3796.19 km
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

# ────────────────────────────────────────────────────────────────────
# ANALYTIC DCM  [HN](t)
# ────────────────────────────────────────────────────────────────────
def dcm_HN_analytic(t):
    """
    [HN] = R3(theta(t)) @ R1(i) @ R3(Omega)
    Rows of [HN] are î_r, î_theta, î_h expressed in N-frame components.
    """
    theta = THETA0_LMO + THETA_DOT_LMO * t
    return R3(theta) @ R1(I_LMO) @ R3(OMEGA_LMO)

# ────────────────────────────────────────────────────────────────────
# NUMERICAL CROSS-CHECK  (build rows from r and r_dot vectors)
# ────────────────────────────────────────────────────────────────────
def orbit_rv(t):
    """Inertial position and velocity of the LMO satellite at time t."""
    theta   = THETA0_LMO + THETA_DOT_LMO * t
    HN      = R3(theta) @ R1(I_LMO) @ R3(OMEGA_LMO)
    NH      = HN.T
    N_r     = NH @ np.array([R_LMO, 0.0, 0.0])
    N_rdot  = NH @ np.array([0.0, R_LMO * THETA_DOT_LMO, 0.0])
    return N_r, N_rdot

def dcm_HN_numeric(t):
    """
    Build [HN] directly from r and r_dot (definition-based, no Euler angles).
        î_r  = r / |r|
        î_h  = (r × r_dot) / |r × r_dot|
        î_θ  = î_h × î_r
    Rows of [HN] = [î_r^T; î_θ^T; î_h^T]
    """
    r, rdot   = orbit_rv(t)
    i_r  = r / np.linalg.norm(r)
    h    = np.cross(r, rdot)
    i_h  = h / np.linalg.norm(h)
    i_th = np.cross(i_h, i_r)
    return np.array([i_r, i_th, i_h])   # rows

# ────────────────────────────────────────────────────────────────────
# EXPANDED ANALYTIC FORM (for report)
# ────────────────────────────────────────────────────────────────────
def print_analytic_form():
    """
    Expand [HN] = R3(θ) R1(i) R3(Ω) symbolically, showing the
    full 3×3 matrix entries in terms of Ω, i, θ.
    
    Row 1 (î_r):
        [cθ cΩ - sθ ci sΩ,   cθ sΩ + sθ ci cΩ,   sθ si]
    Row 2 (î_θ):
        [-sθ cΩ - cθ ci sΩ, -sθ sΩ + cθ ci cΩ,   cθ si]
    Row 3 (î_h):
        [si sΩ,              -si cΩ,               ci   ]
    """
    print("\nAnalytic [HN] (3-1-3 Euler sequence  Ω, i, θ):")
    print("─"*62)
    print("         | cθcΩ - sθci sΩ    cθsΩ + sθci cΩ    sθsi |")
    print("[HN]  =  |-sθcΩ - cθci sΩ  -sθsΩ + cθci cΩ    cθsi |")
    print("         | si sΩ            -si cΩ               ci   |")
    print()
    print("where  θ = θ₀ + θ̇·t  (θ̇ = const for circular orbit)")

# ────────────────────────────────────────────────────────────────────
# VALIDATION at t = 300 s
# ────────────────────────────────────────────────────────────────────
def validate(t=300.0):
    HN_a = dcm_HN_analytic(t)
    HN_n = dcm_HN_numeric(t)

    theta_t = np.degrees(THETA0_LMO + THETA_DOT_LMO * t)

    print("="*62)
    print("ASEN 5010  —  Task 2: Orbit Frame Orientation")
    print("="*62)

    print_analytic_form()

    print(f"\nValidation at t = {t:.0f} s")
    print(f"  θ(300 s) = θ₀ + θ̇·t = 60° + {np.degrees(THETA_DOT_LMO*t):.6f}°"
          f" = {theta_t:.6f}°")
    print()

    labels = ["î_r  (row 1)", "î_θ  (row 2)", "î_h  (row 3)"]
    header = f"  {'':18s}  {'n̂₁':>12s}  {'n̂₂':>12s}  {'n̂₃':>12s}"

    # ── Analytic result ───────────────────────────────────────
    print("Analytic [HN](300 s):")
    print(header)
    for i, lbl in enumerate(labels):
        row = HN_a[i]
        print(f"  {lbl:18s}  {row[0]:12.8f}  {row[1]:12.8f}  {row[2]:12.8f}")

    # ── Numeric cross-check ───────────────────────────────────
    print("\nNumerical cross-check [HN](300 s)  (built from r × ṙ):")
    print(header)
    for i, lbl in enumerate(labels):
        row = HN_n[i]
        print(f"  {lbl:18s}  {row[0]:12.8f}  {row[1]:12.8f}  {row[2]:12.8f}")

    # ── Difference ────────────────────────────────────────────
    diff = np.max(np.abs(HN_a - HN_n))
    print(f"\n  Max element-wise difference : {diff:.2e}  ", end="")
    print("✓ (< 1e-12)" if diff < 1e-12 else "✗ MISMATCH")

    # ── Orthogonality check ───────────────────────────────────
    I_err = np.max(np.abs(HN_a @ HN_a.T - np.eye(3)))
    print(f"  Orthogonality error [HN][HN]ᵀ - I : {I_err:.2e}  ", end="")
    print("✓" if I_err < 1e-14 else "✗")

    # ── det = +1 ─────────────────────────────────────────────
    det_val = np.linalg.det(HN_a)
    print(f"  det([HN])                          : {det_val:.10f}  ", end="")
    print("✓" if np.isclose(det_val, 1.0) else "✗")

    print("="*62)
    return HN_a

if __name__ == "__main__":
    HN = validate(t=300.0)
