import numpy as np

# ── Elementary rotations (reused from earlier tasks) ─────────────────
def R1(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,s],[0,-s,c]])

def R3(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,s,0],[-s,c,0],[0,0,1]])

# ════════════════════════════════════════════════════════════════════
#  ANALYTIC DERIVATION  (shown step-by-step)
# ════════════════════════════════════════════════════════════════════
def derive_RsN():
    """
    Build [RsN] from first principles using the given axis constraints.
    Returns the constant 3×3 DCM.
    """
    # Step 1 — assign r̂₁ and r̂₃ from mission requirement
    r1_N = np.array([-1.0,  0.0,  0.0])   # r̂₁ = −n̂₁
    r3_N = np.array([ 0.0,  1.0,  0.0])   # r̂₃ =  n̂₂  (sun direction)

    # Step 2 — r̂₂ completes right-handed triad: r̂₂ = r̂₃ × r̂₁
    r2_N = np.cross(r3_N, r1_N)           # = n̂₂ × (−n̂₁) = n̂₃

    # Step 3 — rows of [RsN] are the Rs basis vectors in N-frame coords
    RsN = np.array([r1_N, r2_N, r3_N])
    return RsN

# ════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION  — [RsN](t)
# ════════════════════════════════════════════════════════════════════
def dcm_RsN(t=None):
    """
    Return the sun-pointing reference frame DCM [RsN].

    Because the sun direction (n̂₂) and r̂₁ = −n̂₁ are both fixed
    inertial vectors, [RsN] is time-invariant (constant).

    Parameters
    ----------
    t : float or None — time (s); accepted for API consistency but unused.

    Returns
    -------
    RsN : (3,3) ndarray  — constant DCM
    """
    return np.array([[-1.0,  0.0,  0.0],   # row 1 : r̂₁ = −n̂₁
                     [ 0.0,  0.0,  1.0],   # row 2 : r̂₂ =  n̂₃
                     [ 0.0,  1.0,  0.0]])  # row 3 : r̂₃ =  n̂₂ (sun)

# ════════════════════════════════════════════════════════════════════
#  ANGULAR VELOCITY  NωRs/N
# ════════════════════════════════════════════════════════════════════
def omega_RsN(t=None):
    """
    Angular velocity of Rs w.r.t. N expressed in N-frame components.

    [RsN] is constant  →  d[RsN]/dt = 0
    The kinematic relation  [Ṙ] = [ω̃][R]  then gives  ω = 0.

    Returns
    -------
    omega : (3,) ndarray  — [0, 0, 0]  rad/s
    """
    return np.zeros(3)

# ════════════════════════════════════════════════════════════════════
#  VALIDATION & CHECKS
# ════════════════════════════════════════════════════════════════════
def validate():
    print("="*62)
    print("ASEN 5010  —  Task 3: Sun-Pointing Reference Frame")
    print("="*62)

    # ── Analytic derivation printout ──────────────────────────
    print("\nStep-by-step construction:")
    print("  r̂₁ = −n̂₁           →  N-components: [-1,  0,  0]")
    print("  r̂₃ =  n̂₂  (sun)    →  N-components: [ 0,  1,  0]")
    print("  r̂₂ = r̂₃ × r̂₁")
    print("      = n̂₂ × (−n̂₁)")
    print("      = −(n̂₂ × n̂₁)")
    print("      = −(−n̂₃) = n̂₃  →  N-components: [ 0,  0,  1]")

    print("\nAnalytic [RsN]  (constant — time-invariant):")
    print("         ⎡ -1   0   0 ⎤   ← r̂₁ = −n̂₁")
    print("[RsN] =  ⎢  0   0   1 ⎥   ← r̂₂ =  n̂₃")
    print("         ⎣  0   1   0 ⎦   ← r̂₃ =  n̂₂  (sun)")

    # ── Numerical evaluation at t = 0 ────────────────────────
    t_val = 0.0
    RsN   = dcm_RsN(t_val)
    omega = omega_RsN(t_val)

    print(f"\nNumerical [RsN] at t = {t_val:.0f} s:")
    hdr = f"  {'':12s}  {'n̂₁':>10s}  {'n̂₂':>10s}  {'n̂₃':>10s}"
    print(hdr)
    labels = ["r̂₁ (row 1)", "r̂₂ (row 2)", "r̂₃ (row 3)"]
    for i, lbl in enumerate(labels):
        r = RsN[i]
        print(f"  {lbl:12s}  {r[0]:10.6f}  {r[1]:10.6f}  {r[2]:10.6f}")

    print(f"\nAngular velocity  NωRs/N = {omega}  rad/s")
    print("  (Constant DCM → zero angular velocity)")

    # ── Sanity checks ─────────────────────────────────────────
    print("\nSanity checks:")

    # r̂₃ should point at sun (n̂₂)
    sun_dir = np.array([0., 1., 0.])
    r3_ok   = np.allclose(RsN[2], sun_dir)
    print(f"  r̂₃ ‖ n̂₂ (sun)          : {r3_ok} ✓" if r3_ok
          else f"  r̂₃ ‖ n̂₂ (sun)          : FAIL ✗")

    # r̂₁ should equal −n̂₁
    r1_ok = np.allclose(RsN[0], [-1, 0, 0])
    print(f"  r̂₁ = −n̂₁               : {r1_ok} ✓" if r1_ok
          else f"  r̂₁ = −n̂₁               : FAIL ✗")

    # Right-hand rule: r̂₂ = r̂₃ × r̂₁
    rh_check = np.allclose(np.cross(RsN[2], RsN[0]), RsN[1])
    print(f"  r̂₂ = r̂₃ × r̂₁ (RH rule) : {rh_check} ✓" if rh_check
          else f"  r̂₂ = r̂₃ × r̂₁ (RH rule) : FAIL ✗")

    # Orthogonality
    I_err = np.max(np.abs(RsN @ RsN.T - np.eye(3)))
    orth_ok = I_err < 1e-14
    print(f"  Orthogonality error     : {I_err:.2e}  " +
          ("✓" if orth_ok else "✗"))

    # Determinant = +1
    det_val = np.linalg.det(RsN)
    det_ok  = np.isclose(det_val, 1.0)
    print(f"  det([RsN])              : {det_val:.10f}  " +
          ("✓" if det_ok else "✗"))

    # Time-invariance: DCM same at t=0 and t=9999
    ti_ok = np.allclose(dcm_RsN(0), dcm_RsN(9999))
    print(f"  Time-invariant (t=0 vs t=9999): {ti_ok} ✓" if ti_ok
          else f"  Time-invariant: FAIL ✗")

    print("="*62)

if __name__ == "__main__":
    validate()
