import numpy as np

# ════════════════════════════════════════════════════════════════════
#  MRP  ↔  DCM  UTILITIES
# ════════════════════════════════════════════════════════════════════

def mrp_to_dcm(sigma):
    """
    Convert MRP vector σ (3,) to DCM [BN] using the Cayley transform.

        s  = |σ|²
        [BN] = I + (8[σ̃]² − 4(1−s)[σ̃]) / (1+s)²

    Parameters
    ----------
    sigma : (3,) array — MRP vector

    Returns
    -------
    C : (3,3) ndarray — DCM
    """
    s1, s2, s3 = sigma
    s_sq = s1**2 + s2**2 + s3**2
    denom = (1.0 + s_sq)**2

    # Skew-symmetric tilde matrix
    S = np.array([[ 0,  -s3,  s2],
                  [ s3,  0,  -s1],
                  [-s2,  s1,   0]])

    C = np.eye(3) + (8.0 * S @ S - 4.0 * (1.0 - s_sq) * S) / denom
    return C


def dcm_to_mrp(C):
    """
    Convert DCM to MRP using the Shepperd / Euler-parameter method.

    Euler parameters (quaternion):
        q0 = 0.5 * sqrt(1 + tr(C))          — scalar part
        q  = [C32-C23, C13-C31, C21-C12] / (4*q0)

    MRP:
        σ = q / (1 + q0)

    Always returns the SHORT rotation set: |σ| ≤ 1.

    Parameters
    ----------
    C : (3,3) ndarray — DCM

    Returns
    -------
    sigma : (3,) ndarray — MRP (short rotation)
    """
    tr = np.trace(C)
    # Guard against numerical noise pushing tr slightly above 3
    q0 = 0.5 * np.sqrt(max(0.0, 1.0 + tr))

    if q0 > 1e-10:
        q_vec = np.array([C[1,2] - C[2,1],
                          C[2,0] - C[0,2],
                          C[0,1] - C[1,0]]) / (4.0 * q0)
    else:
        # q0 ≈ 0  (180° rotation) — use largest diagonal element
        # to find dominant quaternion component
        q_sq = np.array([C[0,0], C[1,1], C[2,2]])
        idx  = np.argmax(q_sq)
        # fallback: small-angle limit sigma ≈ q_vec
        q_vec = np.array([C[1,2] - C[2,1],
                          C[2,0] - C[0,2],
                          C[0,1] - C[1,0]])
        q_vec /= (np.linalg.norm(q_vec) + 1e-16)
        q_vec *= 0.9999    # near |σ|=1 boundary

    sigma = q_vec / (1.0 + q0)

    # Switch to shadow set if |σ| > 1 (ensure short rotation)
    if np.dot(sigma, sigma) > 1.0:
        sigma = -sigma / np.dot(sigma, sigma)

    return sigma


def mrp_shadow(sigma):
    """Return the shadow MRP set: σ* = −σ / |σ|²"""
    s_sq = np.dot(sigma, sigma)
    return -sigma / s_sq

# ════════════════════════════════════════════════════════════════════
#  CORE ERROR FUNCTION
# ════════════════════════════════════════════════════════════════════

def attitude_error(t, sigma_BN, omega_BN_B, RN, omega_RN_N):
    """
    Compute MRP attitude error and angular velocity tracking error.

    Parameters
    ----------
    t          : float      — current time (s), unused here but kept for API
    sigma_BN   : (3,) array — MRP of B w.r.t. N
    omega_BN_B : (3,) array — BωB/N expressed in B-frame (rad/s)
    RN         : (3,3) array — DCM [RN] mapping N-frame to R-frame
    omega_RN_N : (3,) array — NωR/N expressed in N-frame (rad/s)

    Returns
    -------
    sigma_BR   : (3,) ndarray — MRP attitude error (short rotation)
    omega_BR_B : (3,) ndarray — BωB/R in B-frame (rad/s)

    Derivation
    ----------
    1. Attitude error DCM:
           [BR] = [BN] @ [RN].T
       This follows from:  B/R = (B/N) composed with (N/R) = (R/N)⁻¹

    2. MRP from [BR]:
           σ_B/R = dcm_to_mrp([BR])

    3. Angular velocity error (body frame):
           BωB/R = BωB/N − [BR] @ RωR/N
                 = BωB/N − [BN] @ NωR/N

       Here we use the second form (N-frame ω_R/N rotated to B by [BN]):
           BωR/N = [BN] @ NωR/N
           BωB/R = BωB/N − BωR/N
    """
    # Step 1: build [BN] from σ_B/N
    BN = mrp_to_dcm(sigma_BN)

    # Step 2: attitude error DCM  [BR] = [BN] @ [NR] = [BN] @ [RN].T
    BR = BN @ RN.T

    # Step 3: MRP from [BR]  (always short rotation)
    sigma_BR = dcm_to_mrp(BR)

    # Step 4: angular velocity error in B-frame
    #   BωR/N = [BN] @ NωR/N   (rotate N-frame ω vector into B-frame)
    omega_RN_B = BN @ omega_RN_N
    omega_BR_B = omega_BN_B - omega_RN_B

    return sigma_BR, omega_BR_B

# ════════════════════════════════════════════════════════════════════
#  IMPORT REFERENCE FRAME FUNCTIONS FROM EARLIER TASKS
# ════════════════════════════════════════════════════════════════════
# ── Constants ────────────────────────────────────────────────────────
MU_MARS        = 42828.3;  R_MARS = 3396.19
R_LMO          = R_MARS + 400.0
OMEGA_LMO      = np.radians(20.0);  I_LMO = np.radians(30.0)
THETA0_LMO     = np.radians(60.0)
THETA_DOT_LMO  = np.sqrt(MU_MARS / R_LMO**3)
R_GMO          = 20424.2
OMEGA_GMO      = np.radians(0.0);   I_GMO = np.radians(0.0)
THETA0_GMO     = np.radians(250.0)
THETA_DOT_GMO  = np.sqrt(MU_MARS / R_GMO**3)
N3             = np.array([0.,0.,1.])

def _R1(a):
    c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,s],[0,-s,c]])
def _R3(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,s,0],[-s,c,0],[0,0,1]])
def _dcm_HN(t):
    th=THETA0_LMO+THETA_DOT_LMO*t; return _R3(th)@_R1(I_LMO)@_R3(OMEGA_LMO)
def _pos(r0,Om,inc,th0,thd,t):
    th=th0+thd*t; HN=_R3(th)@_R1(inc)@_R3(Om); return HN.T@np.array([r0,0.,0.])

# ── Task 3: Sun-pointing ──────────────────────────────────────────────
def dcm_RsN(t=None):
    return np.array([[-1.,0.,0.],[0.,0.,1.],[0.,1.,0.]])
def omega_RsN(t=None):
    return np.zeros(3)

# ── Task 4: Nadir-pointing ────────────────────────────────────────────
def dcm_RnN(t):
    return np.diag([-1.,1.,-1.]) @ _dcm_HN(t)
def omega_RnN(t):
    return THETA_DOT_LMO * _dcm_HN(t)[2,:]

# ── Task 5: GMO-pointing ──────────────────────────────────────────────
def dcm_RcN(t):
    rl=_pos(R_LMO,OMEGA_LMO,I_LMO,THETA0_LMO,THETA_DOT_LMO,t)
    rg=_pos(R_GMO,OMEGA_GMO,I_GMO,THETA0_GMO,THETA_DOT_GMO,t)
    dr=rg-rl
    r1=-dr/np.linalg.norm(dr)
    c2=np.cross(dr,N3); r2=c2/np.linalg.norm(c2)
    r3=np.cross(r1,r2)
    return np.array([r1,r2,r3])

def omega_RcN(t, dt=1.0):
    NHd = (dcm_RcN(t+dt) - dcm_RcN(t-dt)).T / (2.*dt)
    RcN = dcm_RcN(t)
    Ot  = NHd @ RcN
    return np.array([Ot[2,1], Ot[0,2], Ot[1,0]])

# ════════════════════════════════════════════════════════════════════
#  VALIDATION
# ════════════════════════════════════════════════════════════════════
def validate():
    # ── Initial conditions (Section 3.1) ─────────────────────
    t0        = 0.0
    sigma_BN0 = np.array([ 0.3, -0.4,  0.5])
    omega_BN0 = np.radians(np.array([1.00, 1.75, -2.20]))  # convert deg/s → rad/s

    print("="*66)
    print("ASEN 5010  —  Task 6: Attitude Error Evaluation")
    print("="*66)
    print(f"\nInitial conditions (t₀ = {t0} s):")
    print(f"  σ_B/N  = {sigma_BN0}")
    print(f"  BωB/N  = {np.degrees(omega_BN0)} deg/s")
    print(f"         = {omega_BN0} rad/s")

    # ── Verify MRP↔DCM round-trip ────────────────────────────
    BN_check = mrp_to_dcm(sigma_BN0)
    sig_check = dcm_to_mrp(BN_check)
    rtrip_err = np.max(np.abs(sig_check - sigma_BN0))
    print(f"\nMRP ↔ DCM round-trip error: {rtrip_err:.2e}  " +
          ("✓" if rtrip_err < 1e-14 else "✗"))

    # ── Three reference frames ────────────────────────────────
    scenarios = [
        ("Sun-pointing  (Rs)",  dcm_RsN(t0),  omega_RsN(t0)),
        ("Nadir-pointing (Rn)", dcm_RnN(t0),  omega_RnN(t0)),
        ("GMO-pointing  (Rc)",  dcm_RcN(t0),  omega_RcN(t0)),
    ]

    results = {}
    for name, RN, omega_RN_N in scenarios:
        sig_BR, om_BR = attitude_error(t0, sigma_BN0, omega_BN0, RN, omega_RN_N)
        results[name] = (sig_BR, om_BR)

        print(f"\n{'─'*66}")
        print(f"  Scenario: {name}")
        print(f"{'─'*66}")
        print(f"  [RN] =")
        for row in RN:
            print(f"    [{row[0]:10.6f}  {row[1]:10.6f}  {row[2]:10.6f}]")
        print(f"  NωR/N = [{omega_RN_N[0]:.9f}, "
              f"{omega_RN_N[1]:.9f}, "
              f"{omega_RN_N[2]:.9f}]  rad/s")
        print()
        print(f"  σ_B/R  = [{sig_BR[0]:12.8f}, "
              f"{sig_BR[1]:12.8f}, "
              f"{sig_BR[2]:12.8f}]")
        print(f"  |σ_B/R| = {np.linalg.norm(sig_BR):.8f}  "
              f"({'short ✓' if np.linalg.norm(sig_BR)<=1 else 'LONG — use shadow ✗'})")
        print(f"  BωB/R  = [{om_BR[0]:12.8f}, "
              f"{om_BR[1]:12.8f}, "
              f"{om_BR[2]:12.8f}]  rad/s")
        print(f"         = [{np.degrees(om_BR[0]):10.6f}, "
              f"{np.degrees(om_BR[1]):10.6f}, "
              f"{np.degrees(om_BR[2]):10.6f}]  deg/s")

        # Consistency: |BωB/R| ≤ |BωB/N| + |NωR/N|  (triangle inequality)
        mag_BR  = np.linalg.norm(om_BR)
        mag_BN  = np.linalg.norm(omega_BN0)
        mag_RN  = np.linalg.norm(RN @ omega_RN_N)  # in B-frame magnitude
        tri_ok  = mag_BR <= mag_BN + mag_RN + 1e-10
        print(f"  Triangle ineq. |ωB/R|≤|ωB/N|+|ωR/N| : " +
              ("✓" if tri_ok else "✗"))

    print(f"\n{'='*66}")
    print("Summary Table")
    print(f"{'='*66}")
    print(f"  {'Scenario':<24s}  {'σ₁':>10s}  {'σ₂':>10s}  {'σ₃':>10s}")
    print(f"  {'':<24s}  {'ω₁ (r/s)':>10s}  {'ω₂ (r/s)':>10s}  {'ω₃ (r/s)':>10s}")
    print(f"  {'─'*24}  {'─'*10}  {'─'*10}  {'─'*10}")
    for name, (sig, om) in results.items():
        short = name[:24]
        print(f"  {short:<24s}  {sig[0]:10.6f}  {sig[1]:10.6f}  {sig[2]:10.6f}")
        print(f"  {'':24s}  {om[0]:10.6f}  {om[1]:10.6f}  {om[2]:10.6f}")
        print(f"  {'─'*24}  {'─'*10}  {'─'*10}  {'─'*10}")

    print("="*66)

if __name__ == "__main__":
    validate()
