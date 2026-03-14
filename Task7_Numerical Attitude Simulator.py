import numpy as np

# ════════════════════════════════════════════════════════════════════
#  SPACECRAFT PARAMETERS  (Section 3.1)
# ════════════════════════════════════════════════════════════════════
I_sc   = np.diag([10.0, 5.0, 7.5])          # kg·m²  inertia tensor
I_inv  = np.linalg.inv(I_sc)

# Initial conditions
SIGMA0 = np.array([ 0.3, -0.4,  0.5])
OMEGA0 = np.radians(np.array([1.00, 1.75, -2.20]))  # rad/s

# ════════════════════════════════════════════════════════════════════
#  MRP UTILITIES
# ════════════════════════════════════════════════════════════════════
def skew(v):
    """3×3 skew-symmetric (tilde) matrix of vector v."""
    return np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]])

def mrp_shadow(sigma):
    """Shadow MRP set: σ* = −σ/|σ|²"""
    return -sigma / np.dot(sigma, sigma)

def mrp_switch(sigma):
    """Switch to shadow set if |σ| > 1 (keep short rotation)."""
    if np.dot(sigma, sigma) > 1.0:
        return mrp_shadow(sigma)
    return sigma.copy()

def mrp_kinematic_matrix(sigma):
    """
    MRP kinematic matrix B(σ) such that σ̇ = (1/4) B(σ) ω

        B(σ) = (1 - |σ|²)I + 2[σ̃] + 2σσᵀ
    """
    s  = sigma
    s2 = np.dot(s, s)
    return (1.0 - s2) * np.eye(3) + 2.0 * skew(s) + 2.0 * np.outer(s, s)

def mrp_to_dcm(sigma):
    """MRP → DCM via Cayley transform."""
    s2    = np.dot(sigma, sigma)
    S     = skew(sigma)
    denom = (1.0 + s2)**2
    return np.eye(3) + (8.0 * S @ S - 4.0 * (1.0 - s2) * S) / denom

# ════════════════════════════════════════════════════════════════════
#  EQUATIONS OF MOTION
# ════════════════════════════════════════════════════════════════════
def eom(X, t, u):
    """
    State derivative  Ẋ = f(X, t, u)

    Parameters
    ----------
    X : (6,) array  — [σ (3), ω (3)]
    t : float       — time (s)  [not used for torque-free / const-u]
    u : (3,) array  — control torque in B-frame (N·m)

    Returns
    -------
    Xdot : (6,) array
    """
    sigma = X[:3]
    omega = X[3:]

    # MRP kinematics:  σ̇ = (1/4) B(σ) ω
    B      = mrp_kinematic_matrix(sigma)
    sdot   = 0.25 * B @ omega

    # Euler's equation:  ω̇ = I⁻¹(−[ω̃][I]ω + u)
    Iw     = I_sc @ omega
    omegadot = I_inv @ (-skew(omega) @ Iw + u)

    return np.concatenate([sdot, omegadot])

# ════════════════════════════════════════════════════════════════════
#  RK4 INTEGRATOR
# ════════════════════════════════════════════════════════════════════
def rk4_step(X, t, dt, u):
    """
    Single RK4 step.  Control u is held piecewise-constant over [t, t+dt].

    k1 = f(Xn,       tn)
    k2 = f(Xn+k1/2,  tn+dt/2)
    k3 = f(Xn+k2/2,  tn+dt/2)
    k4 = f(Xn+k3,    tn+dt)
    Xn+1 = Xn + (dt/6)(k1 + 2k2 + 2k3 + k4)
    """
    k1 = eom(X,            t,        u)
    k2 = eom(X + dt/2*k1,  t + dt/2, u)
    k3 = eom(X + dt/2*k2,  t + dt/2, u)
    k4 = eom(X + dt*k3,    t + dt,   u)
    return X + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate(X0, t_end, dt=1.0, u_func=None, store=True):
    """
    Simulate attitude motion from t=0 to t=t_end.

    Parameters
    ----------
    X0     : (6,) array      — initial state [σ, ω]
    t_end  : float           — simulation end time (s)
    dt     : float           — fixed time step (s)
    u_func : callable(t,X)   — returns (3,) control torque; default zero
    store  : bool            — whether to store full history

    Returns
    -------
    t_hist : (N,) array
    X_hist : (N,6) array   (only if store=True, else last state)
    """
    if u_func is None:
        u_func = lambda t, X: np.zeros(3)

    N  = int(round(t_end / dt))
    X  = X0.copy()
    t  = 0.0

    if store:
        t_hist = np.zeros(N + 1)
        X_hist = np.zeros((N + 1, 6))
        t_hist[0] = t
        X_hist[0] = X

    for i in range(N):
        u  = u_func(t, X)               # update control (piecewise-const)
        X  = rk4_step(X, t, dt, u)
        t  = (i + 1) * dt               # avoid floating-point drift

        # MRP shadow switching — done AFTER RK4, not during
        X[:3] = mrp_switch(X[:3])

        if store:
            t_hist[i+1] = t
            X_hist[i+1] = X

    if store:
        return t_hist, X_hist
    else:
        return t, X

# ════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC QUANTITIES
# ════════════════════════════════════════════════════════════════════
def angular_momentum_B(omega):
    """H = [I]ω  in body frame."""
    return I_sc @ omega

def angular_momentum_N(sigma, omega):
    """NH = [BN]ᵀ [I]ω  in inertial frame."""
    BN = mrp_to_dcm(sigma)
    return BN.T @ (I_sc @ omega)

def kinetic_energy(omega):
    """T = (1/2) ωᵀ [I] ω"""
    return 0.5 * omega @ I_sc @ omega

# ════════════════════════════════════════════════════════════════════
#  VALIDATION
# ════════════════════════════════════════════════════════════════════
def validate():
    X0 = np.concatenate([SIGMA0, OMEGA0])

    print("="*64)
    print("ASEN 5010  —  Task 7: Numerical Attitude Simulator (RK4)")
    print("="*64)
    print(f"\nInitial state:")
    print(f"  σ_B/N(0) = {SIGMA0}")
    print(f"  BωB/N(0) = {np.degrees(OMEGA0)} deg/s")
    print(f"           = {OMEGA0} rad/s")
    print(f"\nInertia tensor [I] = diag(10, 5, 7.5) kg·m²")

    # ────────────────────────────────────────────────────────────
    # CASE 1: Torque-free  u = 0, 500 s
    # ────────────────────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print("CASE 1:  Torque-free  (u = 0),  t = 0 → 500 s,  dt = 1 s")
    print(f"{'═'*64}")

    t_hist, X_hist = simulate(X0, t_end=500.0, dt=1.0,
                               u_func=lambda t, X: np.zeros(3))
    X500   = X_hist[-1]
    sig500 = X500[:3]
    om500  = X500[3:]

    H_B500  = angular_momentum_B(om500)
    NH500   = angular_momentum_N(sig500, om500)
    T500    = kinetic_energy(om500)

    # Reference values at t=0 for conservation checks
    H_B0    = angular_momentum_B(OMEGA0)
    NH0     = angular_momentum_N(SIGMA0, OMEGA0)
    T0      = kinetic_energy(OMEGA0)

    print(f"\n  σ_B/N(500 s) = [{sig500[0]:12.8f}, "
          f"{sig500[1]:12.8f}, {sig500[2]:12.8f}]")
    print(f"\n  BH(500 s) = [I]ω  (body frame):")
    print(f"    = [{H_B500[0]:12.8f}, {H_B500[1]:12.8f}, {H_B500[2]:12.8f}]  kg·m²/s")
    print(f"\n  NH(500 s)  (inertial frame):")
    print(f"    = [{NH500[0]:12.8f}, {NH500[1]:12.8f}, {NH500[2]:12.8f}]  kg·m²/s")
    print(f"\n  T(500 s) = {T500:.10f}  J")

    # ── Conservation checks (torque-free → H and T conserved) ──
    print(f"\n  Conservation checks (torque-free):")
    NH_drift = np.linalg.norm(NH500 - NH0)
    T_drift  = abs(T500 - T0)
    T_rel    = T_drift / T0

    print(f"    |NH(500) − NH(0)| = {NH_drift:.4e}  kg·m²/s  " +
          ("✓" if NH_drift < 1e-8 else "✗"))
    print(f"    T(0) = {T0:.10f}  J")
    print(f"    |T(500) − T(0)|/T(0) = {T_rel:.4e}  " +
          ("✓" if T_rel < 1e-8 else "✗"))

    # ────────────────────────────────────────────────────────────
    # CASE 2: Constant control torque, 100 s
    # ────────────────────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print("CASE 2:  Bu = [0.01, −0.01, 0.02] N·m,  t = 0 → 100 s")
    print(f"{'═'*64}")

    u_const = np.array([0.01, -0.01, 0.02])
    u_func2 = lambda t, X: u_const

    _, X_hist2 = simulate(X0, t_end=100.0, dt=1.0, u_func=u_func2)
    X100   = X_hist2[-1]
    sig100 = X100[:3]
    om100  = X100[3:]

    print(f"\n  Applied torque Bu = {u_const}  N·m")
    print(f"\n  σ_B/N(100 s) = [{sig100[0]:12.8f}, "
          f"{sig100[1]:12.8f}, {sig100[2]:12.8f}]")
    print(f"  BωB/N(100 s) = [{np.degrees(om100[0]):.6f}, "
          f"{np.degrees(om100[1]):.6f}, "
          f"{np.degrees(om100[2]):.6f}]  deg/s")

    # ────────────────────────────────────────────────────────────
    # NUMERICAL QUALITY: dt sensitivity
    # ────────────────────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print("RK4 Step-Size Sensitivity Check  (torque-free, σ at 500 s)")
    print(f"{'═'*64}")
    for dt_test in [1.0, 0.5, 0.1]:
        _, Xh = simulate(X0, t_end=500.0, dt=dt_test,
                         u_func=lambda t, X: np.zeros(3))
        s = Xh[-1, :3]
        print(f"  dt={dt_test:.1f} s:  σ(500) = [{s[0]:.8f}, {s[1]:.8f}, {s[2]:.8f}]")

    print("="*64)

if __name__ == "__main__":
    validate()
