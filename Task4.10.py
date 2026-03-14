"""
ASEN 5010 - Task 10: GMO-Pointing Attitude Control

Uses same K, P gains from Task 8.
Reference frame Rc from Task 5:
    r̂₁ = −Δr/|Δr|         (−r̂₁ points at GMO)
    r̂₂ = (Δr × n̂₃)/|Δr × n̂₃|
    r̂₃ = r̂₁ × r̂₂
Angular velocity NωRc/N via numerical finite difference (Task 5).
"""

import numpy as np
import matplotlib.pyplot as plt

# ════════════════════════════════════════════════════════════════════
#  SPACECRAFT & ORBIT PARAMETERS
# ════════════════════════════════════════════════════════════════════
I_sc   = np.diag([10.0, 5.0, 7.5])
I_inv  = np.linalg.inv(I_sc)

SIGMA0 = np.array([ 0.3, -0.4,  0.5])
OMEGA0 = np.radians(np.array([1.00, 1.75, -2.20]))

MU_MARS        = 42828.3
R_MARS         = 3396.19

# LMO
R_LMO          = R_MARS + 400.0
OMEGA_LMO      = np.radians(20.0)
I_LMO          = np.radians(30.0)
THETA0_LMO     = np.radians(60.0)
THETA_DOT_LMO  = np.sqrt(MU_MARS / R_LMO**3)

# GMO
R_GMO          = 20424.2
OMEGA_GMO      = np.radians(0.0)
I_GMO          = np.radians(0.0)
THETA0_GMO     = np.radians(250.0)
THETA_DOT_GMO  = np.sqrt(MU_MARS / R_GMO**3)

N3 = np.array([0., 0., 1.])

# ── Gains from Task 8 ─────────────────────────────────────────────
P_GAIN = 2.0 * 10.0 / 120.0        # 1/6   N·m·s
K_GAIN = P_GAIN**2 / (4.0 * 5.0)   # 1/720 N·m

# ════════════════════════════════════════════════════════════════════
#  UTILITIES
# ════════════════════════════════════════════════════════════════════
def skew(v):
    return np.array([[ 0,   -v[2],  v[1]],
                     [ v[2], 0,    -v[0]],
                     [-v[1], v[0],  0   ]])

def mrp_switch(s):
    return -s/np.dot(s,s) if np.dot(s,s) > 1.0 else s.copy()

def mrp_B(s):
    s2 = np.dot(s, s)
    return (1-s2)*np.eye(3) + 2*skew(s) + 2*np.outer(s, s)

def mrp_to_dcm(s):
    s2 = np.dot(s, s); S = skew(s)
    return np.eye(3) + (8*S@S - 4*(1-s2)*S) / (1+s2)**2

def dcm_to_mrp(C):
    tr = np.trace(C)
    q0 = 0.5 * np.sqrt(max(0., 1. + tr))
    if q0 > 1e-10:
        qv = np.array([C[1,2]-C[2,1], C[2,0]-C[0,2],
                       C[0,1]-C[1,0]]) / (4*q0)
    else:
        qv = np.array([C[1,2]-C[2,1], C[2,0]-C[0,2], C[0,1]-C[1,0]])
        qv = qv / (np.linalg.norm(qv) + 1e-16) * 0.9999
    s = qv / (1 + q0)
    return -s/np.dot(s,s) if np.dot(s,s) > 1 else s

# ════════════════════════════════════════════════════════════════════
#  ORBIT POSITION FUNCTIONS  (Tasks 1 & 5)
# ════════════════════════════════════════════════════════════════════
def _R1(a):
    c,s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,s],[0,-s,c]])

def _R3(a):
    c,s = np.cos(a), np.sin(a)
    return np.array([[c,s,0],[-s,c,0],[0,0,1]])

def pos_LMO(t):
    th  = THETA0_LMO + THETA_DOT_LMO * t
    HN  = _R3(th) @ _R1(I_LMO) @ _R3(OMEGA_LMO)
    return HN.T @ np.array([R_LMO, 0., 0.])

def pos_GMO(t):
    th  = THETA0_GMO + THETA_DOT_GMO * t
    HN  = _R3(th) @ _R1(I_GMO) @ _R3(OMEGA_GMO)
    return HN.T @ np.array([R_GMO, 0., 0.])

# ════════════════════════════════════════════════════════════════════
#  GMO-POINTING REFERENCE FRAME  (Task 5)
# ════════════════════════════════════════════════════════════════════
def dcm_RcN(t):
    """
    [RcN](t) — GMO-pointing reference frame.
        Δr  = r_GMO − r_LMO
        r̂₁  = −Δr/|Δr|            (−r̂₁ → GMO)
        r̂₂  = (Δr × n̂₃)/|Δr × n̂₃|
        r̂₃  = r̂₁ × r̂₂
    """
    dr  = pos_GMO(t) - pos_LMO(t)
    r1  = -dr / np.linalg.norm(dr)
    c2  = np.cross(dr, N3)
    r2  = c2 / np.linalg.norm(c2)
    r3  = np.cross(r1, r2)
    return np.array([r1, r2, r3])

def omega_RcN(t, dt=1.0):
    """
    NωRc/N via kinematic finite difference (Task 5).
        [ω̃_N] = [NcR_dot] @ [RcN]
        ω extracted from upper diagonal of skew-sym matrix.
    """
    NH_dot  = (dcm_RcN(t + dt) - dcm_RcN(t - dt)).T / (2.0 * dt)
    RcN     = dcm_RcN(t)
    Ot      = NH_dot @ RcN
    return np.array([Ot[2,1], Ot[0,2], Ot[1,0]])

# ════════════════════════════════════════════════════════════════════
#  ATTITUDE ERROR & PD CONTROL
# ════════════════════════════════════════════════════════════════════
def attitude_error(s_BN, w_BN, RN, w_RN_N):
    BN    = mrp_to_dcm(s_BN)
    s_BR  = dcm_to_mrp(BN @ RN.T)
    w_BR  = w_BN - BN @ w_RN_N
    return s_BR, w_BR

def pd_control(s_BR, w_BR):
    """Bu = −K σ_B/R − P BωB/R"""
    return -K_GAIN * s_BR - P_GAIN * w_BR

# ════════════════════════════════════════════════════════════════════
#  EOM & RK4
# ════════════════════════════════════════════════════════════════════
def eom(X, u):
    s, w  = X[:3], X[3:]
    sdot  = 0.25 * mrp_B(s) @ w
    wdot  = I_inv @ (-skew(w) @ I_sc @ w + u)
    return np.concatenate([sdot, wdot])

def rk4_step(X, dt, u):
    k1 = eom(X,           u)
    k2 = eom(X + dt/2*k1, u)
    k3 = eom(X + dt/2*k2, u)
    k4 = eom(X + dt*k3,   u)
    return X + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def simulate(t_end, dt=1.0):
    X   = np.concatenate([SIGMA0, OMEGA0])
    N   = int(round(t_end / dt))
    t_h = np.zeros(N+1)
    X_h = np.zeros((N+1, 6))
    u_h = np.zeros((N+1, 3))
    t_h[0] = 0.; X_h[0] = X

    for i in range(N):
        t_n       = i * dt
        RN        = dcm_RcN(t_n)
        oRN       = omega_RcN(t_n)
        sBR, oBR  = attitude_error(X[:3], X[3:], RN, oRN)
        u         = pd_control(sBR, oBR)
        u_h[i]    = u
        X         = rk4_step(X, dt, u)
        X[:3]     = mrp_switch(X[:3])
        t_h[i+1]  = (i+1)*dt
        X_h[i+1]  = X

    # last control
    RN = dcm_RcN(t_h[-1]); oRN = omega_RcN(t_h[-1])
    sBR, oBR = attitude_error(X[:3], X[3:], RN, oRN)
    u_h[-1]  = pd_control(sBR, oBR)
    return t_h, X_h, u_h

# ════════════════════════════════════════════════════════════════════
#  VALIDATION & PLOT
# ════════════════════════════════════════════════════════════════════
def validate():
    print("="*64)
    print("ASEN 5010  —  Task 10: GMO-Pointing PD Control")
    print("="*64)
    print(f"\n  K = {K_GAIN:.10f} N·m   (= 1/720)")
    print(f"  P = {P_GAIN:.10f} N·m·s (= 1/6)")
    print(f"  Reference: [RcN](t) — GMO frame (both orbits moving)")

    t_h, X_h, u_h = simulate(400.0)

    # ── Validation table ──────────────────────────────────────
    check = [15, 100, 200, 400]
    print(f"\n{'─'*64}")
    print("σ_B/N at required times (short MRP, |σ| ≤ 1):")
    print(f"  {'t(s)':>5}  {'σ₁':>12}  {'σ₂':>12}  "
          f"{'σ₃':>12}  {'|σ|':>10}")
    print(f"  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*10}")
    for tc in check:
        s   = X_h[tc, :3]
        mag = np.linalg.norm(s)
        flag = '✓' if mag <= 1.0 else '✗ shadow!'
        print(f"  {tc:5d}  {s[0]:12.8f}  {s[1]:12.8f}  "
              f"{s[2]:12.8f}  {mag:10.8f} {flag}")

    # ── Tracking error convergence ────────────────────────────
    print(f"\n{'─'*64}")
    print("Tracking error convergence:")
    print(f"  {'t(s)':>5}  {'|σ_B/R|':>12}  {'|BωB/R| (rad/s)':>18}")
    print(f"  {'─'*5}  {'─'*12}  {'─'*18}")
    for tc in check:
        s = X_h[tc,:3]; w = X_h[tc,3:]
        sBR, oBR = attitude_error(s, w, dcm_RcN(tc), omega_RcN(tc))
        print(f"  {tc:5d}  {np.linalg.norm(sBR):12.6f}  "
              f"{np.linalg.norm(oBR):18.9f}")

    # ── Plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(
        "Task 10: GMO-Pointing PD Control\n"
        f"K = {K_GAIN:.6f} N·m,   P = {P_GAIN:.6f} N·m·s",
        fontsize=13, fontweight='bold')

    cols  = ['#1f77b4', '#ff7f0e', '#2ca02c']
    lbl_s = [r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$']
    lbl_w = [r'$\omega_1$', r'$\omega_2$', r'$\omega_3$']
    lbl_u = [r'$u_1$',      r'$u_2$',      r'$u_3$']

    ax = axes[0]
    for j in range(3):
        ax.plot(t_h, X_h[:, j], color=cols[j], lw=1.8, label=lbl_s[j])
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_ylabel(r'$\sigma_{B/N}$', fontsize=12)
    ax.legend(ncol=3, fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.35)
    ax.set_title(r'MRP Attitude  $\sigma_{B/N}$', fontsize=11)

    ax = axes[1]
    for j in range(3):
        ax.plot(t_h, np.degrees(X_h[:, j+3]),
                color=cols[j], lw=1.8, label=lbl_w[j])
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_ylabel(r'$^B\omega_{B/N}$ (deg/s)', fontsize=12)
    ax.legend(ncol=3, fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.35)
    ax.set_title('Angular Velocity', fontsize=11)

    ax = axes[2]
    for j in range(3):
        ax.plot(t_h, u_h[:, j], color=cols[j], lw=1.8, label=lbl_u[j])
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_ylabel(r'$\mathbf{u}$ (N·m)', fontsize=12)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.legend(ncol=3, fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.35)
    ax.set_title('PD Control Torque', fontsize=11)

    for ax in axes:
        for tc in check:
            ax.axvline(tc, color='gray', lw=0.9, ls=':', alpha=0.7)

    plt.tight_layout()
    plt.show()

    print("="*64)

if __name__ == "__main__":
    validate()
