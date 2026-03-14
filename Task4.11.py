import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════
I_sc   = np.diag([10.0, 5.0, 7.5])
I_inv  = np.linalg.inv(I_sc)

SIGMA0 = np.array([ 0.3, -0.4,  0.5])
OMEGA0 = np.radians(np.array([1.00, 1.75, -2.20]))

MU_MARS       = 42828.3;  R_MARS = 3396.19
R_LMO         = R_MARS + 400.0
OMEGA_LMO     = np.radians(20.0);  I_LMO = np.radians(30.0)
THETA0_LMO    = np.radians(60.0)
THETA_DOT_LMO = np.sqrt(MU_MARS / R_LMO**3)

R_GMO         = 20424.2
OMEGA_GMO     = np.radians(0.0);   I_GMO = np.radians(0.0)
THETA0_GMO    = np.radians(250.0)
THETA_DOT_GMO = np.sqrt(MU_MARS / R_GMO**3)

N3 = np.array([0., 0., 1.])

P_GAIN = 2.0 * 10.0 / 120.0        # 1/6   N·m·s
K_GAIN = P_GAIN**2 / (4.0 * 5.0)   # 1/720 N·m

GMO_VIS_ANGLE = np.radians(35.0)   # GMO visibility half-angle threshold

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
#  ORBIT POSITIONS
# ════════════════════════════════════════════════════════════════════
def _R1(a):
    c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,s],[0,-s,c]])
def _R3(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,s,0],[-s,c,0],[0,0,1]])

def pos_LMO(t):
    th = THETA0_LMO + THETA_DOT_LMO*t
    return (_R3(th)@_R1(I_LMO)@_R3(OMEGA_LMO)).T @ np.array([R_LMO,0.,0.])

def pos_GMO(t):
    th = THETA0_GMO + THETA_DOT_GMO*t
    return (_R3(th)@_R1(I_GMO)@_R3(OMEGA_GMO)).T @ np.array([R_GMO,0.,0.])

# ════════════════════════════════════════════════════════════════════
#  REFERENCE FRAMES
# ════════════════════════════════════════════════════════════════════
# --- Sun-pointing (Task 3, constant) ---
def dcm_RsN(t=None):
    return np.array([[-1.,0.,0.],[0.,0.,1.],[0.,1.,0.]])
def omega_RsN(t=None):
    return np.zeros(3)

# --- Nadir-pointing (Task 4) ---
def dcm_RnN(t):
    th = THETA0_LMO + THETA_DOT_LMO*t
    HN = _R3(th)@_R1(I_LMO)@_R3(OMEGA_LMO)
    return np.diag([-1.,1.,-1.]) @ HN
def omega_RnN(t):
    th = THETA0_LMO + THETA_DOT_LMO*t
    HN = _R3(th)@_R1(I_LMO)@_R3(OMEGA_LMO)
    return THETA_DOT_LMO * HN[2,:]

# --- GMO-pointing (Task 5) ---
def dcm_RcN(t):
    dr = pos_GMO(t) - pos_LMO(t)
    r1 = -dr/np.linalg.norm(dr)
    c2 = np.cross(dr, N3); r2=c2/np.linalg.norm(c2)
    return np.array([r1, r2, np.cross(r1,r2)])
def omega_RcN(t, dt=1.0):
    NH_dot = (dcm_RcN(t+dt)-dcm_RcN(t-dt)).T/(2*dt)
    Ot = NH_dot @ dcm_RcN(t)
    return np.array([Ot[2,1],Ot[0,2],Ot[1,0]])

# ════════════════════════════════════════════════════════════════════
#  MODE SELECTION LOGIC
# ════════════════════════════════════════════════════════════════════
MODE_SUN   = 0
MODE_GMO   = 1
MODE_NADIR = 2
MODE_NAMES = {MODE_SUN:'Sun', MODE_GMO:'GMO', MODE_NADIR:'Nadir'}
MODE_COLORS= {MODE_SUN:'#F0C040', MODE_GMO:'#5B8FCC', MODE_NADIR:'#5BAA6E'}

def select_mode(t):
    """
    Determine control mode at time t per Table 1:
      - Sunlit  (r_LMO · n̂₂ > 0)          → Sun-pointing
      - Shadow & GMO visible (angle < 35°)  → GMO-pointing
      - Shadow & GMO not visible             → Nadir-pointing
    """
    r_lmo = pos_LMO(t)
    # Sunlit check: n̂₂ component of LMO position > 0
    if r_lmo[1] > 0:
        return MODE_SUN

    # GMO visibility: angle between r_LMO and r_GMO
    r_gmo = pos_GMO(t)
    cos_a = np.dot(r_lmo, r_gmo) / (np.linalg.norm(r_lmo)*np.linalg.norm(r_gmo))
    cos_a = np.clip(cos_a, -1., 1.)
    angle = np.arccos(cos_a)
    if angle < GMO_VIS_ANGLE:
        return MODE_GMO

    return MODE_NADIR

def get_reference(t, mode):
    """Return (RN, omega_RN_N) for the given mode."""
    if mode == MODE_SUN:
        return dcm_RsN(t), omega_RsN(t)
    elif mode == MODE_GMO:
        return dcm_RcN(t), omega_RcN(t)
    else:
        return dcm_RnN(t), omega_RnN(t)

# ════════════════════════════════════════════════════════════════════
#  ATTITUDE ERROR & PD CONTROL
# ════════════════════════════════════════════════════════════════════
def attitude_error(s_BN, w_BN, RN, w_RN_N):
    BN   = mrp_to_dcm(s_BN)
    s_BR = dcm_to_mrp(BN @ RN.T)
    w_BR = w_BN - BN @ w_RN_N
    return s_BR, w_BR

def pd_control(s_BR, w_BR):
    return -K_GAIN*s_BR - P_GAIN*w_BR

# ════════════════════════════════════════════════════════════════════
#  EOM & RK4
# ════════════════════════════════════════════════════════════════════
def eom(X, u):
    s,w  = X[:3], X[3:]
    sdot = 0.25 * mrp_B(s) @ w
    wdot = I_inv @ (-skew(w) @ I_sc @ w + u)
    return np.concatenate([sdot, wdot])

def rk4_step(X, dt, u):
    k1=eom(X,        u); k2=eom(X+dt/2*k1,u)
    k3=eom(X+dt/2*k2,u); k4=eom(X+dt*k3,  u)
    return X+(dt/6)*(k1+2*k2+2*k3+k4)

# ════════════════════════════════════════════════════════════════════
#  FULL MISSION SIMULATION
# ════════════════════════════════════════════════════════════════════
def simulate_mission(t_end=6500.0, dt=1.0):
    X    = np.concatenate([SIGMA0, OMEGA0])
    N    = int(round(t_end/dt))
    t_h  = np.zeros(N+1)
    X_h  = np.zeros((N+1,6))
    u_h  = np.zeros((N+1,3))
    m_h  = np.zeros(N+1, dtype=int)   # mode history
    t_h[0]=0.; X_h[0]=X

    for i in range(N):
        t_n       = i*dt
        mode      = select_mode(t_n)
        RN, oRN   = get_reference(t_n, mode)
        sBR, oBR  = attitude_error(X[:3], X[3:], RN, oRN)
        u         = pd_control(sBR, oBR)
        u_h[i]    = u
        m_h[i]    = mode
        X         = rk4_step(X, dt, u)
        X[:3]     = mrp_switch(X[:3])
        t_h[i+1]  = (i+1)*dt
        X_h[i+1]  = X

    # last step
    mode = select_mode(t_h[-1])
    RN, oRN = get_reference(t_h[-1], mode)
    sBR,oBR = attitude_error(X[:3],X[3:],RN,oRN)
    u_h[-1] = pd_control(sBR,oBR)
    m_h[-1] = mode
    return t_h, X_h, u_h, m_h

# ════════════════════════════════════════════════════════════════════
#  VALIDATION & PLOT
# ════════════════════════════════════════════════════════════════════
def validate():
    print("="*66)
    print("ASEN 5010  —  Task 11: Full Mission Scenario Simulation")
    print("="*66)
    print(f"\n  Duration: 6500 s,  dt = 1 s")
    print(f"  Mode thresholds:")
    print(f"    Sunlit     : r_LMO · n̂₂ > 0")
    print(f"    GMO visible: angle(r_LMO, r_GMO) < 35°  (shadow side only)")
    print(f"    Nadir      : otherwise")

    t_h, X_h, u_h, m_h = simulate_mission(6500.0)

    # ── Mode summary ─────────────────────────────────────────
    print(f"\n{'─'*66}")
    print("Mode timeline (first switch detections):")
    prev = -1
    for i,(t_i,m_i) in enumerate(zip(t_h,m_h)):
        if m_i != prev:
            print(f"  t = {t_i:7.1f} s  →  {MODE_NAMES[m_i]}-pointing")
            prev = m_i
        if t_i > 7000: break

    # ── Validation table ──────────────────────────────────────
    check = [300, 2100, 3400, 4400, 5600]
    print(f"\n{'─'*66}")
    print("σ_B/N at required times (short MRP, |σ| ≤ 1):")
    print(f"  {'t(s)':>5}  {'Mode':>6}  {'σ₁':>12}  {'σ₂':>12}  "
          f"{'σ₃':>12}  {'|σ|':>10}")
    print(f"  {'─'*5}  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*10}")
    for tc in check:
        s    = X_h[tc, :3]
        mag  = np.linalg.norm(s)
        mode = m_h[tc]
        flag = '✓' if mag <= 1.0 else '✗'
        print(f"  {tc:5d}  {MODE_NAMES[mode]:>6}  "
              f"{s[0]:12.8f}  {s[1]:12.8f}  {s[2]:12.8f}  "
              f"{mag:10.8f} {flag}")

    # ── Tracking errors at check times ───────────────────────
    print(f"\n{'─'*66}")
    print("Attitude tracking error |σ_B/R|:")
    print(f"  {'t(s)':>5}  {'Mode':>6}  {'|σ_B/R|':>12}")
    print(f"  {'─'*5}  {'─'*6}  {'─'*12}")
    for tc in check:
        s=X_h[tc,:3]; w=X_h[tc,3:]
        mode=m_h[tc]
        RN,oRN = get_reference(tc,mode)
        sBR,_  = attitude_error(s,w,RN,oRN)
        print(f"  {tc:5d}  {MODE_NAMES[mode]:>6}  {np.linalg.norm(sBR):12.6f}")

    # ════════════════════════════════════════════════════════
    #  PLOT
    # ════════════════════════════════════════════════════════
    fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True)
    fig.suptitle(
        "Task 11: Full Mission Scenario  (0 – 6500 s)\n"
        f"K = {K_GAIN:.6f} N·m,   P = {P_GAIN:.6f} N·m·s",
        fontsize=13, fontweight='bold')

    cols  = ['#1f77b4','#ff7f0e','#2ca02c']
    lbl_s = [r'$\sigma_1$',r'$\sigma_2$',r'$\sigma_3$']
    lbl_w = [r'$\omega_1$',r'$\omega_2$',r'$\omega_3$']
    lbl_u = [r'$u_1$',     r'$u_2$',     r'$u_3$']

    # ── Mode background shading helper ───────────────────────
    def shade_modes(ax):
        mc = MODE_COLORS
        i, n = 0, len(t_h)
        while i < n:
            m = m_h[i]; j = i
            while j < n and m_h[j] == m:
                j += 1
            ax.axvspan(t_h[i], t_h[min(j,n-1)],
                       alpha=0.10, color=mc[m], linewidth=0)
            i = j

    # σ_B/N
    ax = axes[0]
    shade_modes(ax)
    for j in range(3):
        ax.plot(t_h, X_h[:,j], color=cols[j], lw=1.5, label=lbl_s[j])
    ax.axhline(0, color='k', lw=0.7, ls='--')
    ax.set_ylabel(r'$\sigma_{B/N}$', fontsize=11)
    ax.legend(ncol=3, fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title(r'MRP Attitude  $\sigma_{B/N}$', fontsize=10)

    # BωB/N
    ax = axes[1]
    shade_modes(ax)
    for j in range(3):
        ax.plot(t_h, np.degrees(X_h[:,j+3]),
                color=cols[j], lw=1.5, label=lbl_w[j])
    ax.axhline(0, color='k', lw=0.7, ls='--')
    ax.set_ylabel(r'$^B\omega_{B/N}$ (deg/s)', fontsize=11)
    ax.legend(ncol=3, fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Angular Velocity', fontsize=10)

    # Control torque
    ax = axes[2]
    shade_modes(ax)
    for j in range(3):
        ax.plot(t_h, u_h[:,j], color=cols[j], lw=1.5, label=lbl_u[j])
    ax.axhline(0, color='k', lw=0.7, ls='--')
    ax.set_ylabel(r'$\mathbf{u}$ (N·m)', fontsize=11)
    ax.legend(ncol=3, fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Control Torque', fontsize=10)

    # Mode indicator
    ax = axes[3]
    ax.plot(t_h, m_h, color='#444', lw=1.2, drawstyle='steps-post')
    ax.set_yticks([0,1,2])
    ax.set_yticklabels(['Sun','GMO','Nadir'], fontsize=10)
    ax.set_ylabel('Mode', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title('Active Control Mode', fontsize=10)
    shade_modes(ax)

    # Mark validation times
    for ax in axes:
        for tc in check:
            ax.axvline(tc, color='gray', lw=0.8, ls=':', alpha=0.7)

    # Legend for mode shading
    patches = [mpatches.Patch(color=MODE_COLORS[m], alpha=0.3,
               label=f'{MODE_NAMES[m]}-pointing')
               for m in [MODE_SUN, MODE_GMO, MODE_NADIR]]
    axes[0].legend(handles=patches + [
        plt.Line2D([0],[0],color=cols[j],lw=1.5,label=lbl_s[j])
        for j in range(3)], ncol=6, fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.show()

    print("="*66)
    return t_h, X_h, u_h, m_h

if __name__ == "__main__":
    validate()
