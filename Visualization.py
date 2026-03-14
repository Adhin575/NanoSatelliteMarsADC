"""
ASEN 5010 - Task 11: Full Mission Scenario Simulation
Complete Python visualization code with:
  - 2D property plots (sigma, omega, control torque, mode, tracking error)
  - 3D inertial-space attitude visualization
  - All plots displayed inline (no saving)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ════════════════════════════════════════════════════════════════════
#  SPACECRAFT & ORBIT PARAMETERS
# ════════════════════════════════════════════════════════════════════
I_sc   = np.diag([10.0, 5.0, 7.5])
I_inv  = np.linalg.inv(I_sc)

SIGMA0 = np.array([ 0.3, -0.4,  0.5])
OMEGA0 = np.radians(np.array([1.00, 1.75, -2.20]))

MU_MARS       = 42828.3;  R_MARS = 3396.19
R_LMO         = R_MARS + 400.0
OMEGA_LMO     = np.radians(20.0);  I_LMO_ang = np.radians(30.0)
THETA0_LMO    = np.radians(60.0)
THETA_DOT_LMO = np.sqrt(MU_MARS / R_LMO**3)

R_GMO         = 20424.2
OMEGA_GMO     = np.radians(0.0);   I_GMO_ang = np.radians(0.0)
THETA0_GMO    = np.radians(250.0)
THETA_DOT_GMO = np.sqrt(MU_MARS / R_GMO**3)

N3            = np.array([0., 0., 1.])
P_GAIN        = 2.0 * 10.0 / 120.0
K_GAIN        = P_GAIN**2 / (4.0 * 5.0)
GMO_VIS_ANGLE = np.radians(35.0)

MODE_SUN   = 0;  MODE_GMO = 1;  MODE_NADIR = 2
MODE_NAMES = {0:'Sun', 1:'GMO', 2:'Nadir'}
MODE_COLORS= {0:'#F4C542', 1:'#4A90D9', 2:'#5BAA6E'}

# ════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════
def skew(v):
    return np.array([[ 0,   -v[2],  v[1]],
                     [ v[2], 0,    -v[0]],
                     [-v[1], v[0],  0   ]])

def mrp_switch(s):
    return -s/np.dot(s,s) if np.dot(s,s) > 1.0 else s.copy()

def mrp_B(s):
    s2 = np.dot(s,s)
    return (1-s2)*np.eye(3) + 2*skew(s) + 2*np.outer(s,s)

def mrp_to_dcm(s):
    s2 = np.dot(s,s); S = skew(s)
    return np.eye(3) + (8*S@S - 4*(1-s2)*S)/(1+s2)**2

def dcm_to_mrp(C):
    tr = np.trace(C); q0 = 0.5*np.sqrt(max(0., 1.+tr))
    if q0 > 1e-10:
        qv = np.array([C[1,2]-C[2,1], C[2,0]-C[0,2], C[0,1]-C[1,0]])/(4*q0)
    else:
        qv = np.array([C[1,2]-C[2,1], C[2,0]-C[0,2], C[0,1]-C[1,0]])
        qv = qv/(np.linalg.norm(qv)+1e-16)*0.9999
    s = qv/(1+q0)
    return -s/np.dot(s,s) if np.dot(s,s)>1 else s

# ════════════════════════════════════════════════════════════════════
#  ORBIT POSITIONS
# ════════════════════════════════════════════════════════════════════
def _R1(a):
    c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,s],[0,-s,c]])
def _R3(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,s,0],[-s,c,0],[0,0,1]])

def pos_LMO(t):
    th = THETA0_LMO + THETA_DOT_LMO*t
    return (_R3(th)@_R1(I_LMO_ang)@_R3(OMEGA_LMO)).T @ np.array([R_LMO,0.,0.])

def pos_GMO(t):
    th = THETA0_GMO + THETA_DOT_GMO*t
    return (_R3(th)@_R1(I_GMO_ang)@_R3(OMEGA_GMO)).T @ np.array([R_GMO,0.,0.])

# ════════════════════════════════════════════════════════════════════
#  REFERENCE FRAMES
# ════════════════════════════════════════════════════════════════════
def dcm_RsN(t=None):
    return np.array([[-1.,0.,0.],[0.,0.,1.],[0.,1.,0.]])
def omega_RsN(t=None):
    return np.zeros(3)

def dcm_RnN(t):
    th = THETA0_LMO + THETA_DOT_LMO*t
    HN = _R3(th)@_R1(I_LMO_ang)@_R3(OMEGA_LMO)
    return np.diag([-1.,1.,-1.]) @ HN
def omega_RnN(t):
    th = THETA0_LMO + THETA_DOT_LMO*t
    HN = _R3(th)@_R1(I_LMO_ang)@_R3(OMEGA_LMO)
    return THETA_DOT_LMO * HN[2,:]

def dcm_RcN(t):
    dr = pos_GMO(t) - pos_LMO(t)
    r1 = -dr/np.linalg.norm(dr)
    c2 = np.cross(dr,N3); r2 = c2/np.linalg.norm(c2)
    return np.array([r1, r2, np.cross(r1,r2)])
def omega_RcN(t, dt=1.0):
    NH_dot = (dcm_RcN(t+dt) - dcm_RcN(t-dt)).T/(2*dt)
    Ot = NH_dot @ dcm_RcN(t)
    return np.array([Ot[2,1],Ot[0,2],Ot[1,0]])

# ════════════════════════════════════════════════════════════════════
#  MODE SELECTION
# ════════════════════════════════════════════════════════════════════
def select_mode(t):
    r_lmo = pos_LMO(t)
    if r_lmo[1] > 0:
        return MODE_SUN
    r_gmo = pos_GMO(t)
    cos_a = np.clip(np.dot(r_lmo,r_gmo)/(np.linalg.norm(r_lmo)*np.linalg.norm(r_gmo)),-1.,1.)
    return MODE_GMO if np.arccos(cos_a) < GMO_VIS_ANGLE else MODE_NADIR

def get_reference(t, mode):
    if mode == MODE_SUN:   return dcm_RsN(t), omega_RsN(t)
    elif mode == MODE_GMO: return dcm_RcN(t), omega_RcN(t)
    else:                  return dcm_RnN(t), omega_RnN(t)

# ════════════════════════════════════════════════════════════════════
#  ATTITUDE ERROR & PD CONTROL
# ════════════════════════════════════════════════════════════════════
def attitude_error(s_BN, w_BN, RN, w_RN_N):
    BN    = mrp_to_dcm(s_BN)
    s_BR  = dcm_to_mrp(BN @ RN.T)
    w_BR  = w_BN - BN @ w_RN_N
    return s_BR, w_BR

def pd_control(s_BR, w_BR):
    return -K_GAIN*s_BR - P_GAIN*w_BR

# ════════════════════════════════════════════════════════════════════
#  EOM & RK4
# ════════════════════════════════════════════════════════════════════
def eom(X, u):
    s,w   = X[:3], X[3:]
    sdot  = 0.25 * mrp_B(s) @ w
    wdot  = I_inv @ (-skew(w) @ I_sc @ w + u)
    return np.concatenate([sdot, wdot])

def rk4_step(X, dt, u):
    k1=eom(X,        u); k2=eom(X+dt/2*k1, u)
    k3=eom(X+dt/2*k2,u); k4=eom(X+dt*k3,   u)
    return X + (dt/6)*(k1+2*k2+2*k3+k4)

# ════════════════════════════════════════════════════════════════════
#  FULL MISSION SIMULATION
# ════════════════════════════════════════════════════════════════════
def simulate_mission(t_end=6500.0, dt=1.0):
    print(f"  Running simulation: 0 → {t_end:.0f} s, dt={dt} s ...")
    X     = np.concatenate([SIGMA0, OMEGA0])
    N     = int(round(t_end/dt))
    t_h   = np.zeros(N+1)
    X_h   = np.zeros((N+1,6))
    u_h   = np.zeros((N+1,3))
    m_h   = np.zeros(N+1, dtype=int)
    sBR_h = np.zeros((N+1,3))
    rL_h  = np.zeros((N+1,3))
    rG_h  = np.zeros((N+1,3))
    t_h[0]=0.; X_h[0]=X
    rL_h[0] = pos_LMO(0.)
    rG_h[0] = pos_GMO(0.)

    for i in range(N):
        t_n      = i*dt
        mode     = select_mode(t_n)
        RN,oRN   = get_reference(t_n, mode)
        sBR,oBR  = attitude_error(X[:3], X[3:], RN, oRN)
        u        = pd_control(sBR, oBR)
        u_h[i]   = u
        m_h[i]   = mode
        sBR_h[i] = sBR
        X        = rk4_step(X, dt, u)
        X[:3]    = mrp_switch(X[:3])
        t_h[i+1]  = (i+1)*dt
        X_h[i+1]  = X
        rL_h[i+1] = pos_LMO((i+1)*dt)
        rG_h[i+1] = pos_GMO((i+1)*dt)

    mode = select_mode(t_h[-1])
    RN,oRN = get_reference(t_h[-1], mode)
    sBR,oBR = attitude_error(X[:3],X[3:],RN,oRN)
    u_h[-1]   = pd_control(sBR,oBR)
    m_h[-1]   = mode
    sBR_h[-1] = sBR
    print("  Simulation complete.")
    return t_h, X_h, u_h, m_h, sBR_h, rL_h, rG_h

# ════════════════════════════════════════════════════════════════════
#  HELPER: shade mode backgrounds on axes
# ════════════════════════════════════════════════════════════════════
def shade_modes(ax, t_h, m_h):
    n = len(t_h)
    i = 0
    while i < n:
        m = m_h[i]; j = i
        while j < n and m_h[j] == m: j += 1
        ax.axvspan(t_h[i], t_h[min(j,n-1)],
                   alpha=0.10, color=MODE_COLORS[m], linewidth=0)
        i = j

CHECK_TIMES = [300, 2100, 3400, 4400, 5600]

# ════════════════════════════════════════════════════════════════════
#  PLOT 1 – MRP attitude states
# ════════════════════════════════════════════════════════════════════
def plot_sigma(t_h, X_h, m_h):
    fig, ax = plt.subplots(figsize=(12,4))
    fig.suptitle('MRP Attitude  σ_B/N  vs Time', fontsize=13, fontweight='bold')
    shade_modes(ax, t_h, m_h)
    cols = ['#1f77b4','#ff7f0e','#2ca02c']
    lbls = [r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$']
    for j in range(3):
        ax.plot(t_h, X_h[:,j], color=cols[j], lw=1.6, label=lbls[j])
    ax.axhline(0, color='k', lw=0.7, ls='--', alpha=0.5)
    for tc in CHECK_TIMES:
        ax.axvline(tc, color='gray', lw=0.8, ls=':', alpha=0.6)
    ax.set_xlabel('Time (s)', fontsize=11); ax.set_ylabel(r'$\sigma_{B/N}$', fontsize=11)
    ax.legend(ncol=3, fontsize=10); ax.grid(True, alpha=0.3)
    _mode_legend(ax)
    plt.tight_layout(); plt.show()

# ════════════════════════════════════════════════════════════════════
#  PLOT 2 – Angular velocity
# ════════════════════════════════════════════════════════════════════
def plot_omega(t_h, X_h, m_h):
    fig, ax = plt.subplots(figsize=(12,4))
    fig.suptitle(r'Angular Velocity  $^B\omega_{B/N}$  vs Time', fontsize=13, fontweight='bold')
    shade_modes(ax, t_h, m_h)
    cols = ['#1f77b4','#ff7f0e','#2ca02c']
    lbls = [r'$\omega_1$', r'$\omega_2$', r'$\omega_3$']
    for j in range(3):
        ax.plot(t_h, np.degrees(X_h[:,j+3]), color=cols[j], lw=1.6, label=lbls[j])
    ax.axhline(0, color='k', lw=0.7, ls='--', alpha=0.5)
    for tc in CHECK_TIMES:
        ax.axvline(tc, color='gray', lw=0.8, ls=':', alpha=0.6)
    ax.set_xlabel('Time (s)', fontsize=11); ax.set_ylabel('deg/s', fontsize=11)
    ax.legend(ncol=3, fontsize=10); ax.grid(True, alpha=0.3)
    _mode_legend(ax)
    plt.tight_layout(); plt.show()

# ════════════════════════════════════════════════════════════════════
#  PLOT 3 – Control torque
# ════════════════════════════════════════════════════════════════════
def plot_control(t_h, u_h, m_h):
    fig, ax = plt.subplots(figsize=(12,4))
    fig.suptitle('PD Control Torque  u  vs Time', fontsize=13, fontweight='bold')
    shade_modes(ax, t_h, m_h)
    cols = ['#1f77b4','#ff7f0e','#2ca02c']
    lbls = [r'$u_1$', r'$u_2$', r'$u_3$']
    for j in range(3):
        ax.plot(t_h, u_h[:,j], color=cols[j], lw=1.6, label=lbls[j])
    ax.axhline(0, color='k', lw=0.7, ls='--', alpha=0.5)
    for tc in CHECK_TIMES:
        ax.axvline(tc, color='gray', lw=0.8, ls=':', alpha=0.6)
    ax.set_xlabel('Time (s)', fontsize=11); ax.set_ylabel('N·m', fontsize=11)
    ax.legend(ncol=3, fontsize=10); ax.grid(True, alpha=0.3)
    _mode_legend(ax)
    plt.tight_layout(); plt.show()

# ════════════════════════════════════════════════════════════════════
#  PLOT 4 – Attitude tracking error |σ_B/R|
# ════════════════════════════════════════════════════════════════════
def plot_tracking_error(t_h, sBR_h, m_h):
    err_mag = np.linalg.norm(sBR_h, axis=1)
    fig, ax = plt.subplots(figsize=(12,4))
    fig.suptitle('Attitude Tracking Error  |σ_B/R|  vs Time', fontsize=13, fontweight='bold')
    shade_modes(ax, t_h, m_h)
    ax.plot(t_h, err_mag, color='#8B3A8B', lw=1.8, label=r'$|\sigma_{B/R}|$')
    ax.axhline(0, color='k', lw=0.7, ls='--', alpha=0.5)
    for tc in CHECK_TIMES:
        ax.axvline(tc, color='gray', lw=0.8, ls=':', alpha=0.6)
        idx = int(round(tc))
        ax.annotate(f'{err_mag[idx]:.3f}', xy=(tc, err_mag[idx]),
                    xytext=(tc+60, err_mag[idx]+0.03),
                    fontsize=8, color='#8B3A8B',
                    arrowprops=dict(arrowstyle='->', color='#8B3A8B', lw=0.8))
    ax.set_xlabel('Time (s)', fontsize=11); ax.set_ylabel(r'$|\sigma_{B/R}|$', fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    _mode_legend(ax)
    plt.tight_layout(); plt.show()

# ════════════════════════════════════════════════════════════════════
#  PLOT 5 – Mode timeline (bar chart style)
# ════════════════════════════════════════════════════════════════════
def plot_mode_timeline(t_h, m_h):
    fig, ax = plt.subplots(figsize=(12,2.5))
    fig.suptitle('Active Control Mode Timeline', fontsize=13, fontweight='bold')
    n = len(t_h)
    i = 0
    while i < n:
        m = m_h[i]; j = i
        while j < n and m_h[j] == m: j += 1
        ax.barh(0, t_h[min(j,n-1)]-t_h[i], left=t_h[i],
                height=0.6, color=MODE_COLORS[m], alpha=0.85,
                label=MODE_NAMES[m] if i==list(m_h).index(m) else "")
        mid = (t_h[i]+t_h[min(j,n-1)])/2
        ax.text(mid, 0, MODE_NAMES[m], ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
        i = j
    for tc in CHECK_TIMES:
        ax.axvline(tc, color='gray', lw=1.2, ls=':', alpha=0.8)
        ax.text(tc, 0.38, f't={tc}', ha='center', fontsize=7.5, color='gray')
    ax.set_xlim(0, t_h[-1]); ax.set_yticks([])
    ax.set_xlabel('Time (s)', fontsize=11); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout(); plt.show()

# ════════════════════════════════════════════════════════════════════
#  PLOT 6 – Combined summary dashboard (4 subplots)
# ════════════════════════════════════════════════════════════════════
def plot_dashboard(t_h, X_h, u_h, m_h, sBR_h):
    fig = plt.figure(figsize=(14,10))
    fig.suptitle('Mission Scenario Dashboard  —  ASEN 5010 Task 11\n'
                 f'K={K_GAIN:.6f} N·m   P={P_GAIN:.6f} N·m·s   Duration=6500 s',
                 fontsize=12, fontweight='bold')
    gs = gridspec.GridSpec(4, 1, hspace=0.45)
    cols = ['#1f77b4','#ff7f0e','#2ca02c']

    titles  = [r'MRP Attitude $\sigma_{B/N}$',
               r'Angular Velocity $^B\omega_{B/N}$ (deg/s)',
               r'Control Torque $\mathbf{u}$ (N·m)',
               r'Tracking Error $|\sigma_{B/R}|$']
    ylabels = [r'$\sigma$', 'deg/s', 'N·m', r'$|\sigma_{B/R}|$']

    for idx, (title, ylabel) in enumerate(zip(titles, ylabels)):
        ax = fig.add_subplot(gs[idx])
        shade_modes(ax, t_h, m_h)
        if idx == 0:
            for j in range(3):
                ax.plot(t_h, X_h[:,j], color=cols[j], lw=1.4,
                        label=[r'$\sigma_1$',r'$\sigma_2$',r'$\sigma_3$'][j])
            ax.legend(ncol=3, fontsize=8, loc='upper right')
        elif idx == 1:
            for j in range(3):
                ax.plot(t_h, np.degrees(X_h[:,j+3]), color=cols[j], lw=1.4,
                        label=[r'$\omega_1$',r'$\omega_2$',r'$\omega_3$'][j])
            ax.legend(ncol=3, fontsize=8, loc='upper right')
        elif idx == 2:
            for j in range(3):
                ax.plot(t_h, u_h[:,j], color=cols[j], lw=1.4,
                        label=[r'$u_1$',r'$u_2$',r'$u_3$'][j])
            ax.legend(ncol=3, fontsize=8, loc='upper right')
        else:
            err_mag = np.linalg.norm(sBR_h, axis=1)
            ax.plot(t_h, err_mag, color='#8B3A8B', lw=1.6)
            for tc in CHECK_TIMES:
                idx2 = int(round(tc))
                ax.plot(tc, err_mag[idx2], 'o', ms=5, color='#8B3A8B')

        ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.4)
        for tc in CHECK_TIMES:
            ax.axvline(tc, color='gray', lw=0.7, ls=':', alpha=0.55)
        ax.set_ylabel(ylabel, fontsize=9); ax.grid(True, alpha=0.25)
        ax.set_title(title, fontsize=9, loc='left', pad=3)
        if idx < 3: ax.set_xticklabels([])
        else: ax.set_xlabel('Time (s)', fontsize=10)

    _mode_legend(fig.axes[0])
    plt.show()

# ════════════════════════════════════════════════════════════════════
#  PLOT 7 – Kinetic energy & angular momentum magnitude
# ════════════════════════════════════════════════════════════════════
def plot_energy_momentum(t_h, X_h, m_h, u_h):
    T_arr = np.array([0.5*(X_h[i,3:] @ I_sc @ X_h[i,3:]) for i in range(len(t_h))])
    H_arr = np.array([np.linalg.norm(I_sc @ X_h[i,3:]) for i in range(len(t_h))])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle('Rotational Kinetic Energy & Angular Momentum Magnitude', fontsize=12, fontweight='bold')

    shade_modes(ax1, t_h, m_h); shade_modes(ax2, t_h, m_h)
    ax1.plot(t_h, T_arr, color='#C0392B', lw=1.8, label='T (J)')
    ax1.set_ylabel('Kinetic Energy (J)', fontsize=10)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    ax1.set_title('Rotational KE  T = ½ωᵀ[I]ω', fontsize=9, loc='left')

    ax2.plot(t_h, H_arr, color='#2980B9', lw=1.8, label='|H| (kg·m²/s)')
    ax2.set_ylabel('|H| (kg·m²/s)', fontsize=10)
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.set_title('Angular Momentum Magnitude', fontsize=9, loc='left')

    for ax in [ax1, ax2]:
        for tc in CHECK_TIMES:
            ax.axvline(tc, color='gray', lw=0.8, ls=':', alpha=0.6)
    _mode_legend(ax1)
    plt.tight_layout(); plt.show()

# ════════════════════════════════════════════════════════════════════
#  PLOT 8 – MRP magnitude |σ_B/N| with shadow switch highlights
# ════════════════════════════════════════════════════════════════════
def plot_mrp_magnitude(t_h, X_h, m_h):
    mag = np.linalg.norm(X_h[:,:3], axis=1)
    shadow_mask = mag >= 0.98

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle('MRP Magnitude  |σ_B/N|  — Shadow Switching Indicator', fontsize=12, fontweight='bold')
    shade_modes(ax, t_h, m_h)
    ax.plot(t_h, mag, color='#34495E', lw=1.6, label=r'$|\sigma_{B/N}|$')
    ax.axhline(1.0, color='#E74C3C', lw=1.2, ls='--', alpha=0.8, label='Switch boundary (|σ|=1)')
    ax.fill_between(t_h, mag, 1.0, where=shadow_mask, alpha=0.25, color='#E74C3C', label='Near/at switch')
    ax.set_xlabel('Time (s)', fontsize=11); ax.set_ylabel(r'$|\sigma_{B/N}|$', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    for tc in CHECK_TIMES:
        ax.axvline(tc, color='gray', lw=0.8, ls=':', alpha=0.6)
    _mode_legend(ax)
    plt.tight_layout(); plt.show()

# ════════════════════════════════════════════════════════════════════
#  3D PLOT 1 – LMO Orbit + GMO orbit in inertial space
# ════════════════════════════════════════════════════════════════════
def plot_3d_orbits(t_h, rL_h, rG_h, m_h):
    fig = plt.figure(figsize=(10, 9))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_title('3D Inertial Orbit Trajectories  (LMO & GMO)', fontsize=12, fontweight='bold', pad=15)

    # Mars sphere
    u_, v_ = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    rs = R_MARS/6000
    ax.plot_surface(rs*np.cos(u_)*np.sin(v_), rs*np.sin(u_)*np.sin(v_),
                    rs*np.cos(v_), color='#C0392B', alpha=0.55, linewidth=0, zorder=0)

    # LMO path coloured by mode
    for m_id, mc in MODE_COLORS.items():
        mask = m_h == m_id
        segs = []
        for i in range(len(t_h)-1):
            if mask[i]:
                segs.append([rL_h[i]/6000, rL_h[i+1]/6000])
        if segs:
            lc = Line3DCollection(segs, colors=mc, linewidths=1.8, alpha=0.85)
            ax.add_collection3d(lc)

    # GMO path (full, equatorial)
    t_gmo = np.linspace(0, 2*np.pi/THETA_DOT_GMO, 400)
    gmo_pts = np.array([pos_GMO(t_i)/6000 for t_i in t_gmo])
    ax.plot(gmo_pts[:,0], gmo_pts[:,1], gmo_pts[:,2],
            color='#5B8FCC', lw=1.2, ls='--', alpha=0.6, label='GMO orbit')

    # Start/end markers
    ax.scatter(*rL_h[0]/6000, s=60, c='lime', zorder=10, label='LMO start')
    ax.scatter(*rL_h[-1]/6000, s=60, c='yellow', zorder=10, label='LMO end')
    ax.scatter(*rG_h[0]/6000, s=50, c='#5B8FCC', marker='^', zorder=10, label='GMO pos')

    # Sun direction arrow
    ax.quiver(0, 0, 0, 0, 1.2, 0, color='#F4C542', arrow_length_ratio=0.1,
              linewidth=2, alpha=0.7, label='Sun (n̂₂)')

    # Inertial axes
    for v, lbl, c in [([1,0,0],'n̂₁','#aaa'), ([0,1,0],'n̂₂','#aaa'), ([0,0,1],'n̂₃','#aaa')]:
        ax.quiver(0,0,0,*[x*1.1 for x in v], color=c, arrow_length_ratio=0.08,
                  linewidth=0.8, alpha=0.4)
        ax.text(*[x*1.15 for x in v], lbl, fontsize=9, color='#888')

    # Mode legend patches
    handles = [mpatches.Patch(color=MODE_COLORS[m], label=f'{MODE_NAMES[m]}-pointing') for m in MODE_COLORS]
    ax.legend(handles=handles + [
        Line2D([0],[0], color='lime', marker='o', ms=7, ls='', label='LMO start'),
        Line2D([0],[0], color='yellow', marker='o', ms=7, ls='', label='LMO end'),
        Line2D([0],[0], color='#5B8FCC', ls='--', lw=1.5, label='GMO orbit'),
        Line2D([0],[0], color='#F4C542', lw=2, label='Sun dir'),
    ], fontsize=8, loc='upper left', ncol=2)

    ax.set_xlabel('n̂₁'); ax.set_ylabel('n̂₂'); ax.set_zlabel('n̂₃')
    lim = 4.5; ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
    ax.set_box_aspect([1,1,1])
    plt.tight_layout(); plt.show()

# ════════════════════════════════════════════════════════════════════
#  3D PLOT 2 – Body frame axes in inertial space (snapshot at each check time)
# ════════════════════════════════════════════════════════════════════
def plot_3d_body_frames(t_h, X_h, rL_h, m_h):
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle('Spacecraft Body Frame Orientation at Validation Times', fontsize=12, fontweight='bold')

    check_indices = [int(round(tc)) for tc in CHECK_TIMES]
    ax_len = 0.4

    for plot_idx, (tc, ci) in enumerate(zip(CHECK_TIMES, check_indices)):
        ax = fig.add_subplot(2, 3, plot_idx+1, projection='3d')
        s  = X_h[ci, :3]
        BN = mrp_to_dcm(s)
        NH = BN.T   # columns = body axes in N-frame
        r  = rL_h[ci] / 6000
        mode = m_h[ci]

        # Mars
        u_, v_ = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        rs = R_MARS/6000 * 0.6
        ax.plot_surface(rs*np.cos(u_)*np.sin(v_), rs*np.sin(u_)*np.sin(v_),
                        rs*np.cos(v_), color='#C0392B', alpha=0.4, linewidth=0)

        # LMO orbit arc (short section)
        t_arc = np.linspace(max(0,tc-400), tc, 80)
        arc   = np.array([pos_LMO(ti)/6000 for ti in t_arc])
        ax.plot(arc[:,0], arc[:,1], arc[:,2], color='#aaa', lw=1, ls='--', alpha=0.5)

        # Body frame arrows
        ax_colors = ['#E24B4A','#1D9E75','#378ADD']
        ax_labels  = ['b̂₁','b̂₂','b̂₃']
        for j in range(3):
            d = NH[:,j]
            ax.quiver(*r, *d*ax_len, color=ax_colors[j], arrow_length_ratio=0.15,
                      linewidth=2.5)
            ax.text(*(r+d*ax_len*1.25), ax_labels[j], fontsize=8, color=ax_colors[j], fontweight='bold')

        # SC position dot
        ax.scatter(*r, s=60, c='white', zorder=10, edgecolors='k', linewidths=0.8)

        # Pointing target arrow
        if mode == MODE_SUN:
            ax.quiver(*r, 0, 0.5, 0, color='#F4C542', arrow_length_ratio=0.12,
                      linewidth=1.5, alpha=0.7)
            ax.text(r[0], r[1]+0.55, r[2], 'Sun', fontsize=7, color='#F4C542')
        elif mode == MODE_NADIR:
            nadir = -r/np.linalg.norm(r)*ax_len
            ax.quiver(*r, *nadir, color='#5BAA6E', arrow_length_ratio=0.15, linewidth=1.5, alpha=0.7)

        ax.set_title(f't={tc} s  [{MODE_NAMES[mode]}]',
                     fontsize=9, color=MODE_COLORS[mode], fontweight='bold')
        lim = 1.3
        ax.set_xlim(r[0]-lim, r[0]+lim)
        ax.set_ylim(r[1]-lim, r[1]+lim)
        ax.set_zlim(r[2]-lim, r[2]+lim)
        ax.set_xlabel('n̂₁', fontsize=7); ax.set_ylabel('n̂₂', fontsize=7)
        ax.set_zlabel('n̂₃', fontsize=7)
        ax.tick_params(labelsize=6)

    # Legend
    leg_ax = fig.add_subplot(2, 3, 6)
    leg_ax.axis('off')
    entries = [
        mpatches.Patch(color='#E24B4A', label='b̂₁ — sensor/antenna axis'),
        mpatches.Patch(color='#1D9E75', label='b̂₂ — intermediate axis'),
        mpatches.Patch(color='#378ADD', label='b̂₃ — solar panel axis'),
        mpatches.Patch(color=MODE_COLORS[0], label='Sun-pointing mode'),
        mpatches.Patch(color=MODE_COLORS[1], label='GMO-pointing mode'),
        mpatches.Patch(color=MODE_COLORS[2], label='Nadir-pointing mode'),
    ]
    leg_ax.legend(handles=entries, fontsize=9, loc='center', frameon=True)
    leg_ax.set_title('Legend', fontsize=10, fontweight='bold')
    plt.tight_layout(); plt.show()

# ════════════════════════════════════════════════════════════════════
#  3D PLOT 3 – Body-axis trajectory in inertial space (animated path)
# ════════════════════════════════════════════════════════════════════
def plot_3d_baxis_trajectory(t_h, X_h, rL_h, m_h):
    """Plot where each body axis unit vector traces in inertial space."""
    fig, axes3d = plt.subplots(1, 3, figsize=(15, 5),
                                subplot_kw={'projection':'3d'})
    fig.suptitle('Body Axis Unit Vector Trajectories in Inertial Space', fontsize=12, fontweight='bold')

    step = 5
    t_s  = t_h[::step]; m_s = m_h[::step]
    X_s  = X_h[::step]

    ax_colors = ['#E24B4A','#1D9E75','#378ADD']
    ax_labels  = ['b̂₁ (sensor)', 'b̂₂', 'b̂₃ (solar panels)']

    for j, (ax3d, color, lbl) in enumerate(zip(axes3d, ax_colors, ax_labels)):
        # Compute b-axis direction in N-frame at each step
        dirs = np.array([mrp_to_dcm(X_s[i,:3]).T[:,j] for i in range(len(t_s))])

        # Plot coloured by mode
        for m_id, mc in MODE_COLORS.items():
            mask = m_s == m_id
            idx_list = np.where(mask)[0]
            if len(idx_list) == 0: continue
            # group into continuous segments
            segs = []
            seg = [idx_list[0]]
            for k in range(1, len(idx_list)):
                if idx_list[k] == idx_list[k-1]+1:
                    seg.append(idx_list[k])
                else:
                    segs.append(seg); seg=[idx_list[k]]
            segs.append(seg)
            for sg in segs:
                d = dirs[sg]
                ax3d.plot(d[:,0], d[:,1], d[:,2], color=mc, lw=1.5, alpha=0.75)

        # Unit sphere wireframe
        u_,v_ = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        ax3d.plot_wireframe(np.cos(u_)*np.sin(v_), np.sin(u_)*np.sin(v_),
                            np.cos(v_), color='#ccc', lw=0.3, alpha=0.2)

        # Start marker
        ax3d.scatter(*dirs[0], s=60, c='lime', zorder=10)
        ax3d.scatter(*dirs[-1], s=60, c='red', zorder=10)

        ax3d.set_title(lbl, fontsize=10, color=color, fontweight='bold')
        ax3d.set_xlabel('n̂₁',fontsize=8); ax3d.set_ylabel('n̂₂',fontsize=8)
        ax3d.set_zlabel('n̂₃',fontsize=8)
        ax3d.set_xlim(-1.1,1.1); ax3d.set_ylim(-1.1,1.1); ax3d.set_zlim(-1.1,1.1)
        ax3d.tick_params(labelsize=7)

    # Mode legend
    handles = [mpatches.Patch(color=MODE_COLORS[m], label=f'{MODE_NAMES[m]}-pointing')
               for m in MODE_COLORS]
    handles += [Line2D([0],[0],marker='o',ms=7,color='lime',ls='',label='Start'),
                Line2D([0],[0],marker='o',ms=7,color='red', ls='',label='End')]
    fig.legend(handles=handles, fontsize=8, loc='lower center', ncol=5, bbox_to_anchor=(0.5,-0.02))
    plt.tight_layout(); plt.show()

# ════════════════════════════════════════════════════════════════════
#  HELPER: mode colour legend
# ════════════════════════════════════════════════════════════════════
def _mode_legend(ax):
    patches = [mpatches.Patch(color=MODE_COLORS[m], alpha=0.6,
               label=f'{MODE_NAMES[m]}-pointing') for m in MODE_COLORS]
    ax.legend(handles=patches + ax.get_legend_handles_labels()[0],
              fontsize=8, ncol=6, loc='upper right')

# ════════════════════════════════════════════════════════════════════
#  VALIDATION TABLE
# ════════════════════════════════════════════════════════════════════
def print_validation(t_h, X_h, m_h, sBR_h):
    print("\n" + "="*68)
    print("ASEN 5010  —  Task 11: Mission Scenario Validation")
    print("="*68)
    print(f"  K = {K_GAIN:.8f} N·m   (= 1/720)")
    print(f"  P = {P_GAIN:.8f} N·m·s (= 1/6)")

    print(f"\n  {'t(s)':>5}  {'Mode':>6}  {'σ₁':>12}  {'σ₂':>12}  {'σ₃':>12}  {'|σ|':>10}  {'|σ_B/R|':>10}")
    print(f"  {'─'*5}  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}")
    for tc in CHECK_TIMES:
        s   = X_h[tc,:3]; mag=np.linalg.norm(s)
        br  = np.linalg.norm(sBR_h[tc])
        mn  = MODE_NAMES[m_h[tc]]
        print(f"  {tc:5d}  {mn:>6}  {s[0]:12.8f}  {s[1]:12.8f}  {s[2]:12.8f}"
              f"  {mag:10.8f}  {br:10.6f}")
    print("="*68)

# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("ASEN 5010 — Task 11 Full Mission Scenario Visualization")
    print("="*56)

    # ── Run simulation ──────────────────────────────────────
    t_h, X_h, u_h, m_h, sBR_h, rL_h, rG_h = simulate_mission(6500.0, dt=1.0)

    # ── Print validation table ──────────────────────────────
    print_validation(t_h, X_h, m_h, sBR_h)

    print("\nDisplaying plots (close each window to advance to the next)...\n")

    # ── 2D Property Plots ───────────────────────────────────
    print("  [1/8] MRP attitude states σ_B/N ...")
    plot_sigma(t_h, X_h, m_h)

    print("  [2/8] Angular velocity ω_B/N ...")
    plot_omega(t_h, X_h, m_h)

    print("  [3/8] Control torque u ...")
    plot_control(t_h, u_h, m_h)

    print("  [4/8] Attitude tracking error |σ_B/R| ...")
    plot_tracking_error(t_h, sBR_h, m_h)

    print("  [5/8] Mode timeline ...")
    plot_mode_timeline(t_h, m_h)

    print("  [6/8] Full dashboard (all states) ...")
    plot_dashboard(t_h, X_h, u_h, m_h, sBR_h)

    print("  [7/8] Kinetic energy & angular momentum ...")
    plot_energy_momentum(t_h, X_h, m_h, u_h)

    print("  [8/8] MRP magnitude & shadow-switching ...")
    plot_mrp_magnitude(t_h, X_h, m_h)

    # ── 3D Plots ─────────────────────────────────────────────
    print("\n  [3D-1] LMO & GMO orbit trajectories in inertial space ...")
    plot_3d_orbits(t_h, rL_h, rG_h, m_h)

    print("  [3D-2] Body-frame snapshots at validation times ...")
    plot_3d_body_frames(t_h, X_h, rL_h, m_h)

    print("  [3D-3] Body-axis unit vector trajectories on unit sphere ...")
    plot_3d_baxis_trajectory(t_h, X_h, rL_h, m_h)

    print("\nAll plots displayed. Done.")
