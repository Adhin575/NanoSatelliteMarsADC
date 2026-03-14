"""
ASEN 5010 - Task 8: Sun-Pointing Attitude Control

PD control law:  Bu = −K σ_B/R − P BωB/R

Gain Design (linearized closed-loop)
-------------------------------------
Linearized EOM per principal axis i:
    Iᵢ σ̈ᵢ + P σ̇ᵢ + K σᵢ = 0

    ωₙᵢ  = √(K/Iᵢ)
    ξᵢ   = P / (2√(K Iᵢ))
    τᵢ   = 1/(ξᵢ ωₙᵢ) = 2Iᵢ/P

Constraints:
  (1) τ_max ≤ 120 s  →  2·I_max/P ≤ 120  →  P ≥ 2·10/120 = 1/6 Nm·s
      Choose P = 1/6 (tightest bound, gives τ₁ = 120 s exactly)

  (2) All axes under-damped or critically damped: ξᵢ ≤ 1 for all i
      ξᵢ is maximised when Iᵢ is minimised (I_min = 5 kg·m²)
      ξ_max = P/(2√(K·I_min)) ≤ 1  →  K ≥ P²/(4·I_min)
      Choose K = P²/(4·I_min) → axis 2 (I=5) critically damped,
                                  axes 1,3 under-damped ✓

Result:
    P = 1/6  ≈ 0.16667  N·m·s
    K = 1/720 ≈ 0.001389  N·m

Axis-by-axis modal response:
    I=10: ωₙ=0.01178 rad/s, ξ=0.7071, τ=120.0 s  (under-damped)
    I= 5: ωₙ=0.01667 rad/s, ξ=1.0000, τ= 60.0 s  (critically damped)
    I=7.5: ωₙ=0.01361 rad/s, ξ=0.8165, τ= 90.0 s  (under-damped)
"""

import numpy as np
import matplotlib.pyplot as plt

# ════════════════════════════════════════════════════════════════════
#  SPACECRAFT PARAMETERS
# ════════════════════════════════════════════════════════════════════
I_sc   = np.diag([10.0, 5.0, 7.5])
I_inv  = np.linalg.inv(I_sc)

SIGMA0 = np.array([ 0.3, -0.4,  0.5])
OMEGA0 = np.radians(np.array([1.00, 1.75, -2.20]))

# ── Gains ─────────────────────────────────────────────────────────
I_max, I_min = 10.0, 5.0
TAU          = 120.0

P_GAIN = 2.0 * I_max / TAU                    # 1/6  N·m·s
K_GAIN = P_GAIN**2 / (4.0 * I_min)            # 1/720 N·m

# ════════════════════════════════════════════════════════════════════
#  UTILITIES
# ════════════════════════════════════════════════════════════════════
def skew(v):
    return np.array([[ 0,   -v[2],  v[1]],
                     [ v[2], 0,    -v[0]],
                     [-v[1], v[0],  0   ]])

def mrp_switch(s):
    return -s/np.dot(s,s) if np.dot(s,s)>1.0 else s.copy()

def mrp_B(s):
    s2=np.dot(s,s)
    return (1-s2)*np.eye(3)+2*skew(s)+2*np.outer(s,s)

def mrp_to_dcm(s):
    s2=np.dot(s,s); S=skew(s)
    return np.eye(3)+(8*S@S-4*(1-s2)*S)/(1+s2)**2

def dcm_to_mrp(C):
    tr=np.trace(C); q0=0.5*np.sqrt(max(0.,1.+tr))
    if q0>1e-10:
        qv=np.array([C[1,2]-C[2,1],C[2,0]-C[0,2],C[0,1]-C[1,0]])/(4*q0)
    else:
        qv=np.array([C[1,2]-C[2,1],C[2,0]-C[0,2],C[0,1]-C[1,0]])
        qv=qv/(np.linalg.norm(qv)+1e-16)*0.9999
    s=qv/(1+q0)
    return -s/np.dot(s,s) if np.dot(s,s)>1 else s

# ════════════════════════════════════════════════════════════════════
#  REFERENCE: SUN-POINTING (Task 3)
# ════════════════════════════════════════════════════════════════════
def dcm_RsN(t=None):
    return np.array([[-1.,0.,0.],[0.,0.,1.],[0.,1.,0.]])
def omega_RsN(t=None):
    return np.zeros(3)

# ════════════════════════════════════════════════════════════════════
#  ATTITUDE ERROR & CONTROL
# ════════════════════════════════════════════════════════════════════
def attitude_error(s_BN, w_BN, RN, w_RN_N):
    BN      = mrp_to_dcm(s_BN)
    s_BR    = dcm_to_mrp(BN @ RN.T)
    w_BR_B  = w_BN - BN @ w_RN_N
    return s_BR, w_BR_B

def pd_control(s_BR, w_BR, K=K_GAIN, P=P_GAIN):
    return -K*s_BR - P*w_BR

# ════════════════════════════════════════════════════════════════════
#  EOM & RK4
# ════════════════════════════════════════════════════════════════════
def eom(X, u):
    s,w   = X[:3], X[3:]
    sdot  = 0.25 * mrp_B(s) @ w
    wdot  = I_inv @ (-skew(w) @ I_sc @ w + u)
    return np.concatenate([sdot,wdot])

def rk4_step(X, dt, u):
    k1=eom(X,        u); k2=eom(X+dt/2*k1, u)
    k3=eom(X+dt/2*k2,u); k4=eom(X+dt*k3,   u)
    return X+(dt/6)*(k1+2*k2+2*k3+k4)

def simulate(t_end, ref_dcm, ref_omega, dt=1.0, K=K_GAIN, P=P_GAIN):
    X   = np.concatenate([SIGMA0, OMEGA0])
    N   = int(round(t_end/dt))
    t_h = np.zeros(N+1); X_h=np.zeros((N+1,6)); u_h=np.zeros((N+1,3))
    t_h[0]=0.; X_h[0]=X
    for i in range(N):
        t_n       = i*dt
        RN        = ref_dcm(t_n)
        oRN       = ref_omega(t_n)
        sBR,oBR   = attitude_error(X[:3],X[3:],RN,oRN)
        u         = pd_control(sBR,oBR,K,P)
        u_h[i]    = u
        X         = rk4_step(X,dt,u)
        X[:3]     = mrp_switch(X[:3])
        t_h[i+1]  = (i+1)*dt
        X_h[i+1]  = X
    RN=ref_dcm(t_h[-1]); oRN=ref_omega(t_h[-1])
    sBR,oBR=attitude_error(X[:3],X[3:],RN,oRN)
    u_h[-1]=pd_control(sBR,oBR,K,P)
    return t_h, X_h, u_h

# ════════════════════════════════════════════════════════════════════
#  VALIDATION
# ════════════════════════════════════════════════════════════════════
def validate():
    print("="*66)
    print("ASEN 5010  —  Task 8: Sun-Pointing PD Control")
    print("="*66)

    # ── Gain analysis table ───────────────────────────────────
    print(f"\nGain Design:")
    print(f"  P = 2·I_max/τ_max = 2·{I_max}/{TAU} = {P_GAIN:.10f} N·m·s")
    print(f"  K = P²/(4·I_min) = {P_GAIN:.8f}²/(4·{I_min}) = {K_GAIN:.10f} N·m")
    print(f"\n  {'Axis':<8} {'Iᵢ':>5}  {'ωₙ (rad/s)':>12}  "
          f"{'ξ':>8}  {'τ (s)':>8}  {'Damping'}")
    print(f"  {'─'*8} {'─'*5}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*14}")
    for Ii, ax in zip([10.,5.,7.5],['I₁=10','I₂=5','I₃=7.5']):
        wn  = np.sqrt(K_GAIN/Ii)
        xi  = P_GAIN/(2*np.sqrt(K_GAIN*Ii))
        tau = 2*Ii/P_GAIN
        damp=('critically damped' if abs(xi-1)<1e-9
              else 'under-damped ✓' if xi<1 else 'over-damped ✗')
        print(f"  {ax:<8} {Ii:>5.1f}  {wn:>12.8f}  "
              f"{xi:>8.6f}  {tau:>8.2f}  {damp}")

    # ── Simulate 400 s ────────────────────────────────────────
    t_h, X_h, u_h = simulate(400., dcm_RsN, omega_RsN)

    # ── Validation table ──────────────────────────────────────
    check = [15, 100, 200, 400]
    print(f"\n{'─'*66}")
    print("σ_B/N at required times (short MRP, |σ|≤1):")
    print(f"  {'t(s)':>5}  {'σ₁':>12}  {'σ₂':>12}  {'σ₃':>12}  {'|σ|':>10}")
    print(f"  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*10}")
    for tc in check:
        s = X_h[tc,:3]
        print(f"  {tc:5d}  {s[0]:12.8f}  {s[1]:12.8f}  {s[2]:12.8f}"
              f"  {np.linalg.norm(s):10.8f}")

    # ── Steady-state error ────────────────────────────────────
    s_f   = X_h[-1,:3]; w_f=X_h[-1,3:]
    sBR_f,_ = attitude_error(s_f,w_f,dcm_RsN(),omega_RsN())
    print(f"\n  σ_B/R(400 s) = [{sBR_f[0]:.6f}, {sBR_f[1]:.6f}, {sBR_f[2]:.6f}]")
    print(f"  |σ_B/R(400 s)| = {np.linalg.norm(sBR_f):.4e}  "
          +("✓ converged" if np.linalg.norm(sBR_f)<0.01 else "still converging"))

    # ── Plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(3,1,figsize=(11,10),sharex=True)
    fig.suptitle(f"Task 8: Sun-Pointing PD Control\n"
                 f"K = {K_GAIN:.6f} N·m,   P = {P_GAIN:.6f} N·m·s\n"
                 r"$\tau_1=120\,$s (under), $\tau_2=60\,$s (crit.), "
                 r"$\tau_3=90\,$s (under)",
                 fontsize=12, fontweight='bold')

    cols=['#1f77b4','#ff7f0e','#2ca02c']
    lbls_s=[r'$\sigma_1$',r'$\sigma_2$',r'$\sigma_3$']
    lbls_w=[r'$\omega_1$',r'$\omega_2$',r'$\omega_3$']
    lbls_u=[r'$u_1$',r'$u_2$',r'$u_3$']

    ax=axes[0]
    for j in range(3):
        ax.plot(t_h, X_h[:,j], color=cols[j], lw=1.8, label=lbls_s[j])
    ax.axhline(0,color='k',lw=0.8,ls='--')
    ax.set_ylabel(r'$\sigma_{B/N}$',fontsize=12)
    ax.legend(ncol=3,fontsize=10,loc='upper right')
    ax.grid(True,alpha=0.35)
    ax.set_title(r'MRP Attitude  $\sigma_{B/N}$',fontsize=11)

    ax=axes[1]
    for j in range(3):
        ax.plot(t_h, np.degrees(X_h[:,j+3]),
                color=cols[j], lw=1.8, label=lbls_w[j])
    ax.axhline(0,color='k',lw=0.8,ls='--')
    ax.set_ylabel(r'$^B\omega_{B/N}$ (deg/s)',fontsize=12)
    ax.legend(ncol=3,fontsize=10,loc='upper right')
    ax.grid(True,alpha=0.35)
    ax.set_title('Angular Velocity',fontsize=11)

    ax=axes[2]
    for j in range(3):
        ax.plot(t_h, u_h[:,j], color=cols[j], lw=1.8, label=lbls_u[j])
    ax.axhline(0,color='k',lw=0.8,ls='--')
    ax.set_ylabel(r'$\mathbf{u}$ (N·m)',fontsize=12)
    ax.set_xlabel('Time (s)',fontsize=12)
    ax.legend(ncol=3,fontsize=10,loc='upper right')
    ax.grid(True,alpha=0.35)
    ax.set_title('PD Control Torque',fontsize=11)

    for ax in axes:
        for tc in check:
            ax.axvline(tc,color='gray',lw=0.9,ls=':',alpha=0.7)
    plt.tight_layout()
    plt.show()
    print("="*66)
    return t_h, X_h, u_h

if __name__ == "__main__":
    validate()
