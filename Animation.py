"""
ASEN 5010 - Task 11: 3D Attitude Animation
Animates the spacecraft body frame rotating in inertial space
along the LMO orbit, with mode-switching visualization.

Requirements: pip install matplotlib numpy
Run: python task11_animation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

# ════════════════════════════════════════════════════════════════════
#  PARAMETERS
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
OMEGA_GMO_    = np.radians(0.0);   I_GMO_ang = np.radians(0.0)
THETA0_GMO    = np.radians(250.0)
THETA_DOT_GMO = np.sqrt(MU_MARS / R_GMO**3)
N3 = np.array([0.,0.,1.])

P_GAIN = 2.0*10.0/120.0
K_GAIN = P_GAIN**2/(4.0*5.0)
GMO_VIS_ANGLE = np.radians(35.0)

MODE_SUN=0; MODE_GMO=1; MODE_NADIR=2
MODE_NAMES  = {0:'Sun-pointing', 1:'GMO-pointing', 2:'Nadir-pointing'}
MODE_COLORS = {0:'#F4C542',      1:'#4A90D9',      2:'#5BAA6E'}

# ════════════════════════════════════════════════════════════════════
#  MATH UTILITIES
# ════════════════════════════════════════════════════════════════════
def skew(v):
    return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])

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
#  ORBIT & REFERENCE FRAMES
# ════════════════════════════════════════════════════════════════════
def R1(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,s],[0,-s,c]])
def R3(a): c,s=np.cos(a),np.sin(a); return np.array([[c,s,0],[-s,c,0],[0,0,1]])

def pos_LMO(t):
    th=THETA0_LMO+THETA_DOT_LMO*t
    return (R3(th)@R1(I_LMO_ang)@R3(OMEGA_LMO)).T@np.array([R_LMO,0.,0.])

def pos_GMO(t):
    th=THETA0_GMO+THETA_DOT_GMO*t
    return (R3(th)@R1(I_GMO_ang)@R3(OMEGA_GMO_)).T@np.array([R_GMO,0.,0.])

def dcm_RsN(t=None): return np.array([[-1.,0.,0.],[0.,0.,1.],[0.,1.,0.]])
def omega_RsN(t=None): return np.zeros(3)

def dcm_RnN(t):
    th=THETA0_LMO+THETA_DOT_LMO*t
    return np.diag([-1.,1.,-1.])@(R3(th)@R1(I_LMO_ang)@R3(OMEGA_LMO))
def omega_RnN(t):
    th=THETA0_LMO+THETA_DOT_LMO*t
    return THETA_DOT_LMO*(R3(th)@R1(I_LMO_ang)@R3(OMEGA_LMO))[2,:]

def dcm_RcN(t):
    dr=pos_GMO(t)-pos_LMO(t)
    r1=-dr/np.linalg.norm(dr)
    c2=np.cross(dr,N3); r2=c2/np.linalg.norm(c2)
    return np.array([r1,r2,np.cross(r1,r2)])
def omega_RcN(t,dt=1.0):
    NH_dot=(dcm_RcN(t+dt)-dcm_RcN(t-dt)).T/(2*dt)
    Ot=NH_dot@dcm_RcN(t)
    return np.array([Ot[2,1],Ot[0,2],Ot[1,0]])

def select_mode(t):
    r=pos_LMO(t)
    if r[1]>0: return MODE_SUN
    rg=pos_GMO(t)
    ca=np.clip(np.dot(r,rg)/(np.linalg.norm(r)*np.linalg.norm(rg)),-1.,1.)
    return MODE_GMO if np.arccos(ca)<GMO_VIS_ANGLE else MODE_NADIR

def get_reference(t,mode):
    if mode==MODE_SUN:   return dcm_RsN(t),omega_RsN(t)
    elif mode==MODE_GMO: return dcm_RcN(t),omega_RcN(t)
    else:                return dcm_RnN(t),omega_RnN(t)

# ════════════════════════════════════════════════════════════════════
#  SIMULATION
# ════════════════════════════════════════════════════════════════════
def eom(X,u):
    s,w=X[:3],X[3:]
    return np.concatenate([0.25*mrp_B(s)@w, I_inv@(-skew(w)@I_sc@w+u)])

def rk4(X,dt,u):
    k1=eom(X,u);k2=eom(X+dt/2*k1,u);k3=eom(X+dt/2*k2,u);k4=eom(X+dt*k3,u)
    return X+(dt/6)*(k1+2*k2+2*k3+k4)

def simulate(t_end=6500.0, dt=1.0):
    print("Running simulation...", end=' ', flush=True)
    X=np.concatenate([SIGMA0,OMEGA0])
    N=int(round(t_end/dt))
    t_h=np.zeros(N+1); X_h=np.zeros((N+1,6))
    u_h=np.zeros((N+1,3)); m_h=np.zeros(N+1,dtype=int)
    sBR_h=np.zeros((N+1,3))
    rL_h=np.zeros((N+1,3)); rG_h=np.zeros((N+1,3))
    t_h[0]=0.; X_h[0]=X
    rL_h[0]=pos_LMO(0.); rG_h[0]=pos_GMO(0.)
    for i in range(N):
        tn=i*dt; mode=select_mode(tn)
        RN,oRN=get_reference(tn,mode)
        BN=mrp_to_dcm(X[:3])
        sBR=dcm_to_mrp(BN@RN.T); oBR=X[3:]-BN@oRN
        u=-K_GAIN*sBR-P_GAIN*oBR
        u_h[i]=u; m_h[i]=mode; sBR_h[i]=sBR
        X=rk4(X,dt,u); X[:3]=mrp_switch(X[:3])
        t_h[i+1]=(i+1)*dt; X_h[i+1]=X
        rL_h[i+1]=pos_LMO((i+1)*dt); rG_h[i+1]=pos_GMO((i+1)*dt)
    mode=select_mode(t_h[-1]); RN,oRN=get_reference(t_h[-1],mode)
    BN=mrp_to_dcm(X[:3]); sBR=dcm_to_mrp(BN@RN.T)
    u_h[-1]=-K_GAIN*sBR-P_GAIN*(X[3:]-BN@oRN)
    m_h[-1]=mode; sBR_h[-1]=sBR
    print("done.")
    return t_h,X_h,u_h,m_h,sBR_h,rL_h,rG_h

# ════════════════════════════════════════════════════════════════════
#  BUILD FULL ORBIT PATH (for static background)
# ════════════════════════════════════════════════════════════════════
def make_lmo_orbit_ring():
    T_orb = 2*np.pi/THETA_DOT_LMO
    ts = np.linspace(0, T_orb, 300)
    pts = np.array([pos_LMO(t)/6000 for t in ts])
    return pts

def make_gmo_orbit_ring():
    T_orb = 2*np.pi/THETA_DOT_GMO
    ts = np.linspace(0, T_orb, 300)
    pts = np.array([pos_GMO(t)/6000 for t in ts])
    return pts

def make_mars_sphere():
    u_,v_=np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    rs=R_MARS/6000
    return rs*np.cos(u_)*np.sin(v_), rs*np.sin(u_)*np.sin(v_), rs*np.cos(v_)

# ════════════════════════════════════════════════════════════════════
#  ANIMATION
# ════════════════════════════════════════════════════════════════════
def run_animation(t_h, X_h, u_h, m_h, sBR_h, rL_h, rG_h, step=25):
    """
    step : time-steps per animation frame (step=25 → every 25 s of sim)
    """
    SCALE  = 6000.0   # km → display units
    AX_LEN = 0.28     # body frame arrow length

    # Down-sample to animation frames
    idx    = np.arange(0, len(t_h), step)
    t_anim = t_h[idx];   X_anim = X_h[idx]
    m_anim = m_h[idx];   rL_anim= rL_h[idx]/SCALE
    rG_anim= rG_h[idx]/SCALE
    sBR_anim=sBR_h[idx]
    N_frames = len(idx)

    lmo_ring = make_lmo_orbit_ring()
    gmo_ring = make_gmo_orbit_ring()
    mx,my,mz = make_mars_sphere()

    # ── Figure layout ────────────────────────────────────────
    fig = plt.figure(figsize=(15, 8), facecolor='#0d0d1a')
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            left=0.04, right=0.98, top=0.93, bottom=0.07,
                            hspace=0.55, wspace=0.35)

    ax3d  = fig.add_subplot(gs[:, :2], projection='3d')  # large 3D view
    ax_s  = fig.add_subplot(gs[0, 2])   # sigma
    ax_w  = fig.add_subplot(gs[1, 2])   # omega
    ax_e  = fig.add_subplot(gs[2, 2])   # tracking error

    for ax in [ax_s, ax_w, ax_e]:
        ax.set_facecolor('#0d0d1a')
        ax.tick_params(colors='#aaa', labelsize=7)
        for sp in ax.spines.values(): sp.set_color('#333')
        ax.xaxis.label.set_color('#aaa'); ax.yaxis.label.set_color('#aaa')
        ax.grid(True, alpha=0.2, color='#333')

    ax3d.set_facecolor('#0d0d1a')
    fig.patch.set_facecolor('#0d0d1a')

    # ── Static 3D background ─────────────────────────────────
    ax3d.plot_surface(mx,my,mz, color='#8B2500', alpha=0.65, linewidth=0, zorder=1)

    # Mars atmosphere glow (overlaid semi-transparent sphere)
    mx2,my2,mz2 = mx*1.06,my*1.06,mz*1.06
    ax3d.plot_surface(mx2,my2,mz2, color='#CC4411', alpha=0.10, linewidth=0, zorder=0)

    ax3d.plot(lmo_ring[:,0],lmo_ring[:,1],lmo_ring[:,2],
              color='#556688', lw=1.0, ls='--', alpha=0.45, zorder=2)
    ax3d.plot(gmo_ring[:,0],gmo_ring[:,1],gmo_ring[:,2],
              color='#4A90D9', lw=1.0, ls='--', alpha=0.35, zorder=2)

    # Inertial frame axes
    ax_scale = 1.4
    for v,lbl,c in [([1,0,0],'n̂₁','#5566aa'),
                     ([0,1,0],'n̂₂','#5566aa'),
                     ([0,0,1],'n̂₃','#5566aa')]:
        ax3d.quiver(0,0,0,*[x*ax_scale for x in v],
                    color=c, arrow_length_ratio=0.07, lw=0.8, alpha=0.35)
        ax3d.text(*[x*(ax_scale+0.1) for x in v], lbl,
                  fontsize=8, color='#6677bb', alpha=0.6)

    # Sun direction
    ax3d.quiver(0,0,0, 0,1.5,0, color='#FFD700', arrow_length_ratio=0.07,
                lw=1.8, alpha=0.6, zorder=3)
    ax3d.text(0, 1.65, 0, '☀  n̂₂', fontsize=9, color='#FFD700', alpha=0.8)

    lim = 1.35
    ax3d.set_xlim(-lim,lim); ax3d.set_ylim(-lim,lim); ax3d.set_zlim(-lim,lim)
    ax3d.set_xlabel('n̂₁', color='#556688', fontsize=9)
    ax3d.set_ylabel('n̂₂', color='#556688', fontsize=9)
    ax3d.set_zlabel('n̂₃', color='#556688', fontsize=9)
    ax3d.tick_params(colors='#445566', labelsize=6)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor('#222233')
    ax3d.set_box_aspect([1,1,1])

    # ── Time-series subplot setup ─────────────────────────────
    cols2d = ['#E05555','#55AA77','#4488CC']

    for ax,title,yl in [(ax_s, r'$\sigma_{B/N}$', r'$\sigma$'),
                         (ax_w, r'$\omega_{B/N}$ (deg/s)', 'deg/s'),
                         (ax_e, r'$|\sigma_{B/R}|$ (tracking error)', '')]:
        ax.set_title(title, color='#ccddee', fontsize=8, pad=3)
        ax.set_ylabel(yl, color='#aaa', fontsize=7)
        ax.set_xlim(0, t_h[-1])
        ax.axhline(0, color='#333', lw=0.6)

    ax_e.set_xlabel('Time (s)', color='#aaa', fontsize=7)

    # Pre-plot full background traces (faint)
    for j in range(3):
        ax_s.plot(t_h, X_h[:,j],   color=cols2d[j], lw=0.7, alpha=0.18)
        ax_w.plot(t_h, np.degrees(X_h[:,j+3]), color=cols2d[j], lw=0.7, alpha=0.18)
    ax_e.plot(t_h, np.linalg.norm(sBR_h,axis=1), color='#AA66CC', lw=0.7, alpha=0.18)

    ax_s.set_ylim(np.min(X_h[:,:3])-0.05, np.max(X_h[:,:3])+0.05)
    ax_w.set_ylim(np.degrees(np.min(X_h[:,3:]))-0.5, np.degrees(np.max(X_h[:,3:]))+0.5)
    ax_e.set_ylim(-0.02, np.max(np.linalg.norm(sBR_h,axis=1))+0.05)

    # ── Mutable 3D objects ───────────────────────────────────
    # Spacecraft position dot
    sc_dot,  = ax3d.plot([],[],[], 'o', ms=7,  color='white',  zorder=10)
    gmo_dot, = ax3d.plot([],[],[], '^', ms=5,  color='#4A90D9',zorder=10)

    # Orbit trail (last N_trail frames)
    N_trail = 30
    trail_line, = ax3d.plot([],[],[], '-', lw=1.8, alpha=0.7, color=MODE_COLORS[0], zorder=5)

    # GMO line-of-sight
    gmo_los,  = ax3d.plot([],[],[], '-', lw=0.8, color='#4A90D9', alpha=0.35, zorder=4)
    nadir_los,= ax3d.plot([],[],[], '-', lw=0.8, color='#5BAA6E', alpha=0.35, zorder=4)
    sun_los,  = ax3d.plot([],[],[], '-', lw=0.8, color='#FFD700', alpha=0.35, zorder=4)

    # Body frame arrows (store as quiver objects, recreated each frame)
    body_arrows = [None, None, None]
    ax_body_colors = ['#FF4444', '#44CC88', '#4488FF']
    ax_body_labels = ['b̂₁', 'b̂₂', 'b̂₃']

    # Mode indicator text
    mode_text  = ax3d.text2D(0.02, 0.97, '', transform=ax3d.transAxes,
                              fontsize=11, fontweight='bold', va='top',
                              color=MODE_COLORS[0])
    time_text  = ax3d.text2D(0.02, 0.90, '', transform=ax3d.transAxes,
                              fontsize=9, color='#aabbcc', va='top')
    sigma_text = ax3d.text2D(0.02, 0.84, '', transform=ax3d.transAxes,
                              fontsize=8, color='#ccddee', va='top',
                              fontfamily='monospace')

    # 2D scan lines (vertical time marker)
    vline_s = ax_s.axvline(0, color='#ffffff', lw=0.8, alpha=0.5)
    vline_w = ax_w.axvline(0, color='#ffffff', lw=0.8, alpha=0.5)
    vline_e = ax_e.axvline(0, color='#ffffff', lw=0.8, alpha=0.5)

    # 2D live traces (bright foreground)
    live_s = [ax_s.plot([],[],'-', color=cols2d[j], lw=1.5, alpha=0.9)[0] for j in range(3)]
    live_w = [ax_w.plot([],[],'-', color=cols2d[j], lw=1.5, alpha=0.9)[0] for j in range(3)]
    live_e_line, = ax_e.plot([],[],'-', color='#CC88EE', lw=1.6, alpha=0.9)

    # Mode shading rectangles on 2D plots (pre-drawn)
    def shade_modes_bg(ax, t_h, m_h):
        n=len(t_h); i=0
        while i<n:
            m=m_h[i]; j=i
            while j<n and m_h[j]==m: j+=1
            ax.axvspan(t_h[i],t_h[min(j,n-1)], alpha=0.07,
                       color=MODE_COLORS[m], linewidth=0)
            i=j
    for ax in [ax_s,ax_w,ax_e]:
        shade_modes_bg(ax,t_h,m_h)

    # Title
    title_text = fig.text(0.5, 0.97,
        'ASEN 5010 — Task 11: Mission Scenario Attitude Dynamics',
        ha='center', va='top', fontsize=12, fontweight='bold',
        color='#ddeeff')

    # Legend
    legend_entries = [
        mpatches.Patch(color='#FF4444', label='b̂₁  sensor / antenna'),
        mpatches.Patch(color='#44CC88', label='b̂₂  intermediate'),
        mpatches.Patch(color='#4488FF', label='b̂₃  solar panels'),
        mpatches.Patch(color=MODE_COLORS[0], label='Sun-pointing'),
        mpatches.Patch(color=MODE_COLORS[1], label='GMO-pointing'),
        mpatches.Patch(color=MODE_COLORS[2], label='Nadir-pointing'),
    ]
    ax3d.legend(handles=legend_entries, fontsize=7, loc='lower left',
                facecolor='#1a1a2e', edgecolor='#334', labelcolor='#ccc', ncol=2)

    # ── Animation update ─────────────────────────────────────
    def update(frame):
        nonlocal body_arrows

        fi    = frame % N_frames
        t_cur = t_anim[fi]
        mode  = m_anim[fi]
        s     = X_anim[fi, :3]
        w     = X_anim[fi, 3:]
        r     = rL_anim[fi]
        rg    = rG_anim[fi]

        # Body frame in N
        BN = mrp_to_dcm(s)
        NH = BN.T   # columns = b1,b2,b3 in N-frame

        # ── 3D update ─────────────────────────────────────────
        # Spacecraft dot
        sc_dot.set_data([r[0]],[r[1]]); sc_dot.set_3d_properties([r[2]])
        gmo_dot.set_data([rg[0]],[rg[1]]); gmo_dot.set_3d_properties([rg[2]])

        # Orbit trail
        start = max(0, fi-N_trail)
        trail_x = rL_anim[start:fi+1,0]
        trail_y = rL_anim[start:fi+1,1]
        trail_z = rL_anim[start:fi+1,2]
        trail_line.set_data(trail_x, trail_y)
        trail_line.set_3d_properties(trail_z)
        trail_line.set_color(MODE_COLORS[mode])

        # Line-of-sight indicators
        if mode == MODE_GMO:
            gmo_los.set_data([r[0],rg[0]],[r[1],rg[1]])
            gmo_los.set_3d_properties([r[2],rg[2]])
            gmo_los.set_alpha(0.5)
            nadir_los.set_alpha(0); sun_los.set_alpha(0)
        elif mode == MODE_NADIR:
            nadir_los.set_data([r[0],0],[r[1],0])
            nadir_los.set_3d_properties([r[2],0])
            nadir_los.set_alpha(0.45)
            gmo_los.set_alpha(0); sun_los.set_alpha(0)
        else:  # SUN
            sun_end = [r[0], r[1]+0.5, r[2]]
            sun_los.set_data([r[0],sun_end[0]],[r[1],sun_end[1]])
            sun_los.set_3d_properties([r[2],sun_end[2]])
            sun_los.set_alpha(0.5)
            gmo_los.set_alpha(0); nadir_los.set_alpha(0)

        # Body frame arrows — remove old, draw new
        for arr in body_arrows:
            if arr is not None:
                arr.remove()
        for j in range(3):
            d = NH[:, j]
            body_arrows[j] = ax3d.quiver(
                r[0], r[1], r[2],
                d[0]*AX_LEN, d[1]*AX_LEN, d[2]*AX_LEN,
                color=ax_body_colors[j],
                arrow_length_ratio=0.18,
                linewidth=2.2,
                zorder=15
            )

        # Mode / time text
        mode_text.set_text(f'● {MODE_NAMES[mode]}')
        mode_text.set_color(MODE_COLORS[mode])
        time_text.set_text(f't = {t_cur:.0f} s')
        sigma_text.set_text(
            f'σ = [{s[0]:+.3f}, {s[1]:+.3f}, {s[2]:+.3f}]\n'
            f'|σ_BR| = {np.linalg.norm(sBR_anim[fi]):.3f}'
        )

        # ── 2D plots update ───────────────────────────────────
        mask = t_h <= t_cur
        t_show = t_h[mask]
        for j in range(3):
            live_s[j].set_data(t_show, X_h[mask, j])
            live_w[j].set_data(t_show, np.degrees(X_h[mask, j+3]))
        live_e_line.set_data(t_show, np.linalg.norm(sBR_h[mask], axis=1))

        for vl in [vline_s, vline_w, vline_e]:
            vl.set_xdata([t_cur, t_cur])

        # Mode-coloured dot on 2D plots
        for ax,yval in [(ax_s, X_h[int(t_cur),0] if int(t_cur)<len(t_h) else X_h[-1,0]),
                         (ax_w, np.degrees(X_h[int(t_cur),3]) if int(t_cur)<len(t_h) else 0)]:
            pass  # dots would need scatter updates — kept simple

        return ([sc_dot, gmo_dot, trail_line, gmo_los, nadir_los, sun_los,
                 mode_text, time_text, sigma_text, vline_s, vline_w, vline_e]
                + live_s + live_w + [live_e_line] + body_arrows)

    # ── Run animation ─────────────────────────────────────────
    print(f"Animating {N_frames} frames (step={step}s per frame)...")

    ani = FuncAnimation(
        fig, update,
        frames=N_frames,
        interval=60,      # ms between frames (~16 fps)
        blit=False,       # blit=False needed for 3D quiver removal
        repeat=True
    )

    plt.show()
    return ani


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("ASEN 5010 — Task 11: 3D Attitude Animation")
    print("="*60)
    print(f"  Gains: K={K_GAIN:.6f} N·m,  P={P_GAIN:.6f} N·m·s")
    print(f"  Duration: 6500 s,  dt=1 s\n")

    t_h,X_h,u_h,m_h,sBR_h,rL_h,rG_h = simulate(6500.0, dt=1.0)
    ani = run_animation(t_h,X_h,u_h,m_h,sBR_h,rL_h,rG_h, step=25)
