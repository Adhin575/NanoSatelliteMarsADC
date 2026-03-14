ASEN 5010 — Attitude Dynamics and Control of a Nano-Satellite Orbiting Mars
Aerospace Engineering Sciences · University of Colorado, Boulder

Overview
This repository contains the complete Python implementation of the ASEN 5010 semester capstone project. A nano-satellite in a circular Low Mars Orbit (LMO) must autonomously switch between three attitude pointing modes — sun-pointing, nadir-pointing, and GMO communication-pointing — using a PD feedback controller driven by Modified Rodrigues Parameter (MRP) attitude tracking errors.
The full simulation runs 6,500 seconds at dt = 1 s using a hand-written 4th-order Runge-Kutta integrator (no scipy.integrate or equivalent). All 11 tasks are implemented, along with a full visualization suite and a live 3D animation.

Mission Description
A nano-satellite on a circular LMO (h = 400 km) must complete three mission objectives:
Orbital SituationPointing GoalSunlit side of Mars (r · n̂₂ > 0)Point solar panels b̂₃ at the Sun (n̂₂)Shadow side & GMO within 35° angular separationPoint antenna −b̂₁ at GMO mother spacecraftShadow side & GMO not visiblePoint sensor b̂₁ toward Mars nadir (−r̂)
A second satellite (the GMO mother spacecraft) orbits at the geosynchronous Mars radius of 20,424.2 km in the equatorial plane.

Repository Structure
.
├── Task1_Orbit Simulation.py                                                      # Circular orbit position & velocity via (3-1-3) Euler angles
├── Task2_Orbit Frame Simulation.py                                                # Hill frame DCM [HN](t) — analytic and numeric cross-check
├── Task3_Sun-Pointing Reference Frame Orientation.py                              # Constant sun-pointing reference frame [RsN]
├── Task4_Nadir-Pointing Reference Frame Orientation.py                            # Time-varying nadir frame [RnN](t) and NωRn/N
├── Task5_GMO-Pointing Reference Frame Orientation.py                              # GMO-pointing frame [RcN](t) with finite-difference ω
├── Task6_Attitude Error Evaluation.py                                             # MRP attitude error σB/R and angular velocity error BωB/R
├── Task7_Numerical Attitude Simulator.py.py                                       # RK4 integrator, EOM, torque-free & forced validation
├── Task8_Sun Pointing Control.py                                                  # PD sun-pointing control with gain design
├── Task9_Nadir Pointing Control.py                                                # PD nadir-pointing control
├── Task10_GMO Pointing Control.py                                                 # PD GMO-pointing control
├── Task11_Mission Scenario Simulation.py                                          # Full 6,500 s mission with autonomous mode switching
├── Visualization.py                                                               # 11 publication-quality plots (plt.show only, no saving)
├── Animation.py                                                                   # Live 3D animation with simultaneous time-series panels
└── README.md

Installation & Dependencies
Only the standard scientific Python stack is required — no external solvers or symbolic math libraries.
bashpip install numpy matplotlib
PackageMinimum VersionUsagenumpy1.21Array math, linear algebra, trigonometrymatplotlib3.4All 2D plots, 3D axes, FuncAnimationmpl_toolkits(bundled with matplotlib)Axes3D, Line3DCollection

Note: scipy, sympy, and casadi are not used. All integrators and math are implemented from scratch per the project specification.


Quick Start
bash# Run any individual task
python task1_orbit_simulation.py
python task8_sun_control.py
python task11_mission_scenario.py

# Display all 11 property plots (close each window to advance)
python task11_full_visualization.py

# Run the live 3D animation
python task11_animation.py

The full 6,500-second simulation with dt = 1 s takes approximately 5–10 seconds on a modern laptop. The animation loops automatically — close the window to exit.


Mission Parameters
Spacecraft Initial Conditions
σB/N(t₀) = [0.3, −0.4, 0.5]              (MRP)
BωB/N(t₀) = [1.00, 1.75, −2.20] deg/s
[I] = diag(10, 5, 7.5) kg·m²
Orbit Parameters
ParameterLMOGMORadius3,796.19 km (h = 400 km)20,424.2 kmΩ (RAAN)20°0°i (inclination)30°0°θ₀ (initial latitude)60°250°θ̇ (orbit rate)0.000884797 rad/s0.0000709003 rad/s

Physics & Mathematics
Coordinate Frames

N — Mars-centred inertial frame; Sun assumed along +n̂₂
H — Hill (orbit) frame; î_r = radial, î_θ = along-track, î_h = orbit normal; defined by (3-1-3) Euler angles (Ω, i, θ)
B — Spacecraft body frame; b̂₁ = sensor/antenna, b̂₃ = solar panel normal
R — Time-varying reference frame the controller tracks (Rs, Rn, or Rc)

Modified Rodrigues Parameters (MRP)
Attitude is represented by σ ∈ ℝ³, related to the principal rotation axis ê and angle Φ by:
σ = ê · tan(Φ/4)
Key operations implemented:

DCM from σ — Cayley transform: C = I + (8[σ̃]² − 4(1−|σ|²)[σ̃]) / (1+|σ|²)²
Kinematic ODE — σ̇ = ¼ B(σ) ω, where B(σ) = (1−|σ|²)I + 2[σ̃] + 2σσᵀ
Shadow set switching — when |σ| > 1, switch to σ* = −σ/|σ|² (applied after each full RK4 step, never during)
Attitude error — σB/R = dcm_to_mrp([BN][RN]ᵀ)

Equations of Motion
[I] ω̇ = −[ω̃][I]ω + u
σ̇   = ¼ B(σ) ω
PD Control Law (Eq. 4)
Bu = −K σB/R − P BωB/R
where BωB/R = BωB/N − [BN] NωR/N is the angular velocity tracking error expressed in the body frame.

Gain Design
The linearized closed-loop dynamics on each principal axis i satisfy:
Iᵢ σ̈ᵢ + P σ̇ᵢ + K σᵢ = 0
with natural frequency ωₙᵢ = √(K/Iᵢ), damping ratio ξᵢ = P/(2√(K·Iᵢ)), and time constant τᵢ = 2Iᵢ/P.
Constraint 1 — slowest time constant ≤ 120 s:
τ_max = 2·I_max / P ≤ 120 s  →  P = 2×10/120 = 1/6 ≈ 0.16667 N·m·s
Constraint 2 — all axes under-damped or critically damped (ξ ≤ 1):
ξ_max = P / (2√(K·I_min)) ≤ 1  →  K = P²/(4·I_min) = 1/720 ≈ 0.001389 N·m
Modal response:
AxisIᵢ (kg·m²)ωₙ (rad/s)ξτ (s)Character1 (b̂₁)10.00.0117850.707120.0Under-damped2 (b̂₂)5.00.0166671.00060.0Critically damped3 (b̂₃)7.50.0136080.81790.0Under-damped

The 1% settling time for the slowest axis (ξ = 0.707, τ = 120 s) is approximately 550 s, so all validation times at 400 s show the system still converging — this is physically correct.


Reference Frame Definitions
Sun-Pointing (Task 3)
r̂₁ = −n̂₁,   r̂₃ = n̂₂ (sun),   r̂₂ = r̂₃ × r̂₁ = n̂₃
[RsN] = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]    (constant)
NωRs/N = [0, 0, 0] rad/s
Nadir-Pointing (Task 4)
r̂₁ = −î_r (nadir),   r̂₂ = î_θ,   r̂₃ = r̂₁ × r̂₂ = −î_h
[RnN](t) = diag(−1, 1, −1) · [HN](t)
NωRn/N  = θ̇ · î_h   (orbit-normal rotation)
GMO-Pointing (Task 5)
Δr = r_GMO − r_LMO
r̂₁ = −Δr/|Δr|              (−r̂₁ points at GMO)
r̂₂ = (Δr × n̂₃)/|Δr × n̂₃|
r̂₃ = r̂₁ × r̂₂
NωRc/N computed via central finite differences on [RcN](t)

Mode Switching Logic
pythondef select_mode(t):
    r_lmo = pos_LMO(t)
    if r_lmo[1] > 0:                               # n̂₂ component > 0 → sunlit
        return MODE_SUN
    r_gmo = pos_GMO(t)
    angle = arccos(dot(r_lmo, r_gmo) / (|r_lmo| * |r_gmo|))
    if angle < 35°:                                # GMO within angular threshold
        return MODE_GMO
    return MODE_NADIR
Mission Timeline (6,500 s)
Time (s)EventMode0 → 1918LMO on sunlit side☀ Sun-pointing1918 → 3057Shadow side, GMO > 35°🌍 Nadir-pointing3057 → 4067Shadow side, GMO < 35°📡 GMO-pointing4067 → 5469Shadow side, GMO > 35°🌍 Nadir-pointing5469 → 6500LMO re-enters sunlit side☀ Sun-pointing

Validation Results
Task 7 — Torque-Free Integration (u = 0, t = 500 s)
σB/N(500 s) = [0.13765932,  0.56027024, −0.03217282]
BH(500 s)   = [0.13789720,  0.13266205, −0.31638781]  kg·m²/s
NH(500 s)   = [−0.26412650, 0.25278185,  0.05526875]  kg·m²/s
T(500 s)    = 0.0093841204 J   (relative drift: 4.06×10⁻¹¹ ✓)

σB/N(100 s, u=[0.01,−0.01,0.02] N·m) = [−0.22686111, −0.64138601, 0.24254980]
Task 11 — Full Mission σB/N (short MRP, |σ| ≤ 1)
t (s)Modeσ₁σ₂σ₃|σ|300Sun0.216957410.528132370.506898270.76352100Nadir−0.449077150.546262180.580550100.91493400GMO−0.34758518−0.146208980.297272240.48024400Nadir−0.27423034−0.400222750.234458830.53885600Sun−0.01704300−0.88249083−0.272611570.9238

Visualization Guide (task11_full_visualization.py)
Running task11_full_visualization.py opens 11 windows sequentially. Close each window to advance to the next.
#PlotWhat to Look For1MRP attitude σB/NComponents transitioning between modes; shadow-switch events appear as sign flips2Angular velocity ωB/NDecays from ±2.2 deg/s initial tumble toward orbit-rate values as controller settles3Control torque uLarge initial spike then exponential decay; resets at each mode switch4Tracking error |σB/R|Decreases monotonically within each mode; jumps at transitions are expected5Mode timelineColour-coded bar showing exact Sun / GMO / Nadir windows across 6,500 s6DashboardAll four states in one figure; grey dotted lines mark the five validation times7Kinetic energy & |H|T conserved in torque-free segments; |H| changes when control is active8MRP magnitude|σ| approaching 1 indicates shadow-switch events3D-1Orbit trajectoriesLMO path coloured by mode; GMO dashed ring; Mars sphere3D-2Body frame snapshotsb̂₁/b̂₂/b̂₃ arrows at each of the five validation times3D-3Axis unit-vector pathsWhere each body axis traces on the unit sphere — a classic attitude visualization
Mode colour coding (consistent across all plots):

🟡 Yellow #F4C542 — Sun-pointing
🔵 Blue #4A90D9 — GMO-pointing
🟢 Green #5BAA6E — Nadir-pointing


Animation Guide (task11_animation.py)
The animation opens a single window with two panels:
Left — 3D inertial viewport:

Mars rendered as a shaded sphere with atmosphere glow
Static dashed LMO and GMO orbit rings
White dot = spacecraft position; colour-coded orbit trail changes with active mode
Three live body-frame arrows rotating in real time:

🔴 Red b̂₁ — sensor / antenna axis
🟢 Green b̂₂ — intermediate axis
🔵 Blue b̂₃ — solar panel normal


Dashed line-of-sight indicator (yellow = Sun, blue = GMO, green = nadir)
Overlay text showing current mode, time, σ values, and tracking error magnitude

Right — three live time-series plots:

Faint full-trace background for mission context
Bright foreground trace grows in real time
White vertical scan-line tracks current animation time

Adjustable parameters in run_animation():
ParameterDefaultEffectstep25Simulation seconds per frame — lower = slower, smootherinterval60 msMilliseconds between frames — lower = faster playbackN_trail30Number of frames in the spacecraft orbit trailAX_LEN0.28Body-frame arrow length in display units

The 3D viewport remains mouse-interactive during animation. Click and drag to rotate; scroll to zoom.


RK4 Integrator Notes
The integrator follows Algorithm 1 from the project specification exactly:

Evaluate the reference frame and compute control torque u at the start of each step
Hold u piecewise-constant across the entire RK4 sub-step evaluations
Advance the full state X = [σ, ω] with a single RK4 call
Apply MRP shadow switching (|σ| > 1 → σ* = −σ/|σ|²) after the full step — never during

pythonk1 = eom(Xn, u)
k2 = eom(Xn + k1/2,  u)
k3 = eom(Xn + k2/2,  u)
k4 = eom(Xn + k3,    u)
Xn+1 = Xn + (dt/6)(k1 + 2k2 + 2k3 + k4)
if |sigma| > 1: sigma = -sigma / |sigma|²



References

Schaub, H. and Junkins, J.L. (2018). Analytical Mechanics of Space Systems, 4th ed. AIAA Education Series.\n
Arya, V. (2024). ASEN 5010 Semester Project Specification — Attitude Dynamics and Control of a Nano-Satellite Orbiting Mars. University of Colorado Boulder.
