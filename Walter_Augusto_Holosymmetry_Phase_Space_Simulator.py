# 📦 Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 🛠️ Define Parameters
kappa = 1.0      # Inertia for φ
Lambda4 = 1.0    # Breathing potential energy scale Λ⁴
g0 = 0.5         # Coupling constant for χ
gamma_pp = 0.1   # Coupling strength between φ and χ

# 🧮 Define the Coupled Differential Equations
def breathing_system(t, y):
    phi, phi_dot, chi, chi_dot = y  # unpack variables
    
    # Equations of motion
    phi_ddot = (np.sin(phi) * (Lambda4 + 0.5 * g0 * chi**2) + gamma_pp * chi * Lambda4 * np.cos(phi)) / kappa
    chi_ddot = -g0 * (1 + np.cos(phi)) * chi + gamma_pp * Lambda4 * np.sin(phi)
    
    return [phi_dot, phi_ddot, chi_dot, chi_ddot]

# 🧷 Initial Conditions
phi0 = 0.01      # φ starts slightly off 0
phi_dot0 = 0.0   # φ initially at rest
chi0 = 0.0       # χ starts at 0
chi_dot0 = 0.0   # χ initially at rest
y0 = [phi0, phi_dot0, chi0, chi_dot0]

# 🕒 Time Span
t_span = (0, 100)                     # simulate from t=0 to t=100 (arbitrary units)
t_eval = np.linspace(t_span[0], t_span[1], 5000)  # fine-grained time evaluation points

# 🧠 Solve the System
solution = solve_ivp(breathing_system, t_span, y0, t_eval=t_eval, method='RK45')

# 🎯 Extract Results
t = solution.t
phi = solution.y[0]
chi = solution.y[2]

# 📈 Plot the Breathing Field ϕ(t)
plt.figure(figsize=(10, 5))
plt.plot(t, phi, label=r'Breathing Field $\varphi(t)$', color='orange')
plt.title("Breathing Field Evolution Over Time")
plt.xlabel("Time")
plt.ylabel(r"$\varphi(t)$")
plt.grid(True)
plt.legend()
plt.show()

# 📈 Plot the Compensator Field χ(t)
plt.figure(figsize=(10, 5))
plt.plot(t, chi, label=r'Compensator Field $\chi(t)$', color='purple')
plt.title("Compensator Field Evolution Over Time")
plt.xlabel("Time")
plt.ylabel(r"$\chi(t)$")
plt.grid(True)
plt.legend()
plt.show()

# 📈 Phase Space Plot (ϕ vs ϕ̇)
plt.figure(figsize=(6, 6))
plt.plot(phi, solution.y[1], color='blue')
plt.title(r"Phase Space: $\varphi$ vs $\dot{\varphi}$")
plt.xlabel(r"$\varphi$")
plt.ylabel(r"$\dot{\varphi}$")
plt.grid(True)
plt.show()

# 📈 Phase Space Plot (χ vs χ̇)
plt.figure(figsize=(6, 6))
plt.plot(chi, solution.y[3], color='green')
plt.title(r"Phase Space: $\chi$ vs $\dot{\chi}$")
plt.xlabel(r"$\chi$")
plt.ylabel(r"$\dot{\chi}$")
plt.grid(True)
plt.show()
