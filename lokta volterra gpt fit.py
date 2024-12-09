import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit

# Define the model equations
def lotka_volterra_type_ii(y, t, r, a, h, f, m):
    N, P = y
    dN_dt = r * N - (a * N * P) / (1 + a * h * N)
    dP_dt = (f * a * N * P) / (1 + a * h * N) - m * P
    return [dN_dt, dP_dt]

# Generate data with noise
def generate_noisy_data(t, r, a, h, f, m, N0, P0, num_points=10, noise_level=0):
    y0 = [N0, P0]
    solution = odeint(lotka_volterra_type_ii, y0, t, args=(r, a, h, f, m))
    N, P = solution.T
    indices = np.linspace(0, len(t)-1, num_points).astype(int)
    N_sample = N[indices] + np.random.normal(0, noise_level, num_points)
    P_sample = P[indices] + np.random.normal(0, noise_level, num_points)
    return t[indices], N_sample, P_sample

# Objective function for fitting
def model_func(t, a, h, f):
    y0 = [N_sample[0], P_sample[0]]
    solution = odeint(lotka_volterra_type_ii, y0, t, args=(r, a, h, f, m))
    N, P = solution.T
    return np.concatenate([N, P])

# True parameters
r = 0.1
a_true = 0.1
h_true = 0.05
f_true = 0.1
m = 0.1
N0 = 4
P0 = 1

# Generate noisy data
t = np.linspace(0, 200, 1000)
t_sampled, N_sample, P_sample = generate_noisy_data(t, r, a_true, h_true, f_true, m, N0, P0)

# Prepare data for fitting
y_data = np.concatenate([N_sample, P_sample])

# Fit the parameters a, h, and f
initial_guess = [0.15, 0.02, 0.05]
params_opt, params_cov = curve_fit(
    lambda t, a, h, f: model_func(t_sampled, a, h, f),
    np.concatenate([t_sampled, t_sampled]),
    y_data,
    p0=initial_guess
)

# Extract fitted parameters
a_fit, h_fit, f_fit = params_opt
print(f"Fitted parameters:\na = {a_fit}\nh = {h_fit}\nf = {f_fit}")

# Simulate model with initial guess
solution_initial = odeint(lotka_volterra_type_ii, [N_sample[0], P_sample[0]], t, args=(r, *initial_guess, m))
N_initial, P_initial = solution_initial.T

# Simulate model with fitted parameters
solution_fit = odeint(lotka_volterra_type_ii, [N_sample[0], P_sample[0]], t, args=(r, a_fit, h_fit, f_fit, m))
N_fit, P_fit = solution_fit.T

# Plot the results
plt.figure(figsize=(12, 6))

# Noisy data
plt.plot(t_sampled, N_sample, 'bo', label='Prey Data')
plt.plot(t_sampled, P_sample, 'ro', label='Predator Data')


# Initial guess model
plt.plot(t, N_initial, 'b--', label='Initial Guess Prey', alpha=0.7)
plt.plot(t, P_initial, 'r--', label='Initial Guess Predator', alpha=0.7)

# Fitted model
plt.plot(t, N_fit, 'b-', label='Fitted Prey Population')
plt.plot(t, P_fit, 'r-', label='Fitted Predator Population')

plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('Predator-Prey Model Fit and Data with Initial Guess')
plt.show()
