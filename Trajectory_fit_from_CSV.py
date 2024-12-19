import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import pdb
import pandas as pd



#Load in data from CSV
df = pd.read_csv('Example_input.csv')

# Ask user for initial guess 
a_init = float(input("Enter your initial guess for attack rate, a \n"))
h_init = float(input("Enter your initial guess for handling time, h \n"))
f_init = float(input("Enter your initial guess for conversion efficiency, e \n"))
r0 = float(input("Enter prey intrinsic growth rate, r \n"))
m = float(input("Enter predator intrinsic mortality rate, m \n"))


initial_guess = [a_init, h_init, f_init]


# Define the model equations
def lotka_volterra_type_ii(y, t, r, a, h, f, m):
    N, P = y
    dN_dt = r * N - (a * N * P) / (1 + a * h * N)
    dP_dt = (f * a * N * P) / (1 + a * h * N) - m * P
    return [dN_dt, dP_dt]


# Objective function for fitting
def model_func(t, a, h, f):
    y0 = [N_sample[0], P_sample[0]]
    solution = odeint(lotka_volterra_type_ii, y0, t, args=(r0, a, h, f, m))
    N, P = solution.T
    return np.concatenate([N, P])

# Check how many replicates and initialize output
Replicate_list = df['Replicate'].unique()
Outputs = []; 

# Cycle through fitting replicates individually 

for i in Replicate_list.tolist():
    df1 = df[df['Replicate']==i] 
    df1.to_numpy
    t_sampled = df1.iloc[:,1].to_numpy()
    N_sample = df1.iloc[:,2].to_numpy()
    P_sample = df1.iloc[:,3].to_numpy()

    # Find N0 and P0 
    initial_conditions = df1[df1.iloc[:,1]==0] 
    N0 = initial_conditions.iloc[:,2].item()
    P0 = initial_conditions.iloc[:,3].item()

    # Prepare data for fitting
    y_data = np.concatenate([N_sample, P_sample])

    # smooth out time that we want trajectory over 
    t = np.linspace(0, np.max(t_sampled), 1000)

    # Fit the parameters a, h, and f
    params_opt, params_cov = curve_fit(
        lambda t, a, h, f: model_func(t_sampled, a, h, f),
        np.concatenate([t_sampled, t_sampled]),
        y_data,
        p0=initial_guess
    )

    # Save best fits to output 
    a_fit, h_fit, f_fit = params_opt
    row = [i, a_init, h_init, f_init, r0, m, a_fit, h_fit, f_fit]; 
    Outputs.append(row)
    print(f"Fitted parameters:\na = {a_fit}\nh = {h_fit}\nf = {f_fit}")

    # Simulate model with initial guess
    solution_initial = odeint(lotka_volterra_type_ii, [N0, P0], t, args=(r0, *initial_guess, m))
    N_initial, P_initial = solution_initial.T

    # Simulate model with fitted parameters
    solution_fit = odeint(lotka_volterra_type_ii, [N0, P0], t, args=(r0, a_fit, h_fit, f_fit, m))
    N_fit, P_fit = solution_fit.T

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Measured data
    plt.plot(t_sampled, N_sample, 'bo', label='Prey Data I brojke it!')
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
    

Outputs2 = pd.DataFrame(Outputs, columns = ['Replicate', 'a_init', 'h_init', 'f_init', 'r', 'm', 'a_fit', 'h_fit', 'f_fit'])
Outputs2.to_csv('Fit_parameters.csv')

plt.show()

