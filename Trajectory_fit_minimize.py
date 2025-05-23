import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit 
from scipy.optimize import minimize 
import pdb
import pandas as pd


#Load in data from CSV
df = pd.read_csv('Example_input2.csv')

# Ask user for initial guess 
#a_init = float(input("Enter your initial guess for attack rate, a \n"))
#h_init = float(input("Enter your initial guess for handling time, h \n"))
#f_init = float(input("Enter your initial guess for conversion efficiency, e \n"))
#r0 = float(input("Enter prey intrinsic growth rate, r \n"))
#m = float(input("Enter predator intrinsic mortality rate, m \n"))

#a_init = 0.1 
#h_init = 0.05
#f_init = 0.1
#r0 = 0.06
#m = 0.05

a_init = 0.0015 
h_init = 0.004
f_init = 0.006
r0 = 0.2
m = 0.1

 
initial_guess = [a_init, h_init, f_init]

#Define bounds for a, h, f
upper_bounds = [1, 1, 1]
lower_bounds = [0, 0, 0]

# Define the model equations
def lotka_volterra_type_ii(y, t, r, a, h, f, m):
    N, P = y
    dN_dt = r * N - (a * N * P) / (1 + a * h * N)
    dP_dt = (f * a * N * P) / (1 + a * h * N) - m * P
    return [dN_dt, dP_dt]

# Model function for fitting
def model_func(t, a, h, f):
    y0 = [np.exp(N_sample_log[0]), np.exp(P_sample_log[0])]
    solution = odeint(lotka_volterra_type_ii, y0, t, args=(r0, a, h, f, m))
    N_log, P_log = np.log(solution.T)
    return N_log, P_log

# Objective function
# Working from GUTS txt eq. 3.34
def neg_likelihood(params, x_data, N_data, P_data): 
    a, h, f = params

    # calculate model predictions
    N_pred, P_pred = model_func(x_data,a,h,f)

    # need n, number of obs (half length of y since we have pred and prey for both)
    n = len(N_pred)  

    # need sum of squared residuals for each y variable
    residuals_N = N_data - N_pred
    residuals_P = P_data - P_pred

    # Have the option to specify sigma differently for the two groups 
    nll = n/2*np.log(np.sum((residuals_N) **2)) + n/2*np.log(np.sum((residuals_P) **2))

    return nll


# Check how many replicates and initialize output
Replicate_list = df['Replicate'].unique()
Outputs = []; 

# Cycle through fitting replicates individually 
for i in Replicate_list.tolist():
    df1 = df[df['Replicate']==i] 
    df1.to_numpy
    t_sampled = df1.iloc[:,1].to_numpy()
    N_sample_log = np.log(df1.iloc[:,2].to_numpy())
    P_sample_log = np.log(df1.iloc[:,3].to_numpy())

    # Find N0 and P0 
    initial_conditions = df1[df1.iloc[:,1]==0] 
    N0_log = np.log(initial_conditions.iloc[:,2].item())
    P0_log = np.log(initial_conditions.iloc[:,3].item())

    # smooth out time that we want trajectory over 
    t = np.linspace(0, np.max(t_sampled), 1000)

    # Fit the parameters a, h, and f
    result = minimize(neg_likelihood, initial_guess, args =(t_sampled, N_sample_log, P_sample_log), method = 'Nelder-Mead')

    # Save best fits to output 
    a_fit, h_fit, f_fit = result.x
    row = [i, a_init, h_init, f_init, r0, m, a_fit, h_fit, f_fit]; 
    Outputs.append(row)
    print(f"Fitted parameters:\na = {a_fit}\nh = {h_fit}\nf = {f_fit}")

    # Simulate model with initial guess
    solution_initial = odeint(lotka_volterra_type_ii, [np.exp(N0_log), np.exp(P0_log)], t, args=(r0, *initial_guess, m))
    N_initial, P_initial = solution_initial.T

    # Simulate model with fitted parameters
    solution_fit = odeint(lotka_volterra_type_ii, [np.exp(N0_log), np.exp(P0_log)], t, args=(r0, a_fit, h_fit, f_fit, m))
    N_fit, P_fit = solution_fit.T


    # Plot the results
    plt.figure(figsize=(12, 6))

    # Measured data
    plt.plot(t_sampled, np.exp(N_sample_log), 'bo', label='Prey Data')
    plt.plot(t_sampled, np.exp(P_sample_log), 'ro', label='Predator Data')


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

