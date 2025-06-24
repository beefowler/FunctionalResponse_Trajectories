import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit 
from scipy.optimize import minimize 
import pdb
import pandas as pd
import sklearn 
from sklearn.linear_model import LinearRegression

pdb
#Choose if you will plot or not
plotflag = 0; 

#Load in data from CSV
big_df = pd.read_csv('Template_data.csv')

# Figure out number of identifiers
e_id = list((big_df.filter(like='Experiment_id').columns))

big_df["Experiment_ID"] = big_df.groupby(e_id).ngroup()
num_experiments = big_df["Experiment_ID"].nunique()
print(f"there are {num_experiments} experiments.")

#Figure out column headers we will use
timepoints = list((big_df.filter(like='Time_').columns))
preypoints = list((big_df.filter(like='Prey_').columns))
predpoints = list((big_df.filter(like='Pred_').columns))
Outputs = []; 

# Define the model equations
def lotka_volterra_type_ii(y, t, r, a, h, e, m):
    P, Z = y
    dP_dt = r * P - (a * P * Z) / (1 + a * h * P)
    dZ_dt = (e * a * P * Z) / (1 + a * h * P) - m * Z
    return [dP_dt, dZ_dt]

# Model function for fitting
def model_func(t, a, h, e):
    y0 = [np.exp(P_sample_log[0]), np.exp(Z_sample_log[0])]
    solution = odeint(lotka_volterra_type_ii, y0, t, args=(r0, a, h, e, m))
    P_log, Z_log = np.log(solution.T)
    return P_log, Z_log

# Objective function
# Working from GUTS txt eq. 3.34
def neg_likelihood(params, x_data, P_data, Z_data): 
    a, h, e = params

    # calculate model predictions
    P_pred, Z_pred = model_func(x_data,a,h,e)

    # need n, number of obs
    n = len(P_pred)  

    # need sum of squared residuals for each y variable
    residuals_P = P_data - P_pred
    residuals_Z = Z_data - Z_pred

    # Have the option to specify sigma differently for the two groups 
    nll = n/2*np.log(np.sum((residuals_P) **2)) + n/2*np.log(np.sum((residuals_Z) **2))

    return nll

# Go through separate experiments 
for xp in big_df["Experiment_ID"].unique():

    #cut dataframe down to relevant rows
    small_df = big_df[big_df["Experiment_ID"] == xp] 
    controls = small_df[small_df["Control_Type"] == 'Predator_Free']

    #initialize outputs 
    PredFree_Outputs = []; 
    Coeff_est = []; 

    #cycle through individual control bottles to calculate growth rate 
    for b, row in controls.iterrows():  

        t_sampled = row[timepoints].to_numpy(dtype = float)
        P_sampled = row[preypoints].to_numpy(dtype = float)
        Z_sampled = row[predpoints].to_numpy(dtype = float)

        #pdb.set_trace()

        # Log-transform prey counts (avoid zeros)
        mask = P_sampled > 0
        t_fit = t_sampled[mask]
        log_P = np.log(P_sampled[mask])

        #check that control label is accurate
        if any(Z_sampled>0):
            print('Predators in Predator Free Control!')
            pdb.set_trace()

        # Fit linear model: log(N) = log(N0) + r * t  => exponential growth
        model = LinearRegression().fit(t_fit.reshape(-1, 1), log_P)
        r_est = model.coef_[0]
        PredFree_Outputs.append(r_est) # Save best fits to output 
    

        # Predict and plot this replicate
        if plotflag:
            t_pred = np.linspace(min(t_sampled), max(t_sampled), 100)
            P_pred = np.exp(model.intercept_ + r_est * t_pred)
            plt.plot(t_sampled, N_sample, 'o', label=f'Replicate {i}')
            plt.plot(t_pred, P_pred, '--', label=f'Fit Rep {i}')


        # Plot average growth rate model
        if plotflag:
            avg_r = np.mean(PredFree_Outputs)

            # Estimate initial N0 from t=0 values
            initial = df1[df1['Time'] == 0]
            mean_initial_P = np.mean(initial['Prey'])

            t_pred = np.linspace(min(t_sampled), max(t_sampled), 100)
            avg_pred = mean_initial_P * np.exp(avg_r * t_pred)
            plt.plot(t_pred, avg_pred, 'k-', linewidth=2, label='Avg growth fit')

            plt.xlabel('Time')
            plt.ylabel('Prey count')
            plt.title('Exponential Growth in Predator-Free Controls')
            plt.legend()
            plt.tight_layout()
            plt.show()


    #Save final estimate of r 
    Coeff_est.append(np.mean(PredFree_Outputs))
    r0 = np.mean(PredFree_Outputs)

    # Now do the same thing but for Prey-Free Controls 
    controls2 = small_df[small_df["Control_Type"] == 'Prey_Free']
    PreyFree_Outputs = []; 

    #cycle through individual control bottles to calculate mortality rate 
    for b, row in controls2.iterrows():  

        t_sampled = row[timepoints].to_numpy(dtype = float)
        P_sampled = row[preypoints].to_numpy(dtype = float)
        Z_sampled = row[predpoints].to_numpy(dtype = float)

        #pdb.set_trace()

        # Log-transform prey counts (avoid zeros)
        mask = P_sampled > 0
        t_fit = t_sampled[mask]
        log_P = np.log(P_sampled[mask])

        # Log-transform predator counts (avoid zeros)
        mask = Z_sampled > 0
        t_fit = t_sampled[mask]
        log_Z = np.log(Z_sampled[mask])

         #check that control label is accurate
        if any(P_sampled>10000):
            print('Over 10000 prey in Prey Free Control ...is that ok?')
            pdb.set_trace()

        # Fit linear model: log(N) = log(N0) + r * t  => exponential growth
        model = LinearRegression().fit(t_fit.reshape(-1, 1), log_Z)
        m_est = model.coef_[0]
        PreyFree_Outputs.append(m_est) # Save best fits to output 
    

        # Predict and plot this replicate
        if plotflag:
            t_pred = np.linspace(min(t_sampled), max(t_sampled), 100)
            Z_pred = np.exp(model.intercept_ + m_est * t_pred)
            plt.plot(t_sampled, Z_sampled, 'o', label=f'Replicate {i}')
            plt.plot(t_pred, Z_pred, '--', label=f'Fit Rep {i}')


    # Plot average mortality rate model
    if plotflag:
        avg_m = np.mean(PreyFree_Outputs)

        # Estimate initial N0 from t=0 values
        initial = df2[df2['Time'] == 0]
        mean_initial_P = np.mean(initial['Pred'])

        t_pred = np.linspace(min(t_sampled), max(t_sampled), 100)
        avg_pred = mean_initial_P * np.exp(avg_m * t_pred)
        plt.plot(t_pred, avg_pred, 'k-', linewidth=2, label='Avg growth fit')

        plt.xlabel('Time')
        plt.ylabel('Predator count')
        plt.title('Exponential Decay in Prey-Free Controls')
        plt.legend()
        plt.tight_layout()
        plt.show()

    Coeff_est.append(np.mean(PreyFree_Outputs))
    m = np.mean(PreyFree_Outputs)

    ask_user = 0 
    if ask_user:
        # Ask user for initial guess 
        a_init = float(input("Enter your initial guess for attack rate, a \n"))
        h_init = float(input("Enter your initial guess for handling time, h \n"))
        e_init = float(input("Enter your initial guess for conversion efficiency, e \n"))

    if not ask_user: 
        a_init = 0.0015 
        h_init = 0.004
        e_init = 0.006

    #a_init = 0.1 
    #h_init = 0.05
    #e_init = 0.1
    #r0 = 0.06
    #m = 0.05

 
    initial_guess = [a_init, h_init, e_init]

    #cut dataframe down to relevant rows
    treatments = small_df[small_df["Control_Type"] == 'Treatment']

    #cycle through individual treatment bottles fitting replicates individually 
    for b, row in treatments.iterrows():  

        experiment_id = row["Experiment_ID"]

        t_sampled = row[timepoints].to_numpy(dtype = float)
        P_sampled = row[preypoints].to_numpy(dtype = float)
        Z_sampled = row[predpoints].to_numpy(dtype = float)

        #log scale data
        P_sample_log = np.log(P_sampled)
        Z_sample_log = np.log(Z_sampled)

        # Find N0 and P0 
        P0_log = P_sample_log[t_sampled ==0]
        Z0_log = Z_sample_log[t_sampled ==0]

        # smooth out time that we want trajectory over 
        t = np.linspace(0, np.max(t_sampled), 1000)

        # Fit the parameters a, h, and e
        result = minimize(neg_likelihood, initial_guess, args =(t_sampled, P_sample_log, Z_sample_log), method = 'Nelder-Mead', bounds = ((0, 1), (0,1),(0,1)))

        # Save best fits to output 
        a_fit, h_fit, e_fit = result.x
        row = [experiment_id, b, a_init, h_init, e_init, r0, m, a_fit, h_fit, e_fit]; 
        Outputs.append(row)
        print(f"Fitted parameters:\na = {a_fit}\nh = {h_fit}\nf = {e_fit}")

        # Simulate model with initial guess
        solution_initial = odeint(lotka_volterra_type_ii, np.concatenate((np.exp(P0_log), np.exp(Z0_log))), t, args=(r0, *initial_guess, m))
        P_initial, Z_initial = solution_initial.T

        # Simulate model with fitted parameters
        solution_fit = odeint(lotka_volterra_type_ii, np.concatenate((np.exp(P0_log), np.exp(Z0_log))), t, args=(r0, a_fit, h_fit, e_fit, m))
        P_fit, Z_fit = solution_fit.T


        # Plot the results
        plt.figure(figsize=(12, 6))

        # Measured data
        plt.plot(t_sampled, np.exp(P_sample_log), 'bo', label='Prey Data')
        plt.plot(t_sampled, np.exp(Z_sample_log), 'ro', label='Predator Data')


        # Initial guess model
        plt.plot(t, P_initial, 'b--', label='Initial Guess Prey', alpha=0.7)
        plt.plot(t, Z_initial, 'r--', label='Initial Guess Predator', alpha=0.7)

        # Fitted model
        plt.plot(t, P_fit, 'b-', label='Fitted Prey Population')
        plt.plot(t, Z_fit, 'r-', label='Fitted Predator Population')

        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.legend()
        plt.title('Predator-Prey Model Fit and Data with Initial Guess')
        #plt.show()


    #pdb.set_trace()
  

Outputs2 = pd.DataFrame(Outputs, columns = ['Experiment_ID', 'bottle_number', 'a_init', 'h_init', 'e_init', 'r0', 'm', 'a_fit', 'h_fit', 'e_fit'])
Outputs2.to_csv('TrajectoryFit_parameter_results_.csv')
