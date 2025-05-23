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
# Plot or no plot
plotflag = 1; 

# Plotting setup
if plotflag:
    plt.figure(figsize=(10, 6))

# Load in data from CSV
df = pd.read_csv('Example_control.csv')

# Split into Predator_free and Prey_free controls
df1 = df[df['Control_Type']=='Predator_free']
df2 = df[df['Control_Type'] == 'Prey_free'] 

# Check how many replicates and initialize output
Replicate_list = df1['Replicate'].unique()
PredFree_Outputs = []; 
Coeff_est = []; 

# Cycle through Predator Free replicates fitting individually 
for i in Replicate_list.tolist():
    df_rep = df1[df1['Replicate']==i] 
    t_sampled = df_rep['Time'].to_numpy()
    N_sample = df_rep['Prey'].to_numpy()
    p_sample = df_rep['Pred'].to_numpy()

    # Log-transform prey counts (avoid zeros)
    mask = N_sample > 0
    t_fit = t_sampled[mask]
    log_N = np.log(N_sample[mask])

    #check that control label is accurate
    if any(p_sample>0):
        print('Predators in Predator Free Control!')
        pdb.set_trace()

    # Fit linear model: log(N) = log(N0) + r * t  => exponential growth
    model = LinearRegression().fit(t_fit.reshape(-1, 1), log_N)
    r_est = model.coef_[0]
    PredFree_Outputs.append(r_est) # Save best fits to output 
  

    # Predict and plot this replicate
    if plotflag:
        t_pred = np.linspace(min(t_sampled), max(t_sampled), 100)
        N_pred = np.exp(model.intercept_ + r_est * t_pred)
        plt.plot(t_sampled, N_sample, 'o', label=f'Replicate {i}')
        plt.plot(t_pred, N_pred, '--', label=f'Fit Rep {i}')


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

# Now do the same thing but for Prey-Free Controls 
# Check how many replicates and initialize output
Replicate_list = df2['Replicate'].unique()
PreyFree_Outputs = []; 

# Cycle through Predator Free replicates fitting individually 
for i in Replicate_list.tolist():
    df_rep = df2[df2['Replicate']==i] 
    t_sampled = df_rep['Time'].to_numpy()
    N_sample = df_rep['Prey'].to_numpy()
    P_sample = df_rep['Pred'].to_numpy()

    # Log-transform prey counts (avoid zeros)
    mask = P_sample > 0
    t_fit = t_sampled[mask]
    log_P = np.log(P_sample[mask])

    #check that control label is accurate
    if any(N_sample>0):
        print('Predators in Predator Free Control!')
        pdb.set_trace()

    # Fit linear model: log(N) = log(N0) + r * t  => exponential growth
    model = LinearRegression().fit(t_fit.reshape(-1, 1), log_P)
    m_est = model.coef_[0]
    PreyFree_Outputs.append(m_est) # Save best fits to output 
  

    # Predict and plot this replicate
    if plotflag:
        t_pred = np.linspace(min(t_sampled), max(t_sampled), 100)
        P_pred = np.exp(model.intercept_ + m_est * t_pred)
        plt.plot(t_sampled, P_sample, 'o', label=f'Replicate {i}')
        plt.plot(t_pred, P_pred, '--', label=f'Fit Rep {i}')


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

