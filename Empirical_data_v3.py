import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.optimize import minimize
from scipy.stats import norm
import time
import warnings

### IMPORT DATA

#Timer start. It usually takes a few seconds to run this script.
start_time = time.process_time()
warnings.filterwarnings("ignore")

#File path with options data to import as dataframe calls_data.
path = "C:/Users/giuse/Desktop/Tesi/Options_19_07_31.xlsx"
calls_data = pd.read_excel(path)

#Current price of the index, interest and implied dividend rates.
#Attention they are written from the farthest to the nearest maturity.
#The implied dividend rates are obtained considering using the Put-Call parity for a specific value of the strike.
risk = calls_data['Risk free rate'].to_numpy(dtype=float)
dividends = calls_data['Dividend'].to_numpy(dtype=float)
filter_rates = ~np.isnan(risk)
filter_maturity =  (calls_data['Maturities'] <= 1.2) & (calls_data['Maturities'] >= 0.15)
filter_maturity = filter_maturity[filter_rates]

r = risk[filter_rates][filter_maturity]
q = dividends[filter_rates][filter_maturity]
S = calls_data['Spot'].to_numpy(dtype=float)[0]

#Filter options data based on some conditions regarding mid price, moneyness,implied volatility, maturity in years and type.
calls_data = calls_data[(calls_data['Mid'] >= 1) &
                        (calls_data['Implied Volatility'] >= 0.05) &
                        (calls_data['Implied Volatility'] <= 0.9) &
                        (calls_data['Moneyness'] >= 0.9) & 
                        (calls_data['Moneyness'] <= 1.1) &
                        (calls_data['Maturity'] <= 1.2) &
                        (calls_data['Maturity'] >= 0.15) &
                        (calls_data['Type'] == 'Call') ]

#Import Bid, Ask, Mid prices and Strikes.
calls_bid = calls_data['Bid'].to_numpy(dtype=float)
calls_ask = calls_data['Ask'].to_numpy(dtype=float)
calls_column = calls_data['Mid'].to_numpy(dtype=float)
strikes_column = calls_data['Strike'].to_numpy(dtype=float)
                 
#Import and count the maturities present in the dataset.
maturities = calls_data['Maturity'].value_counts()
maturities_sorted = maturities.sort_index(ascending=False)
date_counts = maturities_sorted.to_numpy()
max_dates = max(date_counts)
dates = calls_data['Date'].to_numpy()
number_curves = len(date_counts)

#Maturities in year: attention they are written from the farthest to the nearest maturity.
tau = maturities_sorted.index.to_numpy()

#Definition of the call prices, strikes and number of strikes for each maturity (date_count).
calls = np.zeros((max_dates,number_curves))
strikes = np.zeros((max_dates,number_curves))
date_count = 0

#Statistics of empirical data: observation (date_counts), number of days (days), average strike spacing (avr_spacing), price percentile table (percent_table).

for i in range(number_curves): 
    for j in range(date_counts[i]):
        calls[j,i] = calls_column[j+date_count]
        strikes[j,i] = strikes_column[j+date_count]
        
    date_count += date_counts[i]
    
days = tau*365
avg_spacing = np.zeros(number_curves)
percent_table = np.zeros((5,number_curves))

for i in range(number_curves):
    non_zero_strikes = strikes[strikes[:, i] > 0, i]
    non_zero_calls = calls[calls[:, i] > 0, i]
    avg_spacing[i] = np.mean(np.diff(non_zero_strikes))
    percent_table[:,i] = np.percentile(non_zero_calls, [5, 10, 50, 90 ,95],axis=0)
    
#Implied forward rates.
F = [S * np.exp((r[i] - q[i]) * tau[i]) for i in range(number_curves)]


#Define the forward strikes (m), the maximum and minimum for each maturity (mmin,mmax) and the relative call prices (C_rel).
m = np.zeros((max_dates,number_curves))
mmin,mmax = np.zeros(number_curves), np.zeros(number_curves)
C_rel = np.zeros((max_dates,number_curves))

for i in range(number_curves):
    for j in range(max_dates):
        m[j,i] = strikes[j,i]/F[i]
        
for i in range(number_curves):
    mmin[i] = m[0,i]
    mmax[i] = m[date_counts[i]-1,i]
        
for i in range(number_curves):
    for j in range(max_dates):
        C_rel[j,i] = calls[j,i]*np.exp(r[i]*tau[i])/F[i]
  
#Maximum number of iteration referred to the highest possible Bernstein polynomial degree.
max_iter = 40

### DEFINITION OF THE RELEVANT FUNCTIONS

#Normalization of the strikes for each maturity.

def normalization(K,K_min,K_max):
    
    K_norm = np.zeros((max_dates,number_curves))
    for i in range(number_curves):
        for j in range(max_dates):
            K_norm[j,i] = (K[j,i] - K_min[i]) / (K_max[i] - K_min[i])
    
    return K_norm

#Computation of each Bernstein monomial.

def Bernstein(N, deg, val):
    value = comb(N, deg) * (val ** deg) * ((1 - val) ** (N - deg))
    return value

#Computation of the SSE, needed for the minimization procedure.

def SSE(C, B, betas):
    Bbetas = np.matmul(B, betas)
    errors = Bbetas - C
    SSE = np.sum(errors ** 2)
    return SSE

#The Information ratio is computed, needed to establish the optimal degree.

def AIC(C, B, betas, num_call, N):
    C_filter = C[C > 0]
    Information_ratio = num_call * np.log(SSE(C_filter, B, betas) / num_call) + 2 * (N + 1)
    return Information_ratio

#Determine if the matrix is positive definitive, fundamental property for QP optimization.

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

#Define the B matrix.

def B(K, N, num_call):
    B = np.zeros((num_call, N + 1))
    for i in range(N + 1):
        for j in range(num_call):
            B[j, i] = Bernstein(N, i, K[j])
    return B

#Define the f vector.

def f(B, C):
    C_filter = C[C > 0]
    f = np.matmul(np.transpose(B), C_filter)
    return f

#Define the Hessian.

def Hessian(B):
    H = np.matmul(np.transpose(B), B)
    return H

#Define the A matrix. The matrix has to be defined iteratively based on the previous maturity optimal Bernstein degree.

def A(N,N_prev,t, first):
    
    if first == 0:
        
        #If this is the first and farthest maturity, define the A matrix as in the 1D case.
        AN = np.zeros((N + 4, N + 1))
        for i in range(N - 1):
            AN[i, i] = 1
            AN[i, i + 1] = -2
            AN[i, i + 2] = 1
        AN[-1, 0] = -1
        AN[-2, 0] = 1
        AN[-3, 0] = -1
        AN[-3, 1] = 1
        AN[-4, -1] = 1
        AN[-5, -1] = -1
        AN[-5, -2] = 1
        
    else: 
        if N < N_prev:
            
            #If this is not the first and farthest maturity, and the current degree is lower than the optimal previous one,
            #a degree elevation procedure for the current maturity needs to be performed.
            
            AN = np.zeros((N + 3 + N_prev + 1, N + 1))
            for i in range(N - 1):
                AN[i, i] = 1
                AN[i, i + 1] = -2
                AN[i, i + 2] = 1
                
            AN[-2 - N_prev, 0] = 1
            AN[-3 - N_prev, 0] = -1
            AN[-3 - N_prev, 1] = 1
            AN[-4 - N_prev, -1] = 1
            AN[-5 - N_prev, -1] = -1
            AN[-5 - N_prev, -2] = 1
                
            for k in range(N_prev+1):
                
                r = N_prev - N
                upper = min(N,k)
                lower = max(0,k-r)
                
                    
                for l in range(lower,upper+1):

                    AN[N+3+k,l] = -comb(r, k-l)*comb(N,l)/comb(N+r,k)
            
                    
        else:
            
            #If this is not the first and farthest maturity, and the current degree is higher than the optimal previous one,
            #a degree elevation procedure for the previous maturity needs to be performed.
            
            AN = np.zeros((N + 3 + N + 1, N + 1))
            for i in range(N - 1):
                AN[i, i] = 1
                AN[i, i + 1] = -2
                AN[i, i + 2] = 1
                
            AN[-2 - N, 0] = 1
            AN[-3 - N, 0] = -1
            AN[-3 - N, 1] = 1
            AN[-4 - N, -1] = 1
            AN[-5 - N, -1] = -1
            AN[-5 - N, -2] = 1
            
            for j in range(N+1):
                AN[N+3+j,j] = -1 
            
          
    return AN

#Define the d matrix. The matrix has to be defined iteratively based on the previous maturity optimal Bernstein degree.

def d(N, tau, S, r, q, Kmin, Kmax, t, N_prev, beta_prev, m_min, m_max, first):
    
    N_prev = int(N_prev)

    if first == 0:
        
        #If this is the first and farthest maturity, define the d matrix as in the 1D case.
        dN = np.zeros(N + 4)
        dN[-1] = -1
        dN[-2] = 1 - m_min
        dN[-3] = -(m_max - m_min)/N
        
    else:
        
        if N > N_prev:
            
            #If this is not the first and farthest maturity, and the current degree is higher than the optimal previous one,
            #a degree elevation procedure for the previous maturity needs to be performed.
            
            dN_std = np.zeros(N + 3 + N + 1)   
            dN_std[-2 - N] = 1 - m_min
            dN_std[-3 - N] = -(m_max - m_min)/N
            dN_partial = np.zeros((N+1,N_prev+1))
            
            for k in range(N+1):
                
                r = N - N_prev
                upper = min(N_prev,k)
                lower = max(0,k-r)
            
                
                for l in range(lower,upper+1):

                    dN_partial[k,l] = -comb(r, k-l)*comb(N_prev,l)/comb(N_prev+r,k)
                    
            dN = np.concatenate((np.zeros(N + 3),np.matmul(dN_partial,beta_prev))) + dN_std
    
        else:
            
            #If this is not the first and farthest maturity, and the current degree is lower than the optimal previous one,
            #a degree elevation procedure for the current maturity needs to be performed.
            
            dN = np.zeros(N + 3 + N_prev + 1)
            dN[-2 - N_prev] = 1 - m_min
            dN[-3 - N_prev] = -(m_max - m_min)/N
            for j in range(N_prev + 1):

                dN[N + 3 + j] = -beta_prev[j]
            
                          
    return dN

#Minimization function: the minimization method chosen is 'trust-constr'.

def minimization(HN,C,BN,fN,num_call,N,AN,dN):
     
    def objFct(x,HN,fN,C,B):
        
        #best_betas = SSE(C,B,x)
        best_betas =  -np.matmul(x, fN) + 0.5 * np.matmul(x, np.matmul(HN, x))
                                                           
        return  best_betas
    
    guess = np.ones(N+1)
    C_filter = C[C > 0]
    constr = [{'type': 'ineq', 'fun': lambda x: np.dot(AN, x) - dN}]

    result = minimize(objFct, guess,args=(HN,fN,C_filter,BN), constraints=constr,  method='trust-constr')
    
    betas = result.x
    
    return betas

#Calculation of the betas for different polynomial degrees and determination of the highest acceptable degree.

def betas_optimal(K, C, num_call, tau, S, r, q, Kmin, Kmax, N_prev, betas_opt,m_min, m_max,first):
   
    start_time = time.process_time()     
    AIC_curve = []
    count = 0
    K_filter = K[K >= 0]
    
    print("Iteration:", first)
    
    if first>0:

        #Set the previous betas as the optimal betas of the previous maturity.
        betas_prev = betas_opt[:,first-1]
        betas_prev = betas_prev[betas_prev!=0]

    else:  
        #If this is the first and farthest maturity, the previous betas are void. 
        betas_prev = np.zeros(int(N_prev))
             
    for i in range(max_iter):
        
        #Actual calculation of the betas for each degree.
        N = i + 2    
        BN = B(K_filter,N,num_call)
        fN = f(BN,C)
        HN = Hessian(BN)
        AN = A(N,int(N_prev),tau,first)
        dN = d(N, tau, S, r, q, Kmin, Kmax, tau, N_prev, betas_prev,m_min, m_max,first)
        
        
        if is_pos_def(HN) == True:
            
            betas_partial = minimization(HN,C,BN,fN,num_call,N,AN,dN)
            AIC_val = AIC(C,BN,betas_partial,num_call,N)
            AIC_curve.append(AIC_val)

        else: 
            count += i 
            break
        
    #N_optimal is the degree that minimized the AIC curve.
            
    N_optimal = AIC_curve.index(min(AIC_curve)) + 2
    
    #Optional graphs to show the AIC curve for each maturity.
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 2 + count), AIC_curve, marker='o')
    plt.title('AIC Curve')
    plt.xlabel('N')
    plt.ylabel('AIC Value')
    plt.grid(True)
    plt.show()
    
    
    #Determination of the optimal N.
    print("Optimal N:", N_optimal)
    if first != number_curves-1:
        N_previous[first+1] = N_optimal
            
    
    #Compution of the optimal betas, optimal degree and total number of constraints.
    BN = B(K,N_optimal,num_call)
    fN = f(BN,C)
    HN = Hessian(BN)
    AN = A(N_optimal,int(N_prev),tau,first)
    dN = d(N_optimal, tau, S, r, q, Kmin, Kmax, tau, N_prev,betas_prev,m_min, m_max,first)
    betas_optimal = minimization(HN,C,BN,fN,num_call,N_optimal,AN,dN)
    N_opt[first] = N_optimal
    
    #Number of constraints (N_constr) for each maturity.
    if first == 0:
        N_constr[first] = N_optimal + 4
    else:
        N_constr[first] = N_optimal + np.maximum(N_optimal,N_prev) + 4
        
    end_time = time.process_time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")
    
    return betas_optimal


### BEST BETAS COMPUTATION

N_previous, N_opt, N_constr, beta_previous = np.zeros(number_curves),np.zeros(number_curves), np.zeros(number_curves), np.zeros(number_curves)
K_norm = normalization(m,mmin,mmax)
final_betas = np.zeros((max_iter, number_curves))

for i in range(number_curves):
   
    final_betas_partial = betas_optimal(K_norm[:,i], C_rel[:,i], date_counts[i], tau[i], S, r[i], q[i], mmin[i], mmax[i], N_previous[i], final_betas,mmin[i], mmax[i],i)  
    final_betas[:,i] = np.concatenate((final_betas_partial, np.zeros(max_iter - len(final_betas_partial))))

# End timer
end_time = time.process_time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time} seconds")



### RESULTS

#Invert the order of the arrays. 
tau = tau[::-1]
final_betas = final_betas[:,::-1]
m = m[:,::-1]
mmax = mmax[::-1]
mmin = mmin[::-1]
r = r[::-1]
q = q[::-1]
F = F[::-1]
K_norm = K_norm[:,::-1]
C_rel = C_rel[:,::-1]
date_counts = date_counts[::-1]
days = days[::-1]
N_opt = N_opt[::-1]
N_constr = N_constr[::-1]
strikes = strikes[:,::-1]
avg_spacing = avg_spacing[::-1]
percent_table = percent_table[:,::-1]

res_x = 100
res_t = 100
x_1D = np.linspace(0, 1, res_x)
t_1D = np.linspace(tau[0], tau[-1], res_t)
t, x = np.meshgrid(t_1D, x_1D)



#FIRST AND SECOND DERIVATIVE

def first_derivative(betas,x_val,N_opt):
    
    first_der = 0
    
    for k in range(N_opt):
        
        first_der += (betas[k+1] - betas[k]) * Bernstein(N_opt-1,k,x_val)

    first_der *= N_opt
    
    return first_der
    
def second_derivative(betas,x_val,N_opt,time):
    
    sec_der = 0
    
    for k in range(N_opt-1):
        
        sec_der += (betas[k+2] - 2*betas[k+1] + betas[k]) * Bernstein(N_opt-2,k,x_val)
    
    sec_der *= N_opt*(N_opt- 1)   
   
    return sec_der
    


### CALCULATION OF THE ESTIMATED RELATIVE CALL PRICES AND DERIVATIVES USING THE BERNSTEIN COEFFICIENTS; INTERPOLATION OF THE RELEVANT QUANTITIES 

def C_final(betas,tau,x,t,rates,dividends,m_min,m_max,F):
    
    rows, cols = len(x), len(t)
    result, first_der, second_der = np.zeros((rows,cols)), np.zeros((rows,cols)), np.zeros((rows,cols))
    
    #Interpolation of the interest, dividend and forward rates.
    r_inter, q_inter, F_inter = np.zeros(cols), np.zeros(cols), np.zeros(cols)
    
    #Interpolation of the maximum and minimum forward strike for each discretized maturities.
    mmin_inter, mmax_inter = np.zeros(cols), np.zeros(cols)
    
    for j_t in range(cols):
        
        #In this snippet, the index of the farthest maturity is identified as upper_index.
        #t_val is the current time.

        t_val = t[j_t]
        upper_index = 0
        
        for i in range(len(tau)-1):
            
            if tau[i] == t_val:
                upper_index = i + 1
                
                break
            
            if tau[i] > t_val:
                upper_index = i
                
                break
            
            if i == len(tau)-2 and upper_index == 0:
                upper_index = i + 1
            
                break
            
            pass
         
        #The optimal polynomial degree of the nearest upper and lower time to maturities. 
        N_upper = np.count_nonzero(betas[:,upper_index]) - 1
        N_lower = np.count_nonzero(betas[:,upper_index - 1]) - 1
        
        #The nearest and farthest maturities from the current value of time t_val are identified.
        tau_up = tau[upper_index]
        tau_down = tau[upper_index - 1]
        
        #Weights computed based on the relative distance between the maturities.
        lower_mat = (tau_up - t_val)/(tau_up - tau_down)
        upper_mat = (t_val - tau_down)/(tau_up - tau_down)
       
        #Interpolation of the interest, dividend, forward rates and maximum and minimum forward strikes.
        r_inter[j_t] = rates[upper_index-1]*lower_mat + rates[upper_index]*upper_mat
        q_inter[j_t] = dividends[upper_index-1]*lower_mat + dividends[upper_index]*upper_mat
        mmin_inter[j_t] = m_min[upper_index-1]*lower_mat + m_min[upper_index]*upper_mat
        mmax_inter[j_t] = m_max[upper_index-1]*lower_mat + m_max[upper_index]*upper_mat
        F_inter[j_t] = F[upper_index-1]*lower_mat + F[upper_index]*upper_mat
        '''
        x_min = (0.9-mmin_inter[j_t])/(mmax_inter[j_t]-mmin_inter[j_t])
        x_max = (1.1-mmin_inter[j_t])/(mmax_inter[j_t]-mmin_inter[j_t])
        x_adj =  np.linspace(x_min, x_max, res)
        '''
        if N_upper >= N_lower:
            
            #If the optimal degree of the highest maturity is greater than the one of the lowest,
            #a degree elevation procedure is performed for the lowest maturity Bernstein coeffiecients.
            #The array new_betas_lower will contain the elevated Bernstein coefficients. 
            #The matrix_coeff contains the coefficients needed to perform the degree elevation.
            
            new_betas_lower = np.zeros(N_upper + 1)
            matrix_coeff = np.zeros((N_upper + 1,N_lower + 1))
            betas_lower = betas[:,upper_index - 1]
            betas_lower = betas_lower[betas_lower != 0]
            betas_upper = betas[:,upper_index]
            betas_upper = betas_upper[betas_upper != 0]
            
            for k in range(N_upper + 1):
                
                r = np.abs(N_upper - N_lower)
                upper = min(N_lower,k)
                lower = max(0,k-r)
            
                    
                for l in range(lower,upper+1):
                
                    matrix_coeff[k,l] = comb(r, k-l)*comb(N_lower,l)/comb(N_lower+r,k)
                    
            #Calculation of the new elevated coefficients. sum_new_bets contains the weighted Bernstein coefficients for the intermidiate maturity.
            new_betas_lower = np.matmul(matrix_coeff,betas_lower)
            sum_new_betas = (new_betas_lower*lower_mat + betas_upper*upper_mat)
            
            
            for i_x in range(rows):
                
                x_val = x[i_x]
                sum_betas = np.zeros_like(sum_new_betas)
                
                #Each intermediate maturity Bernstein coefficient is multiplied with its appropriate degree Bernstein for each normalized strike valye.
                
                for j in range(N_upper + 1):
                    sum_betas[j] = sum_new_betas[j]*Bernstein(N_upper, j, x_val)
                
                #The sum of all the elements of sum_betas corresponds to the estimated value for the the coordinated x_val and t_val.
                #The first and second derivative for the same coordinates are computed as well.
                result[i_x,j_t] = np.sum(sum_betas)
                first_der[i_x,j_t] = first_derivative(sum_new_betas,x_val,N_upper)
                second_der[i_x,j_t] = second_derivative(sum_new_betas,x_val,N_upper,t_val)
            
        else:
            
            #If the optimal degree of the highest maturity is smaller than the one of the lowest,
            #a degree elevation procedure is performed for the highest maturity Bernstein coeffiecients.
            #The rest of the snippet follows the same procedure of the previous case.
            
            new_betas_upper = np.zeros(N_lower + 1)
            matrix_coeff = np.zeros((N_lower + 1,N_upper + 1))
            betas_lower = betas[:,upper_index - 1]
            betas_lower = betas_lower[betas_lower != 0]
            betas_upper = betas[:,upper_index]
            betas_upper = betas_upper[betas_upper != 0]
            
            for k in range(N_lower+1):
    
                r = N_lower-N_upper
                upper = min(N_upper,k)
                lower = max(0,k-r)
                
                for l in range(lower,upper+1):
                    
                    matrix_coeff[k,l] = comb(r, k-l)*comb(N_upper,l)/comb(N_upper+r,k)
                    
            new_betas_upper = np.matmul(matrix_coeff,betas_upper)
            sum_new_betas = (betas_lower*lower_mat + new_betas_upper*upper_mat)
            
            for i_x in range(rows):
                
                x_val = x[i_x]
                sum_betas = np.zeros_like(sum_new_betas)
                
                for i in range(N_lower + 1):
                    sum_betas[i] = sum_new_betas[i]*Bernstein(N_lower, i, x_val)   
                
                
                result[i_x,j_t] = np.sum(sum_betas)
                first_der[i_x,j_t] = first_derivative(sum_new_betas,x_val,N_lower)
                second_der[i_x,j_t] = second_derivative(sum_new_betas,x_val,N_lower,t_val)
    
    
    return result, first_der, second_der, r_inter, q_inter, mmin_inter, mmax_inter, F_inter

#Computation of the relavant quantities.
z, first_der, second_der, r_inter, q_inter, mmin_inter, mmax_inter, F_inter = C_final(final_betas,tau,x_1D,t_1D,r,q,mmin,mmax,F)


                    
#3D AND SCATTER PLOT OF THE ESTIMATED AND TRUE CALL SURFACE VALUES AND DERIVATIVES

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, t, z, cmap='viridis')

#Scatter plot.
for i in range(number_curves):
    
    #Define the coordinates and values of the original data.
    x_orig =  K_norm[:, i]  # m[:, i]  #
    t_orig = np.full_like(x_orig, tau[i])  
    z_orig = C_rel[:, i]  

    #Remove the irrelevant zeros.
    mask = z_orig > 0  
    x_orig = x_orig[mask]
    z_orig = z_orig[mask]
    t_orig = t_orig[mask]
    
    ax.scatter(x_orig, t_orig, z_orig, color='b', s=10, label='True relative call prices')

ax.set_title('Estimated and true call prices')
ax.set_xlabel('Moneyness')
ax.set_ylabel('Time to Maturity')
ax.set_zlabel('Relative call price')
plt.show()

#First derivative plot.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, t, first_der, cmap='viridis')
ax.set_title('First derivative')
ax.set_xlabel('x')
ax.set_ylabel('Time to Maturity')
ax.set_zlabel('First derivative')
plt.show()

#Second derivative plot.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, t, second_der, cmap='viridis')
ax.set_title('Second derivative')
ax.set_xlabel('x')
ax.set_ylabel('Time to Maturity')
ax.set_zlabel('Second derivative')
plt.show()



### IMPLIED VOLATILITY (IV) AND TOTAL IMPLIED VARIANCE (TIV)

#Modified Black&Scholes formula to account for the normalized strikes.
def BSCall(S,T,x_val,F_val,m_max,m_min,r_val,q_val,sigma):
    
    strike_m = (m_max - m_min)*x_val + m_min
    den = strike_m*F_val
    d1 = (np.log(S/den)  + (r_val-q_val+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    price = (S/F_val)*np.exp((r_val-q_val)*T)*norm.cdf(d1) - strike_m*norm.cdf(d2) 
    
    return price

#Optimal IV found with Nelder-Mead minimization procedure.
def IV(S,T,x_val,F_val,m_max,m_min,r_val,q_val,sigma,price):
    
    if T == 0:
        T = 10e-6
        print(T)
    def objFct(sig):
    
        return (price - BSCall(S, T, x_val, F_val, m_max, m_min, r_val, q_val, sig))**2
    
    IVresult = minimize(objFct, sigma, bounds = [(0,None)], method = 'Nelder-Mead')

    return IVresult.x

IV_values = np.zeros((res_x,res_t))

for j in range(res_t): 
    
    if j == 0:
        #Guess for coordinate (0,0) chosen after a first run of simulation.
        guess = 0.23
        
    else:
        #Optimal guess for coordinate (0,j) based on computed IV in (0,j-1).
        guess = IV_values[0,j-1]
        
    for i in range(res_x):
      
      IV_values[i, j] = IV(S, t_1D[j], x_1D[i], F_inter[j], mmax_inter[j], mmin_inter[j], r_inter[j], q_inter[j], guess, z[i, j].item())
      
      #Optimal guess for coordinate (i+1,j) based on computed IV in (i,j).
      guess = IV_values[i,j]
      
TIV = (IV_values**2)*t
IV_true = np.zeros_like(C_rel)



#3D AND SCATTER PLOT OF THE ESTIMATED AND TRUE IV AND TOTAL VARIANCE VALUES.

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, t, IV_values, cmap='viridis')

#Scatter plot of true IV on estimated IV surface.

for i in range(number_curves):
    
    #Define the coordinates and values of the original data.
    x_orig = K_norm[:, i]  
    t_orig = np.full_like(x_orig, tau[i])  
    z_orig = C_rel[:, i]  
        
    #Remove the irrelevant values.
    mask = z_orig > 0  
    x_orig = x_orig[mask]
    t_orig = t_orig[mask]
    z_orig = z_orig[mask]
    
    v_orig = np.zeros_like(z_orig)
    
    for j in range(len(z_orig)):
        if j == 0:
            guess = 0.23
        v_orig[j] = IV(S, t_orig[0], x_orig[j], F[i], mmax[i], mmin[i], r[i], q[i], guess, z_orig[j].item())
        IV_true[j,i] = v_orig[j]
        guess = v_orig[j]
    
    
    ax.scatter(x_orig, t_orig, v_orig, color='b', s=10, label='True implied volatilities.')
 
    
ax.set_title('Implied volatility')
ax.set_xlabel('x')
ax.set_ylabel('Time to Maturity')
ax.set_zlabel('Implied volatility')

# Heatmap IV

plt.figure(figsize=(8, 6))
plt.imshow(IV_values, extent=[x_1D[0], x_1D[-1],t_1D[0], t_1D[-1]], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='C_rel')
plt.title('Implied volatility')
plt.xlabel('x')
plt.ylabel('Time to Maturity')
plt.show()

# 3D graph TIV

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, t, TIV, cmap='viridis')
ax.set_title('Total implied variance')
ax.set_xlabel('x')
ax.set_ylabel('Time to Maturity')
ax.set_zlabel('Total implied variance')



### CONTROL POLYGONS

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(number_curves):
    index = np.nonzero(final_betas[:, i])[0]
    
    values = final_betas[index, i]
    
    non_zero = len(values)
    x_axis = np.linspace(0,1,non_zero)
    y_axis = tau[i]
    z_axis = values
    ax.scatter(x_axis,y_axis,z_axis,marker='o')

ax.set_title('Control Polygons')
ax.set_xlabel('X')
ax.set_ylabel('Time to maturity (years)')
ax.set_zlabel('Bernstein Coefficients')
plt.show()   



### RMSE OF RELATIVE PRICES AND IV

#Index of maturities in time array.

index_mat, RMSE_price, RMSE_iv = np.zeros(number_curves), np.zeros(number_curves), np.zeros(number_curves)
for i in range(len(tau)):
    index_mat = int(np.argmin(np.abs(t_1D - tau[i])))
    length = date_counts[i]
    
    for j in range(length):  
       
       index_strike = np.argmin(np.abs(x_1D - K_norm[j,i]))
       RMSE_price[i] += (z[index_strike,index_mat] - C_rel[j,i])**2
       RMSE_iv[i] += (IV_values[index_strike,index_mat] - IV_true[j,i])**2
    
#RMSE for the call prices.
RMSE_price = np.sqrt(RMSE_price/date_counts)
RMSE_iv = np.sqrt(RMSE_iv/date_counts)



### SPD FOR DIFFERENT MATURITIES 

def spd(sec_der,fwd,mmin,mmax):

    spd_final = np.zeros((res_x,res_t))
                         
    for i in range(res_t):
        denominator = 1/(fwd[i]*(mmax[i]-mmin[i])**2)
        spd_final[:,i] = sec_der[:,i] * denominator
        
    return spd_final

spd_final = spd(second_der,F_inter,mmin_inter,mmax_inter)


#The SPD for each date is computed.
plt.figure(figsize=(10, 6))

for i in range(3):
    
    x_values = np.linspace(strikes[0,i], strikes[date_counts[i]-1,i], res_x)  
    index_mat = int(np.argmin(np.abs(t_1D - tau[i])))
    plt.plot(x_values, spd_final[:,index_mat], linestyle='-', label=f'Time to maturity: {tau[i]:.2f}')
    plt.title('SPD')
    plt.xlabel('Strike')
    plt.ylabel('State price density')
    plt.legend()
    plt.grid(True)
    plt.show()




    


