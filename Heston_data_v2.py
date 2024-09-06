import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.special import comb
from scipy.optimize import minimize
from scipy.integrate import quad
import warnings

### DEFINITION OF THE MAIN CONSTANTS

warnings.filterwarnings("ignore")
method_opt = 'SLSQP'

#Market parameters.
S0 = 1
r = 0.0
q = 0.0

#Heston model parameters.
v0 = 0.1
kappa = 1.15
theta = 0.0347
vvol = 0.39
rho = -0.64

#Perturbation gaussian standard deviation.
sigma_per = 0.005

#Number of perturbed surfaces and highest possible degree of the Bernstein polynomial.

rep = 1000
max_iter = 30

#Definition of the maturities and strikes.

dates = [12/12,9/12,6/12,3/12,2/12,1/12]
number_curves = len(dates)
T = max(dates)
number_strikes = 41
mmin = 0.6
mmax = 1.4
fwd = 1
K = np.linspace(mmin, mmax, number_strikes)

#Array containing the number of constraints.

N_constr, N_constr_nocal = np.zeros(number_curves), np.zeros(number_curves)



### COMPUTATION OF THE TRUE PRICES ACCORDING TO THE HESTON MODEL

def C(w,t,r_minus,g,h,vvol):
    
    return kappa*(r_minus*t-2*np.log((1-g*np.exp(-h*t))/(1-g))/(vvol**2))

def D(w,t,r_minus,g,h):
    
    return r_minus*(1-np.exp(-h*t))/(1-g*np.exp(-h*t))
          
def PI(w,t,vvol,rho,theta,S0,r,K,first,price):
    
    x = np.log(S0 /K)
    
    if first == 1:
        alpha = -0.5 * w**2 - 0.5j * w + 1j * w
        beta = kappa - rho * vvol - 1j * rho * vvol * w
    else:
        alpha = -0.5 * w**2 - 0.5j * w
        beta = kappa - 1j * rho * vvol * w

    gamma = 0.5*vvol**2
    h = np.sqrt(beta**2 - 4*alpha*gamma)
    r_plus = (beta + h)/vvol**2
    r_minus = (beta - h)/vvol**2
    g = r_minus/r_plus
    
    if price == "True": 
        result = np.real(np.exp(C(w,t,r_minus,g,h,vvol) * theta + D(w,t,r_minus,g,h) * v0 + 1j * w * x) / (1j * w))
    else:
        result = np.exp(C(w,t,r_minus,g,h,vvol) * theta + D(w,t,r_minus,g,h)* v0 + 1j * w * x)
        
    return result

def HestonPrice(t,vvol,rho,theta,S0,r,K):
    P_I1_res, _ = quad(PI,0,np.inf,args=(t,vvol,rho,theta,S0,r,K,1,"True"))
    P_I2_res, _ = quad(PI,0,np.inf,args=(t,vvol,rho,theta,S0,r,K,2,"True"))
    P_I1 = 0.5 + P_I1_res/np.pi
    P_I2 = 0.5 + P_I2_res/np.pi
    price = S0*P_I1 - np.exp(-r*t)*K*P_I2
    return price

def HestonPdf(S,t,vvol,rho,theta,S0,r,K):
    P_I2_res, _ = quad(PI,-np.inf,np.inf,args=(t,vvol,rho,theta,S0,r,K,2,"False"))
    pdf = 0.5* P_I2_res/np.pi
    return pdf
    
strikes = np.linspace(0.6,1.4,number_strikes)
selected_prices = np.zeros((len(strikes),len(dates)))
selected_pdf = np.zeros_like(selected_prices)

for j in range(len(dates)):
    for i in range(len(strikes)):
        selected_prices[i,j] = HestonPrice(dates[j],vvol,rho,theta,S0,r,strikes[i])
        selected_pdf[i,j] = HestonPdf(strikes[i],dates[j],vvol,rho,theta,S0,r,strikes[i])

#Gaussian perturbation of prices. If the unperturbed value is close to 0, it is substituted by a random value taken from a uniform distribution (minimal perturbation).
perturbation = np.random.normal(0,sigma_per,(len(strikes),len(dates),rep))
minimal_perturbation = np.random.uniform(low=1e-8, high=1e-6, size=(len(strikes),len(dates),rep))
perturbed_prices = np.zeros_like(perturbation)

for i in range(rep):
    perturbed_prices[:,:,i] = np.maximum(selected_prices + perturbation[:,:,i], minimal_perturbation[:,:,i]) 

print("Perturbed prices computed.")



### DEFINITION OF THE COMMON FUNCTIONS TO COMPUTE THE BEST BETAS

def normalization(K):
    max_K, min_K = np.max(K), np.min(K)
    K_norm = (K - min_K)/(max_K - min_K)
    return K_norm

def Bernstein(N, deg, val):
    value = comb(N, deg, exact=False) * (val ** deg) * ((1 - val) ** (N - deg))
    return value

def SSE(C, B, betas):
    Bbetas = np.matmul(B, betas)
    errors = Bbetas - C
    SSE = np.sum(errors ** 2)
    return SSE

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def B(K, N, num_call):
    B = np.zeros((num_call, N+1))
    for i in range(N+1):
        for j in range(num_call):
            B[j, i] = Bernstein(N, i, K[j])
    return B

def f(B, C):
    f = np.matmul(np.transpose(B), C)
    return f
    
def Hessian(B):
    H = np.matmul(np.transpose(B), B)
    return H

def minimization(HN,C,BN,fN,num_call,N,AN,dN):
     
    def objFct(x,HN,fN,C,B):
        
        best_betas =  -np.matmul(x, fN) + 0.5 * np.matmul(x, np.matmul(HN, x))  #
                                                                    
        return  best_betas
    
    guess = np.ones(N+1)
    C_filter = C[C > 0]
    constr = [{'type': 'ineq', 'fun': lambda x: np.dot(AN, x) - dN}]
    
    result = minimize(objFct, guess,args=(HN,fN,C_filter,BN), constraints=constr, method = method_opt)
    
    betas = result.x
    
    return betas

### BEST BETAS WITH STRIKE BUT NO CALENDAR CONSTRAINTS

K_norm = normalization(strikes)
final_betas_nocal = np.zeros((max_iter, rep, number_curves))

#Definition first and second derivative.

def first_derivative(betas,x_val,N_opt):
    
    first_val = 0
    
    for k in range(N_opt):
        
        first_val += (betas[k+1] - betas[k]) * Bernstein(N_opt-1,k,x_val)

    first_val *= N_opt
    
    return first_val
    
def second_derivative(betas,x_val,N_opt):
    
    sec_val = 0
    
    for k in range(N_opt-1):
        
        sec_val += (betas[k+2] - 2*betas[k+1] + betas[k]) * Bernstein(N_opt-2,k,x_val)

    sec_val *= N_opt*(N_opt- 1)   
            
    return sec_val

def spd(sec_der,fwd,mmin,mmax):

    spd_final = np.zeros((number_strikes,number_curves))
                         
    for i in range(number_curves):
        denominator = 1/(fwd*(mmax-mmin)**2)
        spd_final[:,i] = sec_der[:,i] * denominator
        
    return spd_final

#Definition of A,d and optimal betas is similar to the functions in the other file. 
#In this case, it is not necessary to account for the previous optimal betas as each curve is estimated separately with no calendar constraint.

def A(N):
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
    return AN
    
def d(N, tau, S, r, q, Kmin, Kmax):
    dN = np.zeros(N + 4)
    dN[-1] = -S * np.exp(-q * tau)
    dN[-2] = S * np.exp(-q * tau) - Kmin * np.exp(-r * tau)
    dN[-3] = -np.exp(-r * tau) * (Kmax - Kmin) / N
    return dN
    
def betas_optimal_nocal(K, C, c_actual, num_call, tau, S, r, q, Kmin, Kmax):
    
    #MISE curve saves the value of the MISE for each degree. The argmin of this curve is the optimal polynomial degree.
    MISE_curve = []
    count = 0

    for i in range(max_iter):
        
        N = i + 5       
        BN = B(K,N,num_call)
        HN = Hessian(BN)
        AN = A(N)
        dN = d(N, tau, S, r, q, Kmin, Kmax)
        
        if is_pos_def(HN) == True:
            
            c_estimated = np.zeros((num_call,rep))
            
            for j in range(rep):
                
                fN = f(BN,C[:,j])
                betas = minimization(HN,C[:,j],BN,fN,num_call,N,AN,dN)
                c_estimated[:,j] = np.matmul(BN,betas)
            
            #Calculation of the squared bias and variance. The MISE is computed by numerically integrating the biases and variances 
            #along the normalized strike range. 
            
            c_mean = np.mean(c_estimated, axis=1)
            c_bias = (c_mean - c_actual)**2
            c_variance = np.var(c_estimated, axis=1)
            
            sum_var = np.sum(c_variance)/(number_strikes-1)
            sum_bias = np.sum(c_bias)/(number_strikes-1)
            mise = sum_var + sum_bias
            
            MISE_curve.append(mise)
                                                                
        else:
            count += i 
            break
    
    # Finally, the optimal degree N that minimizes the MISE is found. 
    index_optimal = MISE_curve.index(min(MISE_curve))
    N_optimal = index_optimal + 5
    print("No calendar constraints.")
    print("Optimal N:", N_optimal)
    print("MISE optimal:", min(MISE_curve))
    print("Max N:", 5 + count)
    
    betas_optimal = np.zeros((N_optimal+1,rep))
    
    
    BN = B(K,N_optimal,num_call)
    HN = Hessian(BN)
    AN = A(N_optimal)
    dN = d(N_optimal, tau, S, r, q, Kmin, Kmax)

    c_estimated = np.zeros((num_call,rep))

    
    #Computation of the optimal coefficients for each simulations.
    for j in range(rep):
        
        fN = f(BN,C[:,j])
        betas_optimal[:,j] = minimization(HN,C[:,j],BN,fN,num_call,N_optimal,AN,dN) 
        c_estimated[:,j] = np.matmul(BN,betas_optimal[:,j])
          
    #Optimal bias and variance.
    c_mean_optimal = np.mean(c_estimated, axis=1)
    c_bias_optimal = (c_mean_optimal - c_actual)**2
    c_variance_optimal = np.var(c_estimated, axis=1)
    
    return betas_optimal, c_bias_optimal, c_variance_optimal, min(MISE_curve)


### BEST BETAS WITH STRIKE BUT NO CALENDER CONSTRAINTS COMPUTATION

#Bias and Variance with no calendar constraints
bias_nocal, var_nocal, MISE_optimal_nocal = np.zeros((number_strikes,number_curves)), np.zeros((number_strikes,number_curves)), np.zeros(number_curves)

for i in range(number_curves):
    
    final_betas_partial_nocal, bias_nocal[:,i], var_nocal[:,i], MISE_optimal_nocal[i] = betas_optimal_nocal(K_norm, perturbed_prices[:,i,:], selected_prices[:,i], number_strikes, dates[i], 1, 0, 0, 0.6, 1.4) 
    zeros_matrix = np.zeros((max_iter - len(final_betas_partial_nocal), rep))
    combined_matrix = np.concatenate([final_betas_partial_nocal, zeros_matrix], axis=0)
    final_betas_nocal[:,:,i] = combined_matrix    

#Total MSE without calendar constraints.
MSE_nocal = bias_nocal + var_nocal
ibias_nocal = simpson(bias_nocal, dx=0.02,axis=0)
ivar_nocal = simpson(var_nocal, dx=0.02,axis=0)
MISE_nocal = simpson(MSE_nocal, dx=0.02,axis=0)



### BEST BETAS WITH BOTH STRIKE AND CALENDER CONSTRAINTS

N_previous, beta_previous = np.zeros(number_curves), np.zeros(number_curves)
final_betas = np.zeros((max_iter, rep, number_curves))

#The following functions are defined and share the same logic of the ones in the other file.
def A_all(N,N_prev,t, first):
    
    if first == 0:
  
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

def d_all(N, tau, S, r, q, Kmin, Kmax, t, N_prev, beta_prev, m_min, m_max, first):
    
    N_prev = int(N_prev)

    if first == 0:
        dN = np.zeros(N + 4)
        dN[-1] = -1
        dN[-2] = 1 - m_min
        dN[-3] = -(m_max - m_min)/N
        
    else:
        
        if N > N_prev:
           
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
            
            dN = np.zeros(N + 3 + N_prev + 1)
            dN[-2 - N_prev] = 1 - m_min
            dN[-3 - N_prev] = -(m_max - m_min)/N
            for j in range(N_prev + 1):

                dN[N + 3 + j] = -beta_prev[j]
            
                          
    return dN


def betas_optimal(K, C, c_actual,spd_actual,num_call, tau, S, r, q, Kmin, Kmax, N_prev, betas_opt,m_min, m_max,first):
    
    #MISE saves the value of the MISE for each degree. The argmin of this curve is the optimal polynomial degree.
    MISE_curve = []
    count = 0
    K_filter = K[K >= 0]
    
    print("Iteration:", first)
    
    if first>0:

        betas_prev = betas_opt[:,:,first-1]
        betas_prev = betas_prev[:-(max_iter-int(N_prev)-1),:]

    else:  
    
        betas_prev = np.zeros((int(N_prev),rep))
             
    for i in range(max_iter):
        
        #In this case, only polynomials of at least degree five are considered.
        N = i + 5   
        BN = B(K_filter,N,num_call)
        HN = Hessian(BN)
        AN = A_all(N,int(N_prev),tau,first)
        
        if is_pos_def(HN) == True:
            
            c_estimated = np.zeros((num_call,rep))
            
            for j in range(rep):  

                dN = d_all(N, tau, S, r, q, Kmin, Kmax, tau, N_prev, betas_prev[:,j], m_min, m_max,first) 
                fN = f(BN,C[:,j])
                
                betas_partial = minimization(HN,C[:,j],BN,fN,num_call,N,AN,dN)
                c_estimated[:,j] = np.matmul(BN,betas_partial)
             
            #Calculation of the squared bias and variance. The MISE is computed by numerically integrating the biases and variances 
            #along the normalized strike range. 
            
            c_mean = np.mean(c_estimated, axis=1)
            c_bias = (c_mean - c_actual)**2
            c_variance = np.var(c_estimated, axis=1)
            mise = simpson(c_bias + c_variance, dx=0.02,axis=0)
            MISE_curve.append(mise)

        else: 
            count += i 
            break
    
    # Finally, the optimal degree N that minimizes the MISE is found. 
    N_optimal = MISE_curve.index(min(MISE_curve)) + 5
    print("All constraints.")
    print("Optimal N:", N_optimal)
    print("MISE optimal:", min(MISE_curve))
    print("Max N:", i + 5)
    
    if first != number_curves-1:    
        N_previous[first+1] = N_optimal

    betas_optimal = np.zeros((N_optimal+1,rep))
    c_estimated = np.zeros((num_call,rep))
    spd_estimated = np.zeros((num_call,rep))
    BN = B(K,N_optimal,num_call)
    HN = Hessian(BN)
    AN = A_all(N_optimal,int(N_prev),tau,first)
    
    #Computation of the optimal coefficients for each simulations.
    for j in range(rep):

        dN = d_all(N_optimal, tau, S, r, q, Kmin, Kmax, tau, N_prev, betas_prev[:,j],m_min, m_max,first)
        fN = f(BN,C[:,j])
        betas_optimal[:,j] = minimization(HN,C[:,j],BN,fN,num_call,N_optimal,AN,dN)
        c_estimated[:,j] = np.matmul(BN,betas_optimal[:,j])
        second_der_estimated = np.zeros(number_strikes)
        denominator = 1/(fwd*(mmax-mmin)**2)
        second_der_estimated = second_derivative(betas_optimal[:,j], K, N_optimal)
        spd_estimated[:,j] = second_der_estimated*denominator    
        
    #Optimal bias and variance.
    c_mean = np.mean(c_estimated, axis=1)
    c_bias = (c_mean - c_actual)**2
    c_variance = np.var(c_estimated, axis=1)
        
    spd_mean = np.mean(spd_estimated, axis=1)
    spd_bias = (spd_mean - spd_actual)**2
    spd_variance = np.var(spd_estimated, axis=1)
    
    #Number of constraints (N_constr) for each maturity.
    if first == 0:
        N_constr[first] = N_optimal + 4
    else:
        N_constr[first] = N_optimal + np.maximum(N_optimal,N_prev) + 4
    
    return betas_optimal, c_bias, c_variance, spd_bias, spd_variance, min(MISE_curve)



### BEST BETAS WITH BOTH STRIKE AND CALENDAR CONSTRAINTS COMPUTATION

#Bias and Variance.
bias, var, MISE_optimal = np.zeros((number_strikes,number_curves)), np.zeros((number_strikes,number_curves)), np.zeros(number_curves)
bias_spd, var_spd = np.zeros((number_strikes,number_curves)), np.zeros((number_strikes,number_curves))

for i in range(number_curves):
    
    final_betas_partial, bias[:,i], var[:,i], bias_spd[:,i], var_spd[:,i], MISE_optimal[i] = betas_optimal(K_norm, perturbed_prices[:,i,:], selected_prices[:,i], selected_pdf[:,i], number_strikes, dates[i], 1, 0, 0, 0.6, 1.4, N_previous[i], final_betas,0.6, 1.4,i)  
    zeros_matrix = np.zeros((max_iter - len(final_betas_partial), rep))
    combined_matrix = np.concatenate([final_betas_partial, zeros_matrix], axis=0)
   
    final_betas[:,:,i] = combined_matrix    

#Total MSE with all constraints imposed.
MSE = bias + var
MSE_spd = bias_spd + var_spd
ibias = simpson(bias, dx=0.02,axis=0)
ivar =simpson(var, dx=0.02,axis=0)
MISE = simpson(MSE, dx=0.02,axis=0)
MISE_spd = simpson(MSE_spd, dx=0.02,axis=0)



#ESTIMATED CALL PRICES AND DERIVATIVES

x = np.linspace(0, 1, number_strikes)
first_der, second_der = np.zeros((number_strikes,number_curves)), np.zeros((number_strikes,number_curves))
    
def C_rel_estimated(betas,x,t):

    result, first_der, second_der = np.zeros((len(x),len(t))), np.zeros((len(x),len(t))), np.zeros((len(x),len(t)))
    rows, cols = len(x), len(t)
    
    for j_t in range(cols):
       
        betas_partial = betas[:,:,j_t]
        
        N = 0
        for i in range(max_iter):
            if betas_partial[i,0] > 0:
                N +=1
                
        betas_partial = betas_partial[:N,:]
       
        N -= 1
        bern_val = np.zeros((rows,N+1))
        
        for i_x in range(rows):      
            for j in range(N + 1):
                bern_val[i_x,j] = Bernstein(N, j, x[i_x])
            
            first_der[i_x,j_t] = first_derivative(betas_partial[:,j_t],x[i_x],N)
            second_der[i_x,j_t] = second_derivative(betas_partial[:,j_t],x[i_x],N)

        matrix_mult = np.matmul(bern_val,betas_partial)
        result[:,j_t] = np.mean(matrix_mult,axis=1)

                
    return result, first_der, second_der

#The estimated call curve, its first and second derivatives are computed.
c_rel_est, first_der, second_der = C_rel_estimated(final_betas,x,dates)

#Invert the arrays.
dates = dates[::-1]
c_rel_est = c_rel_est[:,::-1]
selected_prices = selected_prices[:,::-1]
first_der = first_der[:,::-1]
second_der = second_der[:,::-1]
selected_pdf =  selected_pdf[:,::-1]



### SPD FOR DIFFERENT MATURITIES

spd_final = spd(second_der,fwd,mmin,mmax)

#This is not normalized and it should be checked again to ensure that it is correct.
plt.figure()

for i in range(number_curves):
    
    plt.plot(K, spd_final[:,i], linestyle='-', label=f'Time to maturity: {dates[i]:.2f}')
    plt.title('SPD')
    plt.xlabel('Strike')
    plt.ylabel('Estimated state price densities')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
#The relative variance, squared bias and MSE are computed. The relative MSE tends to be greater than 1 for most strikes.
var, var_nocal = var[:,::-1], var_nocal[:,::-1]
ivar, ivar_nocal = ivar[::-1], ivar_nocal[::-1]
bias, bias_nocal = bias[:,::-1], bias_nocal[:,::-1]
ibias, ibias_nocal = ibias[::-1], ibias_nocal[::-1]
MSE_nocal, MSE = MSE_nocal[:,::-1], MSE[:,::-1]
MISE_nocal, MISE = MISE_nocal[::-1], MISE[::-1]
MSE_spd, MISE_spd = MSE_spd[::-1], MISE_spd[::-1]

rel_var = var_nocal/var
rel_bias = bias_nocal/bias
rel_mse = MSE_nocal/MSE



###PLOTS 

#Graphs of the true and estimated call price curve.
#Date index between 0, for the earliest date, and 5, for the furthest date.
index_date = 2

plt.figure()
plt.plot(K_norm, c_rel_est[:,index_date],'--', color='blue', label='Call price estimate')
plt.plot(K_norm, selected_prices[:,index_date],'-',color='red', label='Call price true')
plt.title('Mean Estimate')
plt.xlabel('x')
plt.ylabel('Call price')
plt.legend()
plt.grid(True)
plt.show()

#Graph of the MSE with calendar constraints.
plt.figure()
plt.plot(x, MSE[:,index_date],'-o', label='MSE')
plt.title('MSE')
plt.xlabel('x')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

#Graph of the squared bias with calendar constraints.
plt.figure()
plt.plot(x, bias[:,index_date],'-o', label='bias')
plt.title('Bias')
plt.xlabel('x')
plt.ylabel('Bias')
plt.legend()
plt.grid(True)
plt.show()

#Graph of the variance with calendar constraints.
plt.figure()
plt.plot(x, var[:,index_date],'-o', label='var')
plt.title('Var')
plt.xlabel('x')
plt.ylabel('Var')
plt.legend()
plt.grid(True)
plt.show()


#Graphs of the true and estimated SPD curve.
#Date index between 0, for the earliest date, and 5, for the furthest date.

plt.figure()
plt.plot(strikes, spd_final[:,index_date],'--', color='blue', label='SPD estimate')
plt.plot(strikes, selected_pdf[:,index_date],'-',color='red', label='SPD true')
plt.title('True and estimated SPD')
plt.xlabel('x')
plt.ylabel('SPD')
plt.ylim(0, 5)
plt.legend()
plt.grid(True)
plt.show()

#Graph of the MSE with calendar constraints.
plt.figure()
plt.plot(x, MSE_spd[:,index_date],'-o', label='MSE SPD')
plt.title('MSE SPD')
plt.xlabel('x')
plt.ylabel('MSE SPD')
plt.legend()
plt.grid(True)
plt.show()

#Graph of the squared bias with calendar constraints.
plt.figure()
plt.plot(x, bias_spd[:,index_date],'-o', label='bias SPD')
plt.title('Bias SPD')
plt.xlabel('x')
plt.ylabel('bias SPD')
plt.legend()
plt.grid(True)
plt.show()

#Graph of the variance with calendar constraints.
plt.figure()
plt.plot(x, var_spd[:,index_date],'-o', label='var SPD')
plt.title('Var SPD')
plt.xlabel('x')
plt.ylabel('Var SPD')
plt.legend()
plt.grid(True)
plt.show()

