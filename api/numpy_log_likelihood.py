import numpy as np
from scipy.stats import norm
from functools import partial

dist_neiman_chi_sq = 'neiman_chi_sq'
dist_hellinger = 'hellinger'
dist_kl_divergence = 'kl_divergence'

def calculate_a_b(data):
    a = np.min(data)
    b = np.max(data)
    return a, b

def theta_from_GMM_params(p_arr, mu_arr, sigma_arr):
    theta_1 = np.log(p_arr[0]/(1 - p_arr[0]))
    theta_2 = mu_arr[0]
    theta_3 = mu_arr[1]
    theta_4 = np.log(sigma_arr[0]**2)
    theta_5 = np.log(sigma_arr[1]**2)
    return np.array([theta_1, theta_2, theta_3, theta_4, theta_5])

def GMM_params_from_theta(theta_arr):
    p = 1/(1 + np.exp(-theta_arr[0]))
    p_arr = np.array([p, 1 - p])
    mu_arr = theta_arr[1:3]
    sigma_arr = np.array([np.exp(theta_arr[3]/2), np.exp(theta_arr[4]/2)])
    return p_arr, mu_arr, sigma_arr
    
def my_exp(a, b, c, d):
    return np.exp(a + b + np.sqrt(2)*c + 2*d)

def calculate_stupid_phi(a, b, eta_arr):
    """
    Calculation of phi(eta) for Haar's basis with J = 2

    Parameters
    ----------
    a, b: float
        all data in segment [a, b]
    eta_arr : np.array
        eta_arr = [eta_{0, 0}, eta_{1, 0}, eta_{1, 1}, eta_{2, 0},
                    eta_{2, 1}, eta_{2. 2}, eta_{2, 3}, eta_{0}]
    """
    eta_00 = eta_arr[0]
    eta_10 = eta_arr[1]
    eta_11 = eta_arr[2]
    eta_20 = eta_arr[3]
    eta_21 = eta_arr[4]
    eta_22 = eta_arr[5]
    eta_23 = eta_arr[6]
    eta_0 = eta_arr[7]
    
    exp_sum = (my_exp(eta_0, eta_00, eta_10, eta_20) + my_exp(eta_0, eta_00, eta_10, -eta_20) + 
                my_exp(eta_0, eta_00, -eta_10, eta_21) + my_exp(eta_0, eta_00, -eta_10, -eta_21) + 
                my_exp(eta_0, -eta_00, eta_11, eta_22) + my_exp(eta_0, -eta_00, eta_11, -eta_22) + 
                my_exp(eta_0, -eta_00, -eta_11, eta_23) + my_exp(eta_0, -eta_00, -eta_11, -eta_23))
                
    return np.log((b - a)/8) + np.log(exp_sum) 

def p(y):
    #print(y)
    return int((y >= 0) and (y < 1))

def h(y):
    return int((y >= 0) and (y < 1/2)) - int((y >= 1/2) and (y < 1))

def h_jk(y, j, k):
    return (2**(j/2))*h((2**j)*y - k)

def p_jk(y, j, k):
    return (2**(j/2))*p((2**j)*y - k)

def index_to_jk(n):
    j = int(np.log2(n + 1))
    k = n + 1 - 2**j
    return j, k

def jk_to_index(j, k):
    return 2**j + k - 1

def convert_1d_eta_to_2d_eta(eta_arr):
    eta_0 = eta_arr[-1]
    size = eta_arr.shape[0]
    J = int(np.log2(size)) - 1
    eta_2d = np.array([np.array([float(0) for k in range(2**j)]) for j in range(J + 1)])
    for i in range(size - 1):
        j, k = index_to_jk(i)
        eta_2d[j][k] = eta_arr[i]
        
    return eta_2d, eta_0

def convert_2d_eta_to_1d_eta(eta_2d, eta_0):
    J = eta_2d.shape[0] - 1
    eta_arr = np.zeros(2**(J + 1))
    for j in range(J + 1):
        for k in range(2**j):
            i = jk_to_index(j, k)
            eta_arr[i] = eta_2d[j][k]
            
    eta_arr[-1] = eta_0
    return eta_arr

def f(J, j, k, m):
    if ((k*(2**(J + 1)) <= m*(2**(j))) and ((m + 1)*(2**(j + 1)) <= (2*k + 1)*(2**(J + 1)))):
        return 1
    elif (((2*k + 1)*(2**(J + 1)) <= m*(2**(j + 1))) and ((m + 1)*(2**(j)) <= (k + 1)*(2**(J + 1)))):
        return -1
    else:
        return 0

def calculate_phi(a, b, eta_arr):
    """
    Calculation of phi(eta) for Haar's basis with any J

    Parameters
    ----------
    a, b: float
        all data in segment [a, b]
    eta_arr : np.array
        eta_arr = [eta_{0, 0}, eta_{1, 0}, eta_{1, 1}, eta_{2, 0},
                    eta_{2, 1}, eta_{2. 2}, eta_{2, 3}, eta_{0}]
    """
    eta_2d, eta_0 = convert_1d_eta_to_2d_eta(eta_arr)
    size = eta_arr.shape[0]
    J = int(np.log2(size)) - 1
    res_sum = 0 
    for m in range(2**(J + 1)):
        cur_sum = eta_0
        for j in range(J + 1):
            for k in range(2**j):
                #print(f"Numpy, m = {m}, j = {j}, k = {k}, cur = {eta_2d[j][k]*(2**(j/2))*f(J, j, k, m)}")
                cur_sum += eta_2d[j][k]*(2**(j/2))*f(J, j, k, m)
        #print(f"Numpy, phi, exp of part = {cur_sum}")
        res_sum += np.exp(cur_sum)
    #print(f"Numpy, const = {np.log((b - a)/(2**(J + 1)))}")
    return np.log(res_sum) + np.log((b - a)/(2**(J + 1)))
    
def calculate_g_without_phi(y, eta_arr, a, b):
    eta_2d, eta_0 = convert_1d_eta_to_2d_eta(eta_arr)
    point = (y - a)/(b - a)
    #print(point)
    res_sum = eta_0*p(point)
    size = eta_arr.shape[0]
    J = int(np.log2(size)) - 1
    for j in range(J + 1):
        for k in range(2**j):
            res_sum += eta_2d[j][k]*h_jk(point, j, k)
    
    return res_sum

def L(eta_arr, data):
    a, b = calculate_a_b(data)
    first_part = np.sum(np.array([calculate_g_without_phi(y, eta_arr, a, b) for y in data])) 
    second_part = data.shape[0]*calculate_phi(a, b, eta_arr)
    return first_part - second_part

def dens_estimation(y, eta_arr, phi, a, b):
    eta_2d, eta_0 = convert_1d_eta_to_2d_eta(eta_arr)
    point = (y - a)/(b - a)
    res_sum = eta_0*p(point)
    size = eta_arr.shape[0]
    J = int(np.log2(size)) - 1
    for j in range(J + 1):
        for k in range(2**j):
            res_sum += eta_2d[j][k]*h_jk(point, j, k)
    res_sum -= phi
    
    return np.exp(res_sum)
    
def sigma_arr_with_noise(sigma_arr, sigma_noise):
    sigma_arr_noise = np.array([np.sqrt((sigma_arr[0]**2 + sigma_noise**2)), 
                                np.sqrt(sigma_arr[1]**2 + sigma_noise**2)])
    return sigma_arr_noise

def sigma_arr_with_noise_sqr(sigma_arr, sigma_noise):
    sigma_arr_noise_sqr = np.array([(sigma_arr[0]**2 + sigma_noise**2), 
                                (sigma_arr[1]**2 + sigma_noise**2)])
    return sigma_arr_noise_sqr

def calculate_gauss_int(c, d, mu, sigma):
    return norm.cdf(d, loc=mu, scale=sigma) - norm.cdf(c, loc=mu, scale=sigma)

def calculate_int_of_conv(c, d, theta_arr, sigma_noise):
    p_arr, mu_arr, sigma_arr = GMM_params_from_theta(theta_arr)
    sigma_arr_noise = sigma_arr_with_noise(sigma_arr, sigma_noise)
    first_part = p_arr[0]*calculate_gauss_int(c, d, mu_arr[0], sigma_arr_noise[0])
    second_part = p_arr[1]*calculate_gauss_int(c, d, mu_arr[1], sigma_arr_noise[1])
    return first_part + second_part

def calculate_int_of_sqr_conv(c, d, theta_arr, sigma_noise):
    p_arr, mu_arr, sigma_arr = GMM_params_from_theta(theta_arr)
    sigma_arr_noise = sigma_arr_with_noise(sigma_arr, sigma_noise)
    sigma_arr_noise_sqr = sigma_arr_with_noise_sqr(sigma_arr, sigma_noise)
    sigma_sum_noise_sqr = np.sum(sigma_arr_noise_sqr)
    sigma_sum_noise = np.sqrt(sigma_sum_noise_sqr)
    sigma_harmon_noise = np.sqrt(sigma_arr_noise_sqr[0]*sigma_arr_noise_sqr[1]/sigma_sum_noise_sqr)
    mu_convex_noise = (mu_arr[0]*sigma_arr_noise_sqr[1] + 
                       mu_arr[1]*sigma_arr_noise_sqr[0])/(sigma_sum_noise_sqr)
    first_part = (p_arr[0]**2)*(1/(2*np.sqrt(np.pi)*sigma_arr_noise[0]))*calculate_gauss_int(c, 
                                                            d, mu_arr[0], sigma_arr_noise[0]/(np.sqrt(2)))
    second_part = 2*p_arr[0]*p_arr[1]*np.exp(-((mu_arr[0] - 
        mu_arr[1])**2)/(2*sigma_sum_noise_sqr))/(np.sqrt(2*np.pi) * sigma_sum_noise)*calculate_gauss_int(c, 
                                                            d, mu_convex_noise, sigma_harmon_noise)
    third_part = (p_arr[1]**2)*(1/(2*np.sqrt(np.pi)*sigma_arr_noise[1]))*calculate_gauss_int(c, 
                                                            d, mu_arr[1], sigma_arr_noise[1]/(np.sqrt(2)))
    return first_part + second_part + third_part

def calculate_NCS(eta_arr, theta_arr, a, b, sigma_noise):
    phi = calculate_phi(a, b, eta_arr)
    #print(f"Numpy, phi = {phi}")
    second_part = 2*calculate_int_of_conv(a, b, theta_arr, sigma_noise)
    first_part = 0
    eta_2d, eta_0 = convert_1d_eta_to_2d_eta(eta_arr)
    #print(f"Numpy, eta_2d = {eta_2d}, eta_0 = {eta_0}")
    #print(f"Numpy, theta = {theta_arr}")
    size = eta_arr.shape[0]
    J = int(np.log2(size)) - 1
    for m in range(2**(J + 1)):
        denominator = eta_0 - phi
        for j in range(J + 1):
            for k in range(2**j):
                denominator += eta_2d[j][k]*(2**(j/2))*f(J, j, k, m)
        #print(f"Numpy, exp of denominator = {denominator}")
        denominator = np.exp(denominator)
        point_int_from = a + m*(b - a)/(2**(J + 1))
        point_int_to = a + (m + 1)*(b - a)/(2**(J + 1))
        second_prod_term = calculate_int_of_sqr_conv(point_int_from, point_int_to, theta_arr, sigma_noise)
        #print(f"Numpy, denominator = {denominator}, second_prod_term = {second_prod_term}")
        first_part += (second_prod_term/denominator)
    #print(f"Numpy distance, first_part = {first_part}, second_part = {second_part}")
    return first_part - second_part + 1 
    
def calculate_prior(prior_arr, prior_0, prior_cov_inv):
    diff = prior_arr - prior_0
    return -0.5*np.dot(diff, np.dot(prior_cov_inv, diff))
    
def calculate_log_likelihood(params, data, J, lambda_coef, dist_type,
                             sigma_noise, eta_0, eta_cov_inv, 
                             theta_0, theta_cov_inv):
    params = params['points']
    eta_arr = params[:2**(J + 1)]
    theta_arr = params[2**(J + 1):]
    a, b = calculate_a_b(data)
    L_data = L(eta_arr, data)
    #print(f"L_data = {L_data}")
    if (dist_type == dist_neiman_chi_sq):
        measure_dist = calculate_NCS(eta_arr, theta_arr, a, b, sigma_noise)
    #print(f"measure_dist = {measure_dist}")
    eta_prior_term = calculate_prior(eta_arr, eta_0, eta_cov_inv)
    theta_prior_term = calculate_prior(theta_arr, theta_0, theta_cov_inv)
    #print(f"eta_prior_term = {eta_prior_term}")
    #print(f"theta_prior_term = {theta_prior_term}")
    return L_data + lambda_coef*measure_dist + eta_prior_term + theta_prior_term
    
def posterior_log_likelihood(calculate_log_likelihood, data, J, 
			     lambda_coef, dist_type,
                             sigma_noise, eta_0, eta_cov_inv, 
                             theta_0, theta_cov_inv):
    return partial(calculate_log_likelihood, 
                   data = data, J = J, lambda_coef = lambda_coef,
                   dist_type = dist_type, sigma_noise = sigma_noise,
                   eta_0 = eta_0, eta_cov_inv = eta_cov_inv, 
                   theta_0 = theta_0, theta_cov_inv = theta_cov_inv)
