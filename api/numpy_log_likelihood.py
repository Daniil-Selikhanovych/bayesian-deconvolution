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
    p_arr = np.array([p, 1 - p], dtype = np.float64)
    mu_arr = theta_arr[1:3]
    sigma_arr = np.array([np.exp(theta_arr[3]/2), np.exp(theta_arr[4]/2)], dtype = np.float64)
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
    return ((y >= 0) & (y < 1)).astype(int)

def h(y):
    return ((y >= 0) & (y < 1/2)).astype(int) - ((y >= 1/2) & (y < 1)).astype(int)

def h_jk(j, k, y):
    value = (2**j)*y - k
    value1 = np.where((value >= 0) & (value < 0.5), 1., 0.) 
    value2 = np.where((value >= 0.5) & (value < 1), 1., 0,) 
    return (2**(j/2))*(value1 - value2)
    
def h_jk_phi(j, k, m, J):
    value1 = np.where((k*(2**(J + 1)) <= m*(2**(j))) & ((m + 1)*(2**(j + 1)) <= (2*k + 1)*(2**(J + 1))), 1., 0.) 
    value2 = np.where(((2*k + 1)*(2**(J + 1)) <= m*(2**(j + 1))) & ((m + 1)*(2**(j)) <= (k + 1)*(2**(J + 1))), 1., 0.) 
    return 2**(j/2) * (value1 - value2)
    
def h_jk_arr(j, k, i, arr):
    value = (2**j)*arr[i] - k
    value1 = np.where((value >= 0) & (value < 0.5), 1., 0.) 
    value2 = np.where((value >= 0.5) & (value < 1), 1., 0,) 
    return (2**(j/2))*(value1 - value2)

def p_jk(y, j, k):
    return (2**(j/2))*p((2**j)*y - k)

def index_to_jk(n):
    j = int(np.log2(n + 1))
    k = n + 1 - 2**j
    return j, k

def jk_to_index(j, k):
    return 2**j + k - 1

def convert_1d_eta_to_2d_eta(eta_arr):
    eta_0_0 = eta_arr[-1]
    size = eta_arr.shape[0]
    J = int(np.log2(size)) - 1
    eta_2d = np.zeros((J + 1, 2**J), dtype = np.float64)
    for i in range(size - 1):
        j = int(np.log2(i + 1))
        k = i + 1 - 2**j
        eta_2d[j][k] = eta_arr[i]
        
    return eta_2d, eta_0_0

def convert_2d_eta_to_1d_eta(eta_2d, eta_0_0):
    J = eta_2d.shape[0] - 1
    eta_arr = np.zeros(2**(J + 1), dtype = np.float64)
    for j in range(J + 1):
        for k in range(2**j):
            i = 2**j + k - 1
            eta_arr[i] = eta_2d[j][k]
            
    eta_arr[-1] = eta_0_0
    return eta_arr

def f(J, j, k, m):
    if ((k*(2**(J + 1)) <= m*(2**(j))) and ((m + 1)*(2**(j + 1)) <= (2*k + 1)*(2**(J + 1)))):
        return 1
    elif (((2*k + 1)*(2**(J + 1)) <= m*(2**(j + 1))) and ((m + 1)*(2**(j)) <= (k + 1)*(2**(J + 1)))):
        return -1
    else:
        return 0

def calculate_phi(a, b, J, eta_2d, eta_0_0):
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
    eta_3d = np.tile(eta_2d, (2**(J + 1), 1, 1)).transpose(1, 2, 0)
    cur_fun = partial(h_jk_phi, J = J)                                                                                                           
    f_arr = np.fromfunction(cur_fun, (J + 1, 2**J, 2**(J + 1)), dtype = int).astype(np.float64)
    result_arr = f_arr * eta_3d
    res_sum = np.exp(result_arr.sum(axis = (0, 1)) + eta_0_0).sum()
    #print(f"Numpy, const = {np.log((b - a)/(2**(J + 1)))}")
    return np.log(res_sum) + np.log((b - a)/(2**(J + 1)))
    
def calculate_g_without_phi(point, eta_2d, eta_0_0, J):
    #print(point)
    res_sum = eta_0_0*p(point)
    cur_fun = partial(h_jk, y = point)             
    f_arr = np.fromfunction(cur_fun, (J + 1, 2**J), dtype = int).astype(np.float64)
    result_arr = f_arr * eta_2d
    
    return res_sum + result_arr.sum()
    

def L(eta_2d, eta_0_0, J, data_norm, a, b):
    res_sum = eta_0_0*p(data_norm)
    cur_fun = partial(h_jk_arr, arr = data_norm)             
    f_arr = np.fromfunction(cur_fun, (J + 1, 2**J, data_norm.shape[0]), dtype = int).astype(np.float64)
    eta_3d = np.tile(eta_2d, (data_norm.shape[0], 1, 1)).transpose(1, 2, 0)
    result_arr = f_arr * eta_3d
    sums = res_sum + result_arr.sum(axis = (0, 1))
    first_part = sums.sum()
    second_part = data_norm.shape[0]*calculate_phi(a, b, J, eta_2d, eta_0_0)
    return first_part - second_part

def dens_estimation(data_norm, J, eta_2d, eta_0_0, phi):
    res_sum = eta_0_0*p(data_norm)
    cur_fun = partial(h_jk_arr, arr = data_norm)             
    f_arr = np.fromfunction(cur_fun, (J + 1, 2**J, data_norm.shape[0]), dtype = int).astype(np.float64)
    eta_3d = np.tile(eta_2d, (data_norm.shape[0], 1, 1)).transpose(1, 2, 0)
    result_arr = f_arr * eta_3d
    sums = res_sum + result_arr.sum(axis = (0, 1))
    sums -= phi
    return np.exp(sums)
    
def sigma_arr_with_noise(sigma_arr, sigma_noise):
    sigma_arr_noise = np.array([np.sqrt((sigma_arr[0]**2 + sigma_noise**2)), 
                                np.sqrt(sigma_arr[1]**2 + sigma_noise**2)])
    return sigma_arr_noise

def sigma_arr_with_noise_sqr(sigma_arr, sigma_noise):
    sigma_arr_noise_sqr = np.array([(sigma_arr[0]**2 + sigma_noise**2), 
                                (sigma_arr[1]**2 + sigma_noise**2)])
    return sigma_arr_noise_sqr

def calculate_gauss_int(c, d, mu, sigma):
    return norm.cdf(d, loc = mu, scale = sigma) - norm.cdf(c, loc = mu, scale = sigma)

def calculate_int_of_conv(c, d, p_arr, mu_arr, sigma_arr, sigma_noise):
    sigma_arr_noise = sigma_arr_with_noise(sigma_arr, sigma_noise)
    first_part = p_arr[0]*calculate_gauss_int(c, d, mu_arr[0], sigma_arr_noise[0])
    second_part = p_arr[1]*calculate_gauss_int(c, d, mu_arr[1], sigma_arr_noise[1])
    return first_part + second_part

def calculate_int_of_sqr_conv(c, d, p_arr, mu_arr, sigma_arr, sigma_noise, print_results = False):
    sigma_arr_noise = sigma_arr_with_noise(sigma_arr, sigma_noise)
    sigma_arr_noise_sqr = sigma_arr_with_noise_sqr(sigma_arr, sigma_noise)
    sigma_sum_noise_sqr = np.sum(sigma_arr_noise_sqr)
    sigma_sum_noise = np.sqrt(sigma_sum_noise_sqr)
    sigma_harmon_noise = np.sqrt(sigma_arr_noise_sqr[0]*sigma_arr_noise_sqr[1]/sigma_sum_noise_sqr)
    mu_convex_noise = (mu_arr[0]*sigma_arr_noise_sqr[1] + 
                       mu_arr[1]*sigma_arr_noise_sqr[0])/(sigma_sum_noise_sqr)
    first_part_prod = (p_arr[0]**2)*(1/(2*np.sqrt(np.pi)*sigma_arr_noise[0]))
    first_part_int = calculate_gauss_int(c, d, mu_arr[0], sigma_arr_noise[0]/(np.sqrt(2)))
    first_part = first_part_int*first_part_prod
    second_part_int = calculate_gauss_int(c, d, mu_convex_noise, sigma_harmon_noise)
    second_part_prod = 2*p_arr[0]*p_arr[1]*np.exp(-((mu_arr[0] - 
        mu_arr[1])**2)/(2*sigma_sum_noise_sqr))/(np.sqrt(2*np.pi) * sigma_sum_noise)
    second_part = second_part_prod*second_part_int
    third_part_int = calculate_gauss_int(c, d, mu_arr[1], sigma_arr_noise[1]/(np.sqrt(2)))
    third_part_prod = (p_arr[1]**2)*(1/(2*np.sqrt(np.pi)*sigma_arr_noise[1]))
    third_part = third_part_prod*third_part_int
    if print_results:
        print(f"Numpy, sigma_arr_noise = {sigma_arr_noise.dtype}")
        print(f"Numpy, sigma_arr_noise_sqr = {sigma_arr_noise_sqr.dtype}")
        print(f"Numpy, sigma_sum_noise_sqr = {sigma_sum_noise_sqr.dtype}")
        print(f"Numpy, sigma_sum_noise_sqr = {sigma_sum_noise.dtype}")
        print(f"Numpy, sigma_harmon_noise = {sigma_harmon_noise.dtype}")
        print(f"Numpy, mu_convex_noise = {mu_convex_noise.dtype}")
        print(f"Numpy, first part in integral of sqr conv = {first_part_int.dtype}") 
        print(f"Numpy, second part in integral of sqr conv = {second_part_int.dtype}")  
        print(f"Numpy, third part in integral of sqr conv = {third_part_int.dtype}") 
        print(f"Numpy, first prod in integral of sqr conv = {first_part_prod.dtype}") 
        print(f"Numpy, second prod in integral of sqr conv = {second_part_prod.dtype}")  
        print(f"Numpy, third prod in integral of sqr conv = {third_part_prod.dtype}")
    return first_part + second_part + third_part

def calculate_NCS(J, eta_2d, eta_0_0, p_arr, mu_arr, sigma_arr, a, b, sigma_noise, print_results = False):
    phi = calculate_phi(a, b, J, eta_2d, eta_0_0)
    #print(f"Numpy, phi = {phi}")
    second_part = 2*calculate_int_of_conv(a, b, p_arr, mu_arr, sigma_arr, sigma_noise)
    #print(f"Numpy, eta_2d = {eta_2d}, eta_0 = {eta_0_0}")
    #print(f"Numpy, theta = {theta_arr}")
    m_arr = np.arange(2**(J + 1)).astype(np.float64)
    points_int_from = a + m_arr*(b - a)/(2**(J + 1))
    points_int_to = a + (m_arr + 1)*(b - a)/(2**(J + 1))
    second_prod_term_arr = calculate_int_of_sqr_conv(points_int_from, points_int_to, p_arr, mu_arr, sigma_arr, sigma_noise, print_results)
    cur_fun = partial(h_jk_phi, J = J)                                                                                               
    f_arr = np.fromfunction(cur_fun, (J + 1, 2**J, 2**(J + 1)), dtype = int).astype(np.float64)
    eta_3d = np.tile(eta_2d, (2**(J + 1), 1, 1)).transpose(1, 2, 0)
    denominators = np.exp((f_arr * eta_3d).sum(axis = (0, 1)) + eta_0_0 - phi)
    fractions = second_prod_term_arr/denominators
    first_part = fractions.sum()
    if print_results:
        print(f"Numpy, phi = {phi}")
        print(f"Numpy, second_prod_term_arr = {second_prod_term_arr}")
        print(f"Numpy, denominators = {denominators}")
        print(f"Numpy distance, first_part = {first_part}, second_part = {second_part}")
    return first_part - second_part + 1.
    
def calculate_prior(prior_arr, prior_0, prior_cov_inv):
    diff = prior_arr - prior_0
    #print(f"diff = {diff}")
    return -0.5*np.dot(diff, np.dot(prior_cov_inv, diff))
    
def calculate_log_likelihood(params, data_norm, 
                             a, b, J, lambda_coef, dist_type,
                             sigma_noise, eta_0, eta_cov_inv, 
                             theta_0, theta_cov_inv, 
                             print_results = False):
    params = params['points']
    eta_arr = params[:2**(J + 1)]
    theta_arr = params[2**(J + 1):]
    eta_2d, eta_0_0 = convert_1d_eta_to_2d_eta(eta_arr)
    p_arr, mu_arr, sigma_arr = GMM_params_from_theta(theta_arr)
    L_data = L(eta_2d, eta_0_0, J, data_norm, a, b)
    #print(f"L_data = {L_data}")
    if (dist_type == dist_neiman_chi_sq):
        measure_dist = calculate_NCS(J, eta_2d, eta_0_0, p_arr, mu_arr, sigma_arr, a, b, sigma_noise, print_results = print_results)
    #print(f"measure_dist = {measure_dist}")
    eta_prior_term = calculate_prior(eta_arr, eta_0, eta_cov_inv)
    theta_prior_term = calculate_prior(theta_arr, theta_0, theta_cov_inv)
    #print(f"eta_arr = {eta_arr}")
    #print(f"eta_prior_term = {eta_prior_term}")
    #print(f"theta_prior_term = {theta_prior_term}")
    if print_results:
        print(f"L_data = {L_data}")
        print(f"measure_dist = {measure_dist}")
        print(f"eta_prior_term = {eta_prior_term}")
        print(f"theta_prior_term = {theta_prior_term}")
    return L_data + lambda_coef*measure_dist + eta_prior_term + theta_prior_term
    
def posterior_log_likelihood(calculate_log_likelihood, data_norm,
                             a, b, J, 
			     lambda_coef, dist_type,
                             sigma_noise, eta_0, eta_cov_inv, 
                             theta_0, theta_cov_inv, 
                             print_results = False):
    return partial(calculate_log_likelihood, 
                   data_norm = data_norm, a = a, b = b, 
                   J = J, lambda_coef = lambda_coef,
                   dist_type = dist_type, sigma_noise = sigma_noise,
                   eta_0 = eta_0, eta_cov_inv = eta_cov_inv, 
                   theta_0 = theta_0, theta_cov_inv = theta_cov_inv,
                   print_results = print_results)
