import numpy as np
import torch
from torch.distributions import Normal
from functools import partial

from .numpy_log_likelihood import (calculate_a_b, index_to_jk, 
                                   h_jk, h_jk_phi, h_jk_arr,
                                   sigma_arr_with_noise)
from .numpy_log_likelihood import dist_neiman_chi_sq
from .constants import torch_pi, torch_sqrt_2

def p_torch(y):
    #print(y)
    return ((y >= 0) & (y < 1)).int()

def calculate_a_b_torch(data):
    a = torch.min(data)
    b = torch.max(data)
    return a, b

def convert_1d_eta_to_2d_eta_torch(eta_arr):
    eta_0 = eta_arr[-1]
    size = eta_arr.shape[0]
    J = int(np.log2(size)) - 1
    eta_2d = torch.zeros((J + 1, 2**J), dtype = torch.float64)
    for i in range(size - 1):
        j = int(np.log2(i + 1))
        k = i + 1 - 2**j
        eta_2d[j][k] = eta_arr[i]
        
    return eta_2d, eta_0

def calculate_phi_torch(a, b, J, eta_2d, eta_0_0):
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
    eta_3d = eta_2d.expand((2**(J + 1), J + 1, 2**J)).permute(1, 2, 0)
    cur_fun = partial(h_jk_phi, J = J)                                                                                                           
    f_arr = torch.from_numpy(np.fromfunction(cur_fun, (J + 1, 2**J, 2**(J + 1)), dtype = int)).to(dtype = torch.float64)
    result_arr = f_arr * eta_3d
    res_sum = torch.exp(result_arr.sum(axis = (0, 1)) + eta_0_0).sum()
    return torch.log(res_sum) + torch.log((b - a)/(2**(J + 1)))
    #print(f"Torch, const = {torch.log(const_tensor)}")

def calculate_g_without_phi_torch(point, eta_2d, eta_0_0, J):
    res_sum = eta_0_0*p(point)
    cur_fun = partial(h_jk, y = point)             
    f_arr = torch.from_numpy(np.fromfunction(cur_fun, (J + 1, 2**J), dtype = int)).to(dtype = torch.float64)
    result_arr = f_arr * eta_2d
    
    return res_sum + result_arr.sum()

def L_torch(eta_2d, eta_0_0, J, data_norm, a, b):
    res_sum = eta_0_0*p_torch(data_norm)
    cur_fun = partial(h_jk_arr, arr = data_norm.numpy())             
    f_arr = torch.from_numpy(np.fromfunction(cur_fun, (J + 1, 2**J, data_norm.shape[0]), dtype = int)).to(dtype = torch.float64)
    eta_3d = eta_2d.expand((data_norm.shape[0], J + 1, 2**J)).permute(1, 2, 0)
    result_arr = f_arr * eta_3d
    sums = res_sum + result_arr.sum(axis = (0, 1))
    first_part = sums.sum()
    second_part = data_norm.shape[0]*calculate_phi_torch(a, b, J, eta_2d, eta_0_0)
    return first_part - second_part

def GMM_params_from_theta_torch(theta_arr):
    p = 1/(1 + torch.exp(-theta_arr[0]))
    p_arr = torch.tensor([p, 1 - p], dtype = torch.float64)
    mu_arr = theta_arr[1:3]
    sigma_arr = torch.tensor([torch.exp(theta_arr[3]/2), torch.exp(theta_arr[4]/2)], dtype = torch.float64)
    return p_arr, mu_arr, sigma_arr

def sigma_arr_with_noise_torch(sigma_arr, sigma_noise):
    sigma_arr_noise = torch.tensor([torch.sqrt((sigma_arr[0]**2 + sigma_noise**2)), 
                                torch.sqrt(sigma_arr[1]**2 + sigma_noise**2)], dtype = torch.float64)
    return sigma_arr_noise

def sigma_arr_with_noise_sqr_torch(sigma_arr, sigma_noise):
    sigma_arr_noise_sqr = torch.tensor([(sigma_arr[0]**2 + sigma_noise**2), 
                                (sigma_arr[1]**2 + sigma_noise**2)], dtype = torch.float64)
    return sigma_arr_noise_sqr

def calculate_gauss_int_torch(c, d, mu, sigma):
    torch_normal = Normal(loc = mu, scale = sigma)
    return (torch_normal.cdf(d) - torch_normal.cdf(c)).to(dtype = torch.float64)

def calculate_int_of_conv_torch(c, d, p_arr, mu_arr, sigma_arr, sigma_noise):
    sigma_arr_noise = sigma_arr_with_noise(sigma_arr, sigma_noise)
    first_part = p_arr[0]*calculate_gauss_int_torch(c, d, mu_arr[0], sigma_arr_noise[0])
    second_part = p_arr[1]*calculate_gauss_int_torch(c, d, mu_arr[1], sigma_arr_noise[1])
    return first_part + second_part

def calculate_int_of_sqr_conv_torch(c, d, p_arr, mu_arr, sigma_arr, sigma_noise, print_results = False):
    sigma_arr_noise = sigma_arr_with_noise_torch(sigma_arr, sigma_noise)
    sigma_arr_noise_sqr = sigma_arr_with_noise_sqr_torch(sigma_arr, sigma_noise)
    sigma_sum_noise_sqr = torch.sum(sigma_arr_noise_sqr)
    sigma_sum_noise = torch.sqrt(sigma_sum_noise_sqr)
    sigma_harmon_noise = torch.sqrt(sigma_arr_noise_sqr[0]*sigma_arr_noise_sqr[1]/sigma_sum_noise_sqr)
    mu_convex_noise = (mu_arr[0]*sigma_arr_noise_sqr[1] + 
                       mu_arr[1]*sigma_arr_noise_sqr[0])/(sigma_sum_noise_sqr)
    first_part_prod = (p_arr[0]**2)*(1/(2*torch.sqrt(torch_pi)*sigma_arr_noise[0]))
    first_part_int = calculate_gauss_int_torch(c, d, mu_arr[0], sigma_arr_noise[0]/(torch_sqrt_2))
    first_part = first_part_int*first_part_prod
    second_part_int = calculate_gauss_int_torch(c, d, mu_convex_noise, sigma_harmon_noise)
    second_part_prod = 2*p_arr[0]*p_arr[1]*torch.exp(-((mu_arr[0] - 
        mu_arr[1])**2)/(2*sigma_sum_noise_sqr))/(
        torch.sqrt(2*torch_pi) * sigma_sum_noise)
    second_part = second_part_prod*second_part_int
    third_part_int = calculate_gauss_int_torch(c, d, mu_arr[1], sigma_arr_noise[1]/(torch_sqrt_2))
    third_part_prod = (p_arr[1]**2)*(1/(2*torch.sqrt(torch_pi)*sigma_arr_noise[1]))
    third_part = third_part_prod*third_part_int
    if print_results:
        print(f"Torch, sigma_arr_noise = {sigma_arr_noise.numpy().dtype}")
        print(f"Torch, sigma_arr_noise_sqr = {sigma_arr_noise_sqr.numpy().dtype}")
        print(f"Torch, sigma_sum_noise_sqr = {sigma_sum_noise_sqr.numpy().dtype}")
        print(f"Torch, sigma_sum_noise_sqr = {sigma_sum_noise.numpy().dtype}")
        print(f"Torch, sigma_harmon_noise = {sigma_harmon_noise.numpy().dtype}")
        print(f"Torch, mu_convex_noise = {mu_convex_noise.numpy().dtype}")
        print(f"Torch, first part in integral of sqr conv = {first_part_int.numpy().dtype}") 
        print(f"Torch, second part in integral of sqr conv = {second_part_int.numpy().dtype}")  
        print(f"Torch, third part in integral of sqr conv = {third_part_int.numpy().dtype}")
        print(f"Torch, first prod in integral of sqr conv = {first_part_prod.numpy().dtype}") 
        print(f"Torch, second prod in integral of sqr conv = {second_part_prod.numpy().dtype}")  
        print(f"Torch, third prod in integral of sqr conv = {third_part_prod.numpy().dtype}")                                                          
    return first_part + second_part + third_part

def calculate_NCS_torch(J, eta_2d, eta_0_0, p_arr, mu_arr, sigma_arr, a, b, sigma_noise, print_results = False):
    phi = calculate_phi_torch(a, b, J, eta_2d, eta_0_0)
    #print(f"Torch, phi = {phi}")
    second_part = 2*calculate_int_of_conv_torch(a, b, p_arr, mu_arr, sigma_arr, sigma_noise)
    #print(f"Torch, eta_2d = {eta_2d}, eta_0 = {eta_0_0}")
    #print(f"Torch, theta = {theta_arr}")
    m_arr = torch.arange(2**(J + 1)).to(dtype = torch.float64)
    points_int_from = a + m_arr*(b - a)/(2**(J + 1))
    points_int_to = a + (m_arr + 1)*(b - a)/(2**(J + 1))
    second_prod_term_arr = calculate_int_of_sqr_conv_torch(points_int_from, points_int_to, p_arr, mu_arr, sigma_arr, sigma_noise, print_results)
    cur_fun = partial(h_jk_phi, J = J)                                                                                               
    f_arr = torch.from_numpy(np.fromfunction(cur_fun, (J + 1, 2**J, 2**(J + 1)), dtype = int)).to(dtype = torch.float64)
    eta_3d = eta_2d.expand((2**(J + 1), J + 1, 2**J)).permute(1, 2, 0)
    denominators = torch.exp((f_arr * eta_3d).sum(axis = (0, 1)) + eta_0_0 - phi)
    fractions = second_prod_term_arr/denominators
    first_part = fractions.sum()
    if print_results:
        print(f"Torch phi = {phi.item()}")
        print(f"Torch, second_prod_term_arr = {second_prod_term_arr.numpy()}")
        print(f"Torch, denominators = {denominators.numpy()}")
        print(f"Torch distance, first_part = {first_part.item()}, second_part = {second_part.item()}")
    return first_part - second_part + torch.tensor(1.0, dtype = torch.float64)
        
def calculate_prior_torch(prior_arr, prior_0, prior_cov_inv):
    #print(prior_arr)
    #print(prior_0)
    diff = prior_arr - prior_0
    return -0.5*torch.dot(diff, torch.matmul(prior_cov_inv, diff))

def calculate_log_likelihood_torch(params, data_norm, 
                                   a, b, J, lambda_coef, dist_type,
                                   sigma_noise, eta_0, eta_cov_inv, 
                                   theta_0, theta_cov_inv, 
                                   print_results = False):
    params = params['points']
    eta_arr = params[:2**(J + 1)]
    theta_arr = params[2**(J + 1):]
    eta_2d, eta_0_0 = convert_1d_eta_to_2d_eta_torch(eta_arr)
    p_arr, mu_arr, sigma_arr = GMM_params_from_theta_torch(theta_arr)
    L_data = L_torch(eta_2d, eta_0_0, J, data_norm, a, b)
    if (dist_type == dist_neiman_chi_sq):
        measure_dist = calculate_NCS_torch(J, eta_2d, eta_0_0, p_arr, mu_arr, sigma_arr, a, b, sigma_noise, print_results = print_results)
    eta_prior_term = calculate_prior_torch(eta_arr, eta_0, eta_cov_inv)
    theta_prior_term = calculate_prior_torch(theta_arr, theta_0, theta_cov_inv)
    #print(f"eta_arr = {eta_arr}")
    if print_results:
        print(f"L_data = {L_data.item()}")
        print(f"measure_dist = {measure_dist.item()}")
        print(f"eta_prior_term = {eta_prior_term.item()}")
        print(f"theta_prior_term = {theta_prior_term.item()}")
    return L_data + lambda_coef*measure_dist + eta_prior_term + theta_prior_term                               
    
def calculate_minus_log_likelihood_torch(params, data_norm, 
                                         a, b, J, lambda_coef, dist_type,
                                         sigma_noise, eta_0, eta_cov_inv, 
                                         theta_0, theta_cov_inv):
    params = params['points']
    eta_arr = params[:2**(J + 1)]
    theta_arr = params[2**(J + 1):]
    eta_2d, eta_0_0 = convert_1d_eta_to_2d_eta_torch(eta_arr)
    p_arr, mu_arr, sigma_arr = GMM_params_from_theta_torch(theta_arr)
    L_data = L_torch(eta_2d, eta_0_0, J, data_norm, a, b)
    #print(f"L_data = {L_data}")
    if (dist_type == dist_neiman_chi_sq):
        measure_dist = calculate_NCS_torch(J, eta_2d, eta_0_0, p_arr, mu_arr, sigma_arr, a, b, sigma_noise)
    #print(f"measure_dist = {measure_dist}")
    eta_prior_term = calculate_prior_torch(eta_arr, eta_0, eta_cov_inv)
    theta_prior_term = calculate_prior_torch(theta_arr, theta_0, theta_cov_inv)
    #print(f"eta_arr = {eta_arr}")
    #print(f"eta_prior_term = {eta_prior_term}")
    #print(f"theta_prior_term = {theta_prior_term}")
    return -(L_data + lambda_coef*measure_dist + eta_prior_term + theta_prior_term)
    
def posterior_log_likelihood_torch(calculate_log_likelihood_torch, 
                                   data_norm, a, b, J, 
			           lambda_coef, dist_type,
                                   sigma_noise, eta_0, eta_cov_inv, 
                                   theta_0, theta_cov_inv, print_results = False):
    return partial(calculate_log_likelihood_torch, 
                   data_norm = data_norm, a = a, b = b, 
                   J = J, lambda_coef = lambda_coef,
                   dist_type = dist_type, sigma_noise = sigma_noise,
                   eta_0 = eta_0, eta_cov_inv = eta_cov_inv, 
                   theta_0 = theta_0, theta_cov_inv = theta_cov_inv,
                   print_results = print_results)

    
def potential_fn_MCMC(calculate_minus_log_likelihood_torch, 
                      data_norm, a, b, J, 
	              lambda_coef, dist_type,
                      sigma_noise, eta_0, eta_cov_inv, 
                      theta_0, theta_cov_inv):
    return partial(calculate_minus_log_likelihood_torch, 
                   data_norm = data_norm, a = a, b = b, 
                   J = J, lambda_coef = lambda_coef,
                   dist_type = dist_type, sigma_noise = sigma_noise,
                   eta_0 = eta_0, eta_cov_inv = eta_cov_inv, 
                   theta_0 = theta_0, theta_cov_inv = theta_cov_inv)
   
