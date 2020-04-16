import numpy as np
import torch
from torch.distributions import Normal
from functools import partial
from .numpy_log_likelihood import (calculate_a_b, index_to_jk, p, 
                                   h_jk, f, sigma_arr_with_noise)
from .numpy_log_likelihood import dist_neiman_chi_sq
from .constants import torch_pi, torch_sqrt_2

def convert_1d_eta_to_2d_eta_torch(eta_arr):
    eta_0 = eta_arr[-1]
    size = eta_arr.shape[0]
    J = int(np.log2(size)) - 1
    eta_2d = [torch.from_numpy(np.array(
        [float(0) for k in range(2**j)])) for j in range(J + 1)]
    for i in range(size - 1):
        j, k = index_to_jk(i)
        eta_2d[j][k] = eta_arr[i]
        
    return eta_2d, eta_0

def calculate_phi_torch(a, b, eta_arr):
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
    eta_2d, eta_0 = convert_1d_eta_to_2d_eta_torch(eta_arr)
    #print(f"Torch, eta_0 = {eta_0}")
    size = eta_arr.shape[0]
    J = int(np.log2(size)) - 1
    res_sum = torch.zeros(1, dtype=torch.float64)[0]
    for m in range(2**(J + 1)):
        cur_sum = eta_0
        for j in range(J + 1):
            for k in range(2**j):
                сur = np.array([(2**(j/2))*f(J, j, k, m)], dtype = np.float64)
                cur = torch.mul(eta_2d[j][k], torch.from_numpy(сur)[0])
                #print(f"Torch, m = {m}, j = {j}, k = {k}, cur = {cur}")
                cur_sum = cur_sum + cur
        #print(f"Torch, phi, exp of part = {cur_sum}")
        res_sum += torch.exp(cur_sum)
    const = (b - a)/(2**(J + 1))
    const_tensor = torch.from_numpy(np.array([const]))[0]
    #print(f"Torch, const = {torch.log(const_tensor)}")
    return torch.log(res_sum) + torch.log(const_tensor)

def calculate_g_without_phi_torch(y, eta_arr, a, b):
    eta_2d, eta_0 = convert_1d_eta_to_2d_eta_torch(eta_arr)
    point = (y - a)/(b - a)
    res_sum = eta_0*p(point)
    size = eta_arr.shape[0]
    J = int(np.log2(size)) - 1
    for j in range(J + 1):
        for k in range(2**j):
            res_sum += eta_2d[j][k]*h_jk(point, j, k)
    
    return res_sum

def L_torch(eta_arr, data):
    a, b = calculate_a_b(data)
    first_part = np.sum(np.array([calculate_g_without_phi_torch(y, eta_arr, a, b) for y in data])) 
    second_part = data.shape[0]*calculate_phi_torch(a, b, eta_arr)
    return first_part - second_part

def GMM_params_from_theta_torch(theta_arr):
    p = 1/(1 + torch.exp(-theta_arr[0]))
    p_arr = torch.tensor([p, 1 - p], dtype=torch.float64)
    mu_arr = theta_arr[1:3]
    sigma_arr = torch.tensor([torch.exp(theta_arr[3]/2), torch.exp(theta_arr[4]/2)], dtype=torch.float64)
    return p_arr, mu_arr, sigma_arr

def sigma_arr_with_noise_torch(sigma_arr, sigma_noise):
    sigma_arr_noise = torch.tensor([torch.sqrt((sigma_arr[0]**2 + sigma_noise**2)), 
                                torch.sqrt(sigma_arr[1]**2 + sigma_noise**2)], dtype=torch.float64)
    return sigma_arr_noise

def sigma_arr_with_noise_sqr_torch(sigma_arr, sigma_noise):
    sigma_arr_noise_sqr = torch.tensor([(sigma_arr[0]**2 + sigma_noise**2), 
                                (sigma_arr[1]**2 + sigma_noise**2)], dtype=torch.float64)
    return sigma_arr_noise_sqr

def calculate_gauss_int_torch(c, d, mu, sigma):
    torch_normal = Normal(loc=mu, scale=sigma)
    return torch_normal.cdf(d) - torch_normal.cdf(c)

def calculate_int_of_conv_torch(c, d, theta_arr, sigma_noise):
    p_arr, mu_arr, sigma_arr = GMM_params_from_theta_torch(theta_arr)
    sigma_arr_noise = sigma_arr_with_noise(sigma_arr, sigma_noise)
    first_part = p_arr[0]*calculate_gauss_int_torch(c, d, mu_arr[0], sigma_arr_noise[0])
    second_part = p_arr[1]*calculate_gauss_int_torch(c, d, mu_arr[1], sigma_arr_noise[1])
    return first_part + second_part

def calculate_int_of_sqr_conv_torch(c, d, theta_arr, sigma_noise):
    p_arr, mu_arr, sigma_arr = GMM_params_from_theta_torch(theta_arr)
    sigma_arr_noise = sigma_arr_with_noise_torch(sigma_arr, sigma_noise)
    sigma_arr_noise_sqr = sigma_arr_with_noise_sqr_torch(sigma_arr, sigma_noise)
    sigma_sum_noise_sqr = torch.sum(sigma_arr_noise_sqr)
    sigma_sum_noise = torch.sqrt(sigma_sum_noise_sqr)
    sigma_harmon_noise = torch.sqrt(sigma_arr_noise_sqr[0]*sigma_arr_noise_sqr[1]/sigma_sum_noise_sqr)
    mu_convex_noise = (mu_arr[0]*sigma_arr_noise_sqr[1] + 
                       mu_arr[1]*sigma_arr_noise_sqr[0])/(sigma_sum_noise_sqr)
    first_part = (p_arr[0]**2)*(1/(2*torch.sqrt(torch_pi)*sigma_arr_noise[0]))
    first_part = calculate_gauss_int_torch(c, d, mu_arr[0], sigma_arr_noise[0]/(torch_sqrt_2))*first_part
    second_part = 2*p_arr[0]*p_arr[1]*torch.exp(-((mu_arr[0] - 
        mu_arr[1])**2)/(2*sigma_sum_noise_sqr))/(
        torch.sqrt(2*torch_pi) * sigma_sum_noise)*calculate_gauss_int_torch(c, 
                                                            d, mu_convex_noise, sigma_harmon_noise)
    third_part = (p_arr[1]**2)*(1/(2*torch.sqrt(torch_pi)*sigma_arr_noise[1]))*calculate_gauss_int_torch(c, 
                                                            d, mu_arr[1], sigma_arr_noise[1]/(torch_sqrt_2))
    return first_part + second_part + third_part

def calculate_NCS_torch(eta_arr, theta_arr, a, b, sigma_noise):
    phi = calculate_phi_torch(a, b, eta_arr)
    #print(f"PyTorch, phi = {phi}")
    second_part = 2*calculate_int_of_conv_torch(a, b, theta_arr, sigma_noise)
    first_part = 0
    eta_2d, eta_0 = convert_1d_eta_to_2d_eta_torch(eta_arr)
    #print(f"PyTorch, eta_2d = {eta_2d}, eta_0 = {eta_0}")
    #print(f"PyTorch, theta = {theta_arr}")
    size = eta_arr.shape[0]
    J = int(np.log2(size)) - 1
    for m in range(2**(J + 1)):
        denominator = eta_0 - phi
        for j in range(J + 1):
            for k in range(2**j):
                denominator += eta_2d[j][k]*(2**(j/2))*f(J, j, k, m)
        #print(f"Pytroch, exp of denominator = {denominator}")
        denominator = torch.exp(denominator)
        point_int_from = a + m*(b - a)/(2**(J + 1))
        point_int_to = a + (m + 1)*(b - a)/(2**(J + 1))
        second_prod_term = calculate_int_of_sqr_conv_torch(point_int_from, 
                                                           point_int_to, theta_arr, sigma_noise)
        #print(f"Pytorch, denominator = {denominator}, second_prod_term = {second_prod_term}")
        first_part += (second_prod_term/denominator)
    #print(f"PyTorch distance, first_part = {first_part}, second_part = {second_part}")
    return first_part - second_part + 1 
        
def calculate_prior_torch(prior_arr, prior_0, prior_cov_inv):
    #print(prior_arr)
    #print(prior_0)
    diff = prior_arr - prior_0
    return -0.5*torch.dot(diff, torch.matmul(prior_cov_inv, diff))

def calculate_log_likelihood_torch(params, data, 
                             J, lambda_coef, dist_type,
                             sigma_noise, eta_0, 
                             eta_cov_inv, theta_0,
                             theta_cov_inv):
    params = params['points'][0]
    #print(params)
    eta_arr = params[:2**(J + 1)]
    theta_arr = params[2**(J + 1):]
    a, b = calculate_a_b(data)
    L_data = L_torch(eta_arr, data)
    if (dist_type == dist_neiman_chi_sq):
        measure_dist = calculate_NCS_torch(eta_arr, theta_arr, a, b, sigma_noise)
    eta_prior_term = calculate_prior_torch(eta_arr, eta_0, eta_cov_inv)
    theta_prior_term = calculate_prior_torch(theta_arr, theta_0, theta_cov_inv)
    #print(f"L_data = {L_data}")
    #print(f"measure_dist = {measure_dist}")
    #print(f"eta_prior_term = {eta_prior_term}")
    #print(f"theta_prior_term = {theta_prior_term}")
    return L_data + lambda_coef*measure_dist + eta_prior_term + theta_prior_term
    
def calculate_minus_log_likelihood_torch(params, data, 
                             J, lambda_coef, dist_type,
                             sigma_noise, eta_0, 
                             eta_cov_inv, theta_0,
                             theta_cov_inv):
    params = params['points'][0]
    #print(params)
    eta_arr = params[:2**(J + 1)]
    theta_arr = params[2**(J + 1):]
    a, b = calculate_a_b(data)
    L_data = L_torch(eta_arr, data)
    if (dist_type == dist_neiman_chi_sq):
        measure_dist = calculate_NCS_torch(eta_arr, theta_arr, a, b, sigma_noise)
    eta_prior_term = calculate_prior_torch(eta_arr, eta_0, eta_cov_inv)
    theta_prior_term = calculate_prior_torch(theta_arr, theta_0, theta_cov_inv)
    #print(f"L_data = {L_data}")
    #print(f"measure_dist = {measure_dist}")
    #print(f"eta_prior_term = {eta_prior_term}")
    #print(f"theta_prior_term = {theta_prior_term}")
    return -(L_data + lambda_coef*measure_dist + eta_prior_term + theta_prior_term)
    
def posterior_log_likelihood_torch(calculate_log_likelihood_torch, data, J, 
			     lambda_coef, dist_type,
                             sigma_noise, eta_0, eta_cov_inv, 
                             theta_0, theta_cov_inv):
    return partial(calculate_log_likelihood_torch, 
                   data = data, J = J, lambda_coef = lambda_coef,
                   dist_type = dist_type, sigma_noise = sigma_noise,
                   eta_0 = eta_0, eta_cov_inv = eta_cov_inv, 
                   theta_0 = theta_0, theta_cov_inv = theta_cov_inv)
    
def potential_fn_MCMC(calculate_minus_log_likelihood, data, J, 
			     lambda_coef, dist_type,
                             sigma_noise, eta_0, eta_cov_inv, 
                             theta_0, theta_cov_inv):
    return partial(calculate_minus_log_likelihood_torch, 
                   data = data, J = J, lambda_coef = lambda_coef,
                   dist_type = dist_type, sigma_noise = sigma_noise,
                   eta_0 = eta_0, eta_cov_inv = eta_cov_inv, 
                   theta_0 = theta_0, theta_cov_inv = theta_cov_inv)
   
