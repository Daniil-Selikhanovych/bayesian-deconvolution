import numpy as np
import torch
from .numpy_log_likelihood import (dist_neiman_chi_sq, 
                                  dist_hellinger,
                                  dist_kl_divergence,
				  theta_from_GMM_params)

data_num_default = 100
J_default = 2
lambda_coef_default = 1

random_state_data_default = 42
random_state_eta_default = 4
random_state_theta_default = 12

p_arr_default = np.array([0.6, 0.4], dtype=np.float64)
mu_arr_default = np.array([4, -5], dtype=np.float64)
sigma_arr_default = np.array([1, 1], dtype=np.float64)
sigma_noise_default = float(1)

sigma_gamma = float(1)
sigma_eta_0_default_noise = float(1)
eta_arr_default = np.array([0.08, 0.6, 0.5, -0.5, 0.6, -0.8, 1.75, 0], dtype=np.float64)
eta_size = eta_arr_default.shape[0]
np.random.seed(random_state_eta_default)
eta_0_default = eta_arr_default + sigma_eta_0_default_noise*np.random.randn(eta_size)
eta_cov_default = (sigma_gamma**2)*np.eye(eta_size)

sigma_theta = 1
sigma_theta_0_default_noise = 1
theta_arr_default = theta_from_GMM_params(p_arr_default, mu_arr_default, sigma_arr_default)
theta_size = theta_arr_default.shape[0]
np.random.seed(random_state_theta_default)
theta_0_default = theta_arr_default + sigma_theta_0_default_noise*np.random.randn(theta_size)
theta_cov_default = (sigma_theta**2)*np.eye(theta_size)

dist_type_default = dist_neiman_chi_sq
dist_type_arr = [dist_neiman_chi_sq, dist_hellinger, dist_kl_divergence]

colors_default = {0: 'green', 1: 'purple'}

torch_pi = torch.acos(torch.zeros(1)) * 2 
torch_sqrt_2 = torch.sqrt(torch.tensor(2, dtype=torch.float64))

device_default = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype_default = torch.float64
