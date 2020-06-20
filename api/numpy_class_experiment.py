import numpy as np
import matplotlib.pyplot as plt
from numpy import random, linalg, histogram
from scipy.stats import norm

from .numpy_log_likelihood import (theta_from_GMM_params, sigma_arr_with_noise,
                                   convert_1d_eta_to_2d_eta, calculate_a_b, 
                                   calculate_phi, dens_estimation)
                                   
from .constants import (data_num_default, random_state_data_default,
			random_state_eta_default, random_state_theta_default,
			p_arr_default, mu_arr_default,
			sigma_arr_default, sigma_noise_default,
			eta_0_default, eta_cov_default,
			theta_0_default, theta_cov_default,
			dist_type_default, dist_type_arr, colors_default)
#p_arr_default = constants.p_arr_default
#mu_arr_default = constants.mu_arr_default
#sigma_arr_default = constants.sigma_arr_default
#sigma_noise_default = constants.sigma_noise_default
#eta_0_default = constants.eta_0_default
#eta_cov_default = constants.eta_cov_default
#theta_0_default = constants.theta_0_default
#theta_cov_default = constants.theta_cov_default
#dist_type_default = constants.dist_type_default

def GMM_density(x, p_arr, mu_arr, sigma_arr):
    x_arr = x*np.ones(p_arr.shape[0])
    return np.sum(p_arr*norm.pdf(x_arr, loc=mu_arr, scale=sigma_arr))

class Deconv1dExperiment():
    def __init__(self, p_arr = p_arr_default, mu_arr = mu_arr_default,
                 sigma_arr = sigma_arr_default, sigma_noise = sigma_noise_default,
                 eta_0 = eta_0_default, eta_cov = eta_cov_default, theta_0 = theta_0_default,
                 theta_cov = theta_cov_default, dist_type = dist_type_default):

        if (dist_type not in dist_type_arr):
            raise ValueError("Only three options are now available: " +
                             f"dist_type='{dist_neiman_chi_sq}', dist_type='{dist_hellinger}'" +
                             f", dist_type='{dist_kl_divergence}'")
        if (np.sum(p_arr) != 1 or not np.all(p_arr >= 0)):
            raise ValueError("p_arr should be from probabilistic simplex")
        
        super().__init__()
        
        self.p_arr = p_arr
        self.mu_arr = mu_arr
        self.sigma_arr = sigma_arr
        self.sigma_noise = sigma_noise
        
        self.eta_0 = eta_0
        self.eta_cov = eta_cov
        
        self.theta_0 = theta_0
        self.theta_cov = theta_cov
        
        self.dist_type = dist_type
        
        u_eta, s_eta, vh_eta = linalg.svd(self.eta_cov, full_matrices=True)
        s_sqrt_eta = np.diag(np.sqrt(s_eta))
        s_inv_eta = np.diag(s_eta**(-1))
        self.eta_cov_sqrt = np.dot(u_eta, np.dot(s_sqrt_eta, vh_eta))
        self.eta_cov_inv = np.dot(u_eta, np.dot(s_inv_eta, vh_eta))
        
        u_theta, s_theta, vh_theta = linalg.svd(self.theta_cov, full_matrices=True)
        s_sqrt_theta = np.diag(np.sqrt(s_theta))
        s_inv_theta = np.diag(s_theta**(-1))
        self.theta_cov_sqrt = np.dot(u_theta, np.dot(s_sqrt_theta, vh_theta))
        self.theta_cov_inv = np.dot(u_theta, np.dot(s_inv_theta, vh_theta))
        
        self.theta_arr = theta_from_GMM_params(p_arr, mu_arr, sigma_arr)
        
    def plot_real_distribution(self):
        fig = plt.figure(figsize=(10,5))

        plt.xlabel(r'$x$') 
        plt.ylabel('Density of GMM') 
        plt.title('Density of GMM') 

        mu_max = np.max(self.mu_arr)
        mu_min = np.min(self.mu_arr)
        diff = mu_max - mu_min
        x_arr = np.linspace(mu_min - diff, mu_max + diff, num=1000)
        GMM_density_arr = np.array([GMM_density(x, self.p_arr, 
                                                self.mu_arr, self.sigma_arr) for x in x_arr])

        plt.plot(x_arr, GMM_density_arr, label = r'Density of GMM')
        for mu in self.mu_arr:
            plt.axvline(x=mu, color='r')

        plt.legend()
        plt.grid(True) 
        return fig
        
    def generate_noise_data(self, data_num = data_num_default, random_state_data=random_state_data_default):
        np.random.seed(random_state_data)
        sigma_noise_arr = sigma_arr_with_noise(self.sigma_arr, self.sigma_noise)
        component_choose = np.random.choice(self.p_arr.shape[0], data_num, p=self.p_arr)
        data = np.array([norm.rvs(size=1, loc = self.mu_arr[component_choose[i]],
                         scale = sigma_noise_arr[component_choose[i]])[0] 
                         for i in range(data_num)])
        return data, component_choose
    
    def plot_real_distribution_with_data(self, data, component_choose):
        fig = self.plot_real_distribution()
        for i in range(data.shape[0]):
            plt.scatter(data[i], 0, marker = '*', c=colors_default[component_choose[i]])
        return fig
        
    def generate_theta_from_prior(self, random_state_theta = random_state_theta_default):
        return generate_from_prior(random_state_theta, self.theta_cov_sqrt, self.theta_0)
    
    def generate_eta_from_prior(self, random_state_eta = random_state_eta_default):
        return generate_from_prior(random_state_eta, self.eta_cov_sqrt, self.eta_0)
    
    def plot_real_distribution_with_dens_estimation(self, data, component_choose, eta_arr):
        fig = self.plot_real_distribution_with_data(data, component_choose)
        a, b = calculate_a_b(data)
        data = (data - a)/(b - a)
        eta_2d, eta_0 = convert_1d_eta_to_2d_eta(eta_arr)
        size = eta_arr.shape[0]
        J = int(np.log2(size)) - 1
        phi = calculate_phi(a, b, J, eta_2d, eta_0)
        y_arr = np.linspace(a, b, num = 1000, endpoint=False)
        y_arr_norm = (y_arr - a)/(b - a)
        density_est_arr = np.array([dens_estimation(point, J, eta_2d, eta_0, phi) for point in y_arr_norm])
        plt.plot(y_arr, density_est_arr, label = r'Density estimation', c = 'black')
        plt.legend()
        return fig
