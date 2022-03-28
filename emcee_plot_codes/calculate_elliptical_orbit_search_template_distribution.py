
import numpy as np
import sympy as sy
sy.init_printing(use_unicode=True)
import cloudpickle
import sys, time
import math
import emcee
from multiprocessing import Pool
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
from schwimmbad import MPIPool
import multiprocessing

parser = argparse.ArgumentParser(description='Generate a template-bank for a user-defined no. of templates for coherent full keplerian elliptical orbit search')
parser.add_argument('-o', '--output_path', help='Output path to save results',  default="/hercules/scratch/vishnu/5D_Template_Bank/template_bank_files/")
parser.add_argument('-t', '--obs_time', help='Observation time in minutes', default='72', type=float)
parser.add_argument('-p', '--porb_low', help='Lower limit of Orbital Period in minutes', default='360', type=float)
parser.add_argument('-P', '--porb_high', help='Upper limit of Orbital Period in minutes', type=float)
parser.add_argument('-c', '--max_comp_mass', help='Maximum mass of Companion in solar mass units', default='8', type=float)
parser.add_argument('-d', '--min_pulsar_mass', help='Minimum mass of Pulsar in solar mass units', default='1.4', type=float)
parser.add_argument('-s', '--spin_period', help='Fastest spin period of pulsar in ms', default='5', type=float)
parser.add_argument('-e', '--max_eccentricity', help='Upper limit of eccentricity', default='0.5', type=float)
parser.add_argument('-f', '--fraction', help='Probability fraction of orbits of different inclination angles', default='1', type=float)
parser.add_argument('-b', '--coverage', help='Coverage of template-bank', default='0.9', type=float)
parser.add_argument('-m', '--mismatch', help='Mismatch of template-bank', default='0.2', type=float)
parser.add_argument('-n', '--ncpus', help='Number of CPUs to use for calculation', default='32', type=int)
parser.add_argument('-z', '--templates', help='Number of templates to be added to template-bank', default='100', type=int)
parser.add_argument('-file', '--filename', help='output filename', default='5D_template_bank', type=str)

args = parser.parse_args()
args.porb_high = args.porb_high if args.porb_high else args.obs_time * 10
phi, pi, T, f0, f, tau, psi, E, cosE, sinE, t, e, c0, c1, c2, c3, c4, c5, c6, c7, s1, s2, s3, s4, s5, s6, s7, M, omega, alpha, kappa, u, A, Omega, U, X, Y = sy.symbols(""" phi
pi T f0 f tau psi E cosE sinE t e c0 c1 c2 c3 c4 c5 c6 c7 s1 s2 s3 s4 s5 s6 s7 M omega alpha kappa u A Omega U X Y""")


with open('pickled_files/pickled_elliptical_orbit_determinant_metric_tensor_fn.pkl', 'rb') as f:
    determinant_5D_lambdified = cloudpickle.load(f)

def det_sq_root(freq, angular_velocity, projected_radius, orbital_phase, longitude_periastron, eccentricity, obs_time, pi_value):

    ''' This calculation is done in dimensionless units. Hence, multiply with the scaling factor (2 * pi * f * T)**5 '''

    X_val =  projected_radius * np.sin(longitude_periastron)/obs_time
    Y_val =  projected_radius * np.cos(longitude_periastron)/obs_time
    omega_val = angular_velocity * obs_time
    determinant = determinant_5D_lambdified(X_val, Y_val, eccentricity, omega_val, orbital_phase, np.pi)
    determinant = abs(determinant)
    determinant = math.sqrt(determinant)
    determinant = determinant * ((2 * np.pi * freq * obs_time)**5)
    return determinant


def calculate_alpha(sini, max_companion_mass, min_pulsar_mass, current_candidate_mass, current_companion_mass):
    alpha = sini * max_companion_mass * ((current_candidate_mass + current_companion_mass)**(2/3))/(current_companion_mass * \
     (max_companion_mass + min_pulsar_mass)**(2/3))
    p = 1 - np.sqrt(1 - alpha**2)
    return p

def number_templates(dimension,coverage,mismatch,volume):
    n_dim_ball_volume = math.pow(np.pi,dimension/2)/math.gamma((dimension/2) + 1)
    N=math.log(1-coverage)/math.log(1-math.pow(mismatch,dimension/2) * n_dim_ball_volume/volume)
    return N



def log_posterior(theta, freq, obs_time, lowest_angular_velocity, highest_angular_velocity, max_initial_orbital_phase, max_longitude_periastron, max_eccentricity):

    ''' theta is a five-dimensional vector of our model holding all the orbital template parameters '''
    
    angular_velocity, projected_radius, orbital_phase, longitude_periastron, eccentricity = theta
    if not (lowest_angular_velocity < angular_velocity < highest_angular_velocity and 0. < orbital_phase < max_initial_orbital_phase and 0. < longitude_periastron < max_longitude_periastron and \

            0. < eccentricity < max_eccentricity):

        return -np.inf

    max_projected_radius = alpha * G**(1/3) * max_companion_mass * M_0 * angular_velocity**(-2/3)/(c * ((min_pulsar_mass + max_companion_mass) * M_0)**(2/3))
    if not (0. <= projected_radius <= max_projected_radius):
        return -np.inf
    determinant = det_sq_root(freq, angular_velocity, projected_radius, orbital_phase, longitude_periastron, eccentricity, obs_time, np.pi)

    if determinant == 0:
        return -np.inf

    if math.isnan(determinant):
        return -np.inf

    determinant = np.log(determinant)
    return determinant



total_templates_required = args.templates

''' Define your prior ranges'''

G = 6.67e-11
M_0 = 1.989e+30
c = 2.99792458e+08
pi_1 = np.pi
obs_time = args.obs_time * 60 
p_orb_upper_limit = args.porb_high * 60
p_orb_low_limit = args.porb_low * 60 
min_pulsar_mass = args.min_pulsar_mass
max_companion_mass = args.max_comp_mass
alpha = args.fraction
coverage = args.coverage
mismatch = args.mismatch
ncpus = args.ncpus
fastest_spin_period_ms = args.spin_period
spin_freq = 1/(fastest_spin_period_ms * 1e-03) 
max_initial_orbital_phase = 2 * np.pi
max_longitude_periastron = 2 * np.pi
max_eccentricity = args.max_eccentricity
output_path = args.output_path
lowest_angular_velocity = 2 * np.pi/p_orb_upper_limit
highest_angular_velocity = 2 * np.pi/p_orb_low_limit


ndim = 5
np.random.seed(42)
nwalkers = 800
burn_in_steps = 100
filename = output_path + args.filename + '.h5'
backend = emcee.backends.HDFBackend(filename)
angular_velocity_start_values = np.random.uniform(lowest_angular_velocity, highest_angular_velocity, nwalkers)
max_projected_radius = alpha * G**(1/3) * max_companion_mass * M_0 * highest_angular_velocity**(-2/3)/(c * ((min_pulsar_mass + max_companion_mass) * M_0)**(2/3))
projected_radius_start_values = np.random.uniform(0., max_projected_radius, nwalkers)
orbital_phase_start_values = np.random.uniform(0., max_initial_orbital_phase, nwalkers)
longitude_periastron_start_values = np.random.uniform(0., max_longitude_periastron, nwalkers)
eccentricity_start_values = np.random.uniform(0., max_eccentricity, nwalkers)

initial_guess = np.column_stack((angular_velocity_start_values, projected_radius_start_values, orbital_phase_start_values, longitude_periastron_start_values, eccentricity_start_values))


#Use processes if your dealing with processors in the same node with shared memory (OpenMP)
#with Pool(processes = ncpus) as pool: 

#Use MPIPool if your dealing with processors across multiple nodes
with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, backend=backend, args=[spin_freq, obs_time, lowest_angular_velocity, highest_angular_velocity,\
                                     max_initial_orbital_phase, max_longitude_periastron, max_eccentricity], pool=pool)
    start = time.time()
    state = sampler.run_mcmc(initial_guess, burn_in_steps)
    end = time.time()
    multi_time = end - start
    print("Burn-In Phase took {0:.1f} seconds".format(multi_time))
    sampler.reset()
    start = time.time()
    sampler.run_mcmc(state, total_templates_required)
    end = time.time()
    multi_time = end - start
    print("Main Phase took {0:.1f} seconds".format(multi_time))

