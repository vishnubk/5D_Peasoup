#!/usr/bin/env
# -*- coding: utf-8 -*- Tested on python 2.7, python 3.3 and python 3.6
# =============================================================================
# Created By  : Vishnu Balakrishnan
# Created Date: Mon June 8th 12:34:00 UTC 2020
# =============================================================================

"""This Module has been built to Calculate and Generate Random Templates based on Messenger et al. 2008 (arXiv:0809.5223).
These templates are used to do a fully coherent search for elliptical binary orbits in radio observations.

      Receipe to Calculate Required Templates for your elliptical-orbit search!


1. Compute determinant of the metric tensor. The vectorised metric tensor function has been pickled to save compute time and avoid re-doing the calculation. 
    This will be used later by emcee as a constant density function to distribute templates in your parameter space.

2. Compute proper volume/volume integral of your parameter space and calculate number of required templates
   based on required coverage and mismatch.

3. Generating the elliptical-orbit template-bank using Metropolis Hastings is too slow, hence pass the total templates value you get from running this code to calculate_elliptical_orbit_search_template_distribution.py as --templates. 

4. This code has only been tested till python 3.6. One of its dependencies mcquad which does the MC integration is not supported in later python versions. I'll try to update this in due time.

"""


import numpy as np
import sympy as sy
sy.init_printing(use_unicode=True)
import cloudpickle
import sys, os
import math
from skmonaco import mcquad
import argparse

parser = argparse.ArgumentParser(description='Calculate required templates for a coherent full keplerian elliptical orbit search')
parser.add_argument('-o', '--output_path', help='Output path to save results',  default="generated_template_banks/")
parser.add_argument('-t', '--obs_time', help='Observation time in minutes', default='72', type=float)
parser.add_argument('-p', '--porb_low', help='Lower limit of Orbital Period in minutes', default='360', type=float)
parser.add_argument('-P', '--porb_high', help='Upper limit of Orbital Period in minutes', type=float)
parser.add_argument('-c', '--max_comp_mass', help='Maximum mass of Companion in solar mass units', default='8', type=float)
parser.add_argument('-d', '--min_pulsar_mass', help='Minimum mass of Pulsar in solar mass units', default='1.4', type=float)
parser.add_argument('-s', '--spin_period', help='Fastest spin period of pulsar in ms', default='5', type=float)
parser.add_argument('-e', '--max_eccentricity', help='Upper limit of eccentricity', default='0.5', type=float)
parser.add_argument('-emin', '--min_eccentricity', help='Lower limit of eccentricity', default='0.0', type=float)
parser.add_argument('-f', '--fraction', help='Probability fraction of orbits of different inclination angles', default='1', type=float)
parser.add_argument('-b', '--coverage', help='Coverage of template-bank', default='0.9', type=float)
parser.add_argument('-perimin', '--min_periastron', help='Minimum value of Longitude of Periastron in degrees', default='0.0', type=float)
parser.add_argument('-perimax', '--max_periastron', help='Maximum value of Longitude of Periastron in degrees', default='360', type=float)
parser.add_argument('-m', '--mismatch', help='Mismatch of template-bank', default='0.2', type=float)
parser.add_argument('-n', '--ncpus', help='Number of CPUs to use for calculation', default='48', type=int)
parser.add_argument('-i', '--nmc', help='Number of iterations for monte-carlo integration', default='100000', type=int)
parser.add_argument('-file', '--output_filename', help='Filename for template-bank header', type=str)

args = parser.parse_args()
args.porb_high = args.porb_high if args.porb_high else args.obs_time * 10
phi, pi, T, f0, f, tau, psi, E, cosE, sinE, t, e, c0, c1, c2, c3, c4, c5, c6, c7, s1, s2, s3, s4, s5, s6, s7, M, omega, alpha, kappa, u, A, Omega, U, X, Y = sy.symbols(""" phi
pi T f0 f tau psi E cosE sinE t e c0 c1 c2 c3 c4 c5 c6 c7 s1 s2 s3 s4 s5 s6 s7 M omega alpha kappa u A Omega U X Y""")


with open('pickled_files/pickled_elliptical_orbit_determinant_metric_tensor_fn.pkl', 'rb') as f:
    determinant_5D_lambdified = cloudpickle.load(f)



def det_sq_root(freq, angular_velocity, projected_radius, orbital_phase, longitude_periastron, eccentricity, obs_time, pi_value):

    #X_val =  projected_radius * np.cos(longitude_periastron)/obs_time
    X_val =  projected_radius * np.sin(longitude_periastron)/obs_time
    #Y_val =  projected_radius * np.sin(longitude_periastron)/obs_time
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


G = 6.67e-11
M_0 = 1.989e+30
c = 2.99792458e+08
pi_1 = np.pi
obs_time = args.obs_time * 60 # 18 mins
p_orb_upper_limit = args.porb_high * 60
p_orb_low_limit = args.porb_low * 60 #1.5 hours
min_pulsar_mass = args.min_pulsar_mass
max_companion_mass = args.max_comp_mass
alpha = args.fraction
volume_integral_iterations = args.nmc
coverage = args.coverage
mismatch = args.mismatch
ncpus = args.ncpus
batch_size_integration = int(volume_integral_iterations/ncpus)
fastest_spin_period_ms = args.spin_period
freq = 1/(fastest_spin_period_ms * 1e-03) # 5ms
max_initial_orbital_phase = 2 * np.pi
max_longitude_periastron = math.radians(args.max_periastron)
min_longitude_periastron = math.radians(args.min_periastron)
max_eccentricity = args.max_eccentricity
min_eccentricity = args.min_eccentricity
output_path = args.output_path
output_filename = args.output_filename
probability = calculate_alpha(alpha,max_companion_mass,min_pulsar_mass,min_pulsar_mass,max_companion_mass)
highest_angular_velocity = 2 * np.pi/p_orb_upper_limit
highest_limit_projected_radius = alpha * G**(1/3) * max_companion_mass * M_0 * highest_angular_velocity**(-2/3)/(c * \
    ((min_pulsar_mass + max_companion_mass) * M_0)**(2/3))

def volume_integral(t):

    angular_velocity, projected_radius, orbital_phase, longitude_periastron, eccentricity = t
    max_projected_radius = alpha * G**(1/3) * max_companion_mass * M_0 * angular_velocity**(-2/3)/(c * \
    ((min_pulsar_mass + max_companion_mass) * M_0)**(2/3))
    if projected_radius <= max_projected_radius:
        result = det_sq_root(freq, angular_velocity, projected_radius, orbital_phase, longitude_periastron, eccentricity, obs_time, np.pi)
        return result

    return 0


# Define the limits of your coordinates here in xl and xu
volume_integral_result, estimated_volume_integral_error = mcquad(volume_integral,npoints = volume_integral_iterations,xl=[(2 * np.pi)/p_orb_upper_limit, 0., 0., min_longitude_periastron, min_eccentricity], \
                                     xu=[(2 * np.pi)/p_orb_low_limit, highest_limit_projected_radius, max_initial_orbital_phase, max_longitude_periastron, max_eccentricity],nprocs=ncpus, batch_size=batch_size_integration)

print('Volume Integral: ', volume_integral_result, 'Volume Integral Error: ', estimated_volume_integral_error)
print('Volume integral error is: %.2f' %((estimated_volume_integral_error/volume_integral_result) * 100), ' %')
total_templates_required = number_templates(5,coverage,mismatch,np.around(volume_integral_result))


print('observation time (mins):', obs_time/60, 'mass companion:', max_companion_mass, 'asini_max:', highest_limit_projected_radius, 'orbital period low (hrs):', p_orb_low_limit/3600, 'orbital period high (hrs):', p_orb_upper_limit/3600, 'spin period (ms):', (1/freq) * 1e+3, 'prob:', \
 probability, 'templates: ', total_templates_required, 'integration error percentage: ', (estimated_volume_integral_error/volume_integral_result) * 100, 'coverage: ', coverage, 'mismatch: ', mismatch, 'phase: ', max_initial_orbital_phase)

if not args.output_filename:
    sys.exit()


''' The file below creates a header file if you would like to check the parameters of your search later '''

if not os.path.exists(output_path + output_filename + '_elliptical_orbit_template_bank_header.txt'):
    with open(output_path + output_filename + '_elliptical_orbit_template_bank_header.txt', 'w') as outfile:
             outfile.write('observation time (mins): ' + str(obs_time/60) + ',' + 'orbital period low (hrs): ' + str(p_orb_low_limit/3600) + ',' + 'orbital period high (hrs): ' + str(p_orb_upper_limit/3600) + ',' + 'spin period (ms): ' + str(fastest_spin_period_ms) + ',' + 'fraction: ' + str(alpha) + ',' + 'templates: ' + str(total_templates_required) + ',' + 'integration error percentage: ' + str((estimated_volume_integral_error/volume_integral_result) * 100) + ',' + 'coverage: ' + str(coverage) + ',' + 'mismatch: ' + str(mismatch) + ',' + 'phase: ' + str(max_initial_orbital_phase) + ',' + 'mass companion: ' + str(max_companion_mass) + ',' + 'mass pulsar: ' + str(min_pulsar_mass) + ',' + 'Prob: ' + str(probability) + ',' + 'Max_Periastron: ' + str(max_longitude_periastron) + ',' + 'Max_Ecc: ' + str(max_eccentricity) + '\n')





