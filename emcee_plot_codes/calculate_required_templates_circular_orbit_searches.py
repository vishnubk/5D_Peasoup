
#!/usr/bin/env 
# -*- coding: utf-8 -*- Tested on python 2.7, python 3.3, python 3.6
# =============================================================================
# Created By  : Vishnu Balakrishnan
# Created Date: Mon June 4th 12:34:00 UTC 2018
# =============================================================================

"""This Module has been built to Calculate and Generate Random Templates based on Messenger et al. 2008 (arXiv:0809.5223). 
These templates are used to do a fully coherent search for circular binary orbits in radio observations. 

      Receipe to Generate Random Template Bank!

1. Initialise your signal phase model and calculate the metric tensor of your parameter space.

2. Compute determinant of this metric. This will be used later as a constant density function to distribute templates
   in your parameter space.

2. Compute proper volume/volume integral of your parameter space and calculate number of required templates 
   based on required coverage and mismatch.

4. For each proposal template, draw random values from angular velocity, 
   projected radius and orbital phase (parameters of interest).

5. Implement a MCMC based on metropolis hastings algorithm using square root of the determinant of the metric tensor as your constant density function. Write Results to file. 

6. This code has only been tested till python 3.6. One of its dependencies mcquad which does the MC integration is not supported in later python versions. I'll try to update this in due time.

"""

#Uncomment this if your in python 2.7
#from __future__ import division

import sys
import numpy as np
import math
import os
from skmonaco import mcquad
import sympy as sy
import argparse


parser = argparse.ArgumentParser(description='Calculate required templates for a coherent keplerian circular orbit search based on Messenger at al. 2008 and generate template-bank based on Metropolis hastings')
parser.add_argument('-o', '--output_path', help='Output path to save results',  default="generated_template_banks/")
parser.add_argument('-t', '--obs_time', help='Observation time in minutes', default='72', type=float)
parser.add_argument('-p', '--porb_low', help='Lower limit of Orbital Period in minutes', default='360', type=float)
parser.add_argument('-P', '--porb_high', help='Upper limit of Orbital Period in minutes', type=float)
parser.add_argument('-c', '--max_comp_mass', help='Maximum mass of Companion in solar mass units', default='8', type=float)
parser.add_argument('-d', '--min_pulsar_mass', help='Minimum mass of Pulsar in solar mass units', default='1.4', type=float)
parser.add_argument('-s', '--spin_period', help='Fastest spin period of pulsar in ms', default='5', type=float)
parser.add_argument('-f', '--fraction', help='Probability fraction of orbits of different inclination angles', default='1', type=float)
parser.add_argument('-b', '--coverage', help='Coverage of template-bank', default='0.9', type=float)
parser.add_argument('-m', '--mismatch', help='Mismatch of template-bank', default='0.2', type=float)
parser.add_argument('-n', '--ncpus', help='Number of CPUs to use for calculation', default='32', type=int)
parser.add_argument('-i', '--nmc', help='Number of iterations for monte-carlo integration', default='100000', type=int)
parser.add_argument('-file', '--output_filename', help='Filename for template-bank', type=str)

args = parser.parse_args()
args.porb_high = args.porb_high if args.porb_high else args.obs_time * 10	




#Define all variable for metric tensor calculations
output_path = args.output_path
f, tau, omega, psi, phi, t, T, a, pi, f0 = sy.symbols('f \\tau \\Omega \\psi \phi t T a \pi f_0')
sy.init_printing(use_unicode=True) #pretty printing

## Phase Model for Circular Binary Orbits
phi = 2 * pi * f * (t + tau * sy.sin(omega * t + psi))

def time_average(a):
    b = (1/T) * sy.integrate(a, (t, 0, T))
    return b

variables=[f, tau, omega, psi]

metric_tensor=np.empty(shape=(4,4), dtype=object)
for i in range(len(variables)):
    for j in range(len(variables)):
         metric_tensor[i][j]=(time_average(sy.diff(phi,variables[i]) * sy.diff(phi,variables[j])) - time_average(sy.diff(phi,variables[i])) * time_average(sy.diff(phi,variables[j])))

metric_tensor_w_f_row_column = metric_tensor[1:4,1:4]
variables=[tau, omega, psi]
metric_tensor3D=np.empty(shape=(3,3), dtype=object)
for i in range(len(variables)):
    for j in range(len(variables)):
        metric_tensor3D[i][j]=metric_tensor_w_f_row_column[i][j] * metric_tensor[0][0] - (metric_tensor[0][i+1] * \
                                                                                         metric_tensor[j+1][0])
metric_tensor3D=sy.Matrix(metric_tensor3D)

''' matrix.det() method in sympy does an in-built symplification in python3 which gives wrong results! If in python2.7, you can run metric_tensor3D.det(), however in python3 stick to this workaround by manually ggiving the formula for a determinant '''

A = sy.Matrix(3, 3, sy.symbols('A:3:3'))
det_metric_tensor3D = A.det().subs(zip(list(A), list(metric_tensor3D)))
det_metric_tensor3D=det_metric_tensor3D/metric_tensor[0][0]**3

#det_metric_tensor3D=metric_tensor3D.det()/metric_tensor[0][0]**3
expr=det_metric_tensor3D**0.5
expr_numpy = sy.lambdify([f, psi, omega, tau, T, pi], expr, "numpy")

def det_sq_root(angular_velocity, projected_radius, orbital_phase, freq, obs_time):
    
    return expr_numpy(freq, orbital_phase, angular_velocity, projected_radius, obs_time, np.pi)

def calculate_alpha(sini, max_companion_mass, min_pulsar_mass, current_pulsar_mass, current_companion_mass):
    ''' Basically, this checks if a certain mass range is covered by your asini trials
        Explanation:
              Say min_pulsar_mass = 1.4 M0, max_companion_mass = 8M0, if args.fraction = 1, then as long as current_pulsar_mass is above the minimum pulsar mass and  
              current_companion_mass is below the max_companion_mass, then probability of detection = 1. This probability assumes that the orbit is circular, 
              and your data is sensitive enough to find the pulsar for a given coverage and mismatch.
              
              If current_companion_mass > 8 M0 or current_pulsar_mass < 1.4 M0, then p < 1, and we only partially cover that parameter space based on our asini trials 
         '''

    alpha = sini * max_companion_mass * ((current_pulsar_mass + current_companion_mass)**(2/3))/(current_companion_mass * \
     (max_companion_mass + min_pulsar_mass)**(2/3))
    p = 1 - np.sqrt(1 - alpha**2)
    return p

def number_templates(dimension,coverage,mismatch,volume):
    ''' This calculates the required random templates based on volume, coverage and mismatch '''
    n_dim_ball_volume = math.pow(np.pi,dimension/2)/math.gamma((dimension/2) + 1)
    N=math.log(1-coverage)/math.log(1-math.pow(mismatch,dimension/2) * n_dim_ball_volume/volume)
    return N


#Templates For Vishnu targeted observation.
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
volume_integral_iterations = args.nmc
ncpus = args.ncpus
batch_size_integration = int(volume_integral_iterations/ncpus) # This is used to batch the integration and parallelise it in mcquad
coverage = args.coverage
mismatch = args.mismatch
fastest_spin_period_ms = args.spin_period
freq = 1/(fastest_spin_period_ms * 1e-03) 
max_initial_orbital_phase = 2 * np.pi
probability = calculate_alpha(alpha,max_companion_mass,min_pulsar_mass,min_pulsar_mass,max_companion_mass)
highest_angular_velocity = 2 * np.pi/p_orb_upper_limit
highest_limit_projected_radius = alpha * G**(1/3) * max_companion_mass * M_0 * highest_angular_velocity**(-2/3)/(c * ((min_pulsar_mass + max_companion_mass) * M_0)**(2/3))



def volume_integral(t, max_companion_mass, spin_freq, obs_time, min_pulsar_mass):
    angular_velocity, projected_radius, orbital_phase = t
    max_projected_radius = alpha * G**(1/3) * max_companion_mass * M_0 * angular_velocity**(-2/3)/(c * ((min_pulsar_mass + max_companion_mass) * M_0)**(2/3))
    if projected_radius <= max_projected_radius:

        return expr_numpy(spin_freq, orbital_phase, angular_velocity, projected_radius, obs_time, np.pi)

    return 0



# Define the limits of your coordinates here in xl and xu
volume_integral_result, estimated_volume_integral_error = mcquad(volume_integral,npoints = volume_integral_iterations, xl= [(2 * np.pi)/p_orb_upper_limit, 0., 0.], \
                                     xu=[(2 * np.pi)/p_orb_low_limit, highest_limit_projected_radius, max_initial_orbital_phase], nprocs=ncpus, batch_size=batch_size_integration, args=[max_companion_mass, freq, obs_time, min_pulsar_mass])

print('Volume Integral: ', volume_integral_result, 'Volume Integral Error: ', estimated_volume_integral_error)
print('Volume integral error is: %.2f' %((estimated_volume_integral_error/volume_integral_result) * 100), ' %')
total_templates_targed_search = number_templates(3,coverage,mismatch,np.around(volume_integral_result)) 
    
print('observation time (mins):', obs_time/60, 'mass companion:', max_companion_mass, 'orbital period low (hrs):', p_orb_low_limit/3600, 'orbital period high (hrs):', p_orb_upper_limit/3600, 'spin period (ms):', (1/freq) * 1e+3, 'prob:', \
 probability, 'templates: ', total_templates_targed_search, 'integration error percentage: ', (estimated_volume_integral_error/volume_integral_result) * 100, 'coverage: ', coverage, 'mismatch: ', mismatch, 'phase: ', max_initial_orbital_phase)

if not args.output_filename:
    sys.exit()

'''Implementation of a simple Independence chain Metropolis Hastings algorithm. You can read more about this algorithm in section 7.2.2 of the link given below   

       https://bookdown.org/rdpeng/advstatcomp/metropolis-hastings.html#independence-metropolis-algorithm           '''



burn_in_steps = 100
output_filename = args.output_filename
N = int(np.around(total_templates_targed_search + burn_in_steps))
angular_velocity_init = np.random.uniform((2 * np.pi)/p_orb_upper_limit,(2 * np.pi)/p_orb_low_limit)
max_projected_radius = alpha * G**(1/3) * max_companion_mass * M_0 * angular_velocity_init**(-2/3)/(c * \
((min_pulsar_mass + max_companion_mass) * M_0)**(2/3))
projected_radius_init = np.random.uniform(0,max_projected_radius)
orbital_phase_init = np.random.uniform(0, 2*np.pi)


p = det_sq_root(angular_velocity_init, projected_radius_init, orbital_phase_init, freq, obs_time)
counts = 0
ntrials = 0
accepted_templates = []
rejected_templates = []
while(counts!=N):
    angular_velocity = np.random.uniform((2 * np.pi)/p_orb_upper_limit,(2 * np.pi)/p_orb_low_limit)
    max_projected_radius = alpha * G**(1/3) * max_companion_mass * M_0 * angular_velocity**(-2/3)/(c * \
    ((min_pulsar_mass + max_companion_mass) * M_0)**(2/3))
    projected_radius = np.random.uniform(0,max_projected_radius)
    orbital_phase= np.random.uniform(0, 2*np.pi)
    
    p_prop = det_sq_root(angular_velocity, projected_radius, orbital_phase, freq, obs_time)
    u = np.random.rand()
    if u < p_prop/p:
        p = p_prop
        counts+=1
        accepted_templates.append([angular_velocity, projected_radius, orbital_phase,p])
        
    ntrials+=1

accepted_templates = np.asarray(accepted_templates)
accepted_templates = accepted_templates[burn_in_steps:]


''' The file below creates a header file if you would like to check the parameters of your search later '''

if not os.path.exists(output_path + output_filename + '_circular_orbit_header.txt'):
    with open(output_path + output_filename + '_circular_orbit_header.txt', 'w') as outfile:
             outfile.write('observation time (mins): ' + str(obs_time/60) + ',' + 'orbital period low (hrs): ' + str(p_orb_low_limit/3600) + ',' + 'orbital period high (hrs): ' + str(p_orb_upper_limit/3600) + ',' + 'spin period (ms): ' + str((1/freq) * 1e+3) + ',' + 'fraction: ' + str(alpha) + ',' + 'prob: ' + str(probability) + ',' + 'templates: ' + str(total_templates_targed_search) + ',' + 'integration error percentage: ' + str((estimated_volume_integral_error/volume_integral_result) * 100) + ',' + 'coverage: ' + str(coverage) + ',' + 'mismatch: ' + str(mismatch) + ',' + 'phase: ' + str(max_initial_orbital_phase) + ',' + 'mass companion: ' + str(max_companion_mass) + '\n')


''' Use the file outputted below as an input to the GPU Pipeline '''
with open(output_path + output_filename + '_circular_orbit_gpu_format.txt', 'a') as outfile:
    for i in range(len(accepted_templates)):
        outfile.write(str(accepted_templates[i][0]) + ' ' + str(accepted_templates[i][1]) + ' ' + str(accepted_templates[i][2]) + ' ' + '\n')
    
    ''' Adding an extra template in the end with asini/c=0, to cover isolated pulsars! ''' 
    outfile.write(str(accepted_templates[0][0]) + ' ' + str(0.0) + ' ' + str(accepted_templates[0][2]) + ' ' + '\n')




  
