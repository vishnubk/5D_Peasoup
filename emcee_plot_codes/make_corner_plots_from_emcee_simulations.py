import emcee
import numpy as np
import cornerimport emcee
import numpy as np
import corner

# Corner Plot for Circular orbit Search


sampler = emcee.backends.HDFBackend("circular_template_bank_15ms_paper_scott_6hr_orbit_final.h5", read_only=True)

ndim = 3
tau = sampler.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)

# Emcee requires log posterior in natural log units, changed it to base 10 here

log_prob_samples = np.log10(np.exp(sampler.get_log_prob(discard=burnin, flat=True, thin=thin)))
samples[:,0] = (2 * np.pi/samples[:,0])/3600 # Convert angular velocity to porb in hours.
samples[:,2] = np.degrees(samples[:,2]) # Orbital Phase in Degrees




all_samples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)
labels=["Orbital Period \n (hrs)", "Projected Radius \n (lt-s)", "Orbital Phase \n (degrees)"]
labels += [r"log$_{10}$ $\left(\sqrt{|det(\gamma_{\alpha \beta})|}\right)$"]

figure = corner.corner(all_samples, labels=labels, color='black', title_kwargs={"fontsize": 12}, \
                       smooth=True, smooth1d=True, scale_hist=True, levels=(0.1175031 , 0.39346934, 0.67534753, 0.86466472), \
                      );

figure.savefig('templates_density_plots_paper/determinant_scaling_circular_orbit_search_15ms.pdf')


# Corner Plot for Elliptical orbit Search


sampler = emcee.backends.HDFBackend("ellitpical_template_bank_15ms_paper_scott_scaling_six_hour_orbit_fast.h5", read_only=True)

ndim = 5
tau = sampler.get_autocorr_time()
burnin = int(2 * np.max(tau)) 
thin = int(0.5 * np.min(tau)) 
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = np.log10(np.exp(sampler.get_log_prob(discard=burnin, flat=True, thin=thin)))

samples[:,0] = (2 * np.pi/samples[:,0])/3600 # Convert angular velocity to porb in hours.
samples[:,2] = np.degrees(samples[:,2])  # Orbital Phase in Degrees
samples[:,3] = np.degrees(samples[:,3])  # Longitude of periastron in degrees



all_samples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)

labels=["Orbital Period \n (hrs)", "Projected Radius \n (lt-s)", "Orbital Phase \n (degrees)", \
        "Longitude of Periastron \n (degrees)" , "Eccentricity"]
labels += [r"log$_{10}$ $\left(\sqrt{|det(\gamma_{\alpha \beta})|}\right)$"]

figure = corner.corner(all_samples, labels=labels, color='black', \
                       title_kwargs={"fontsize": 12}, smooth=True, smooth1d=True, plot_contours=True,\
                      max_n_ticks=6, scale_hist=True, levels=(0.1175031 , 0.39346934, 0.67534753, 0.86466472));
figure.savefig('templates_density_plots_paper/elliptical_orbit_determinant_scaling_ecc_0_9.pdf')

