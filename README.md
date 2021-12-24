# 5D Peasoup

## General Description 
C++ Cuda GPU Pulsar Search Pipeline using the template-bank algorithm. This pipeline was built on the original code written almost entirely by Ewan Barr (MPIfR). You can find the original repo here that does an 1-D acceleration search. https://github.com/ewanbarr/peasoup

This pipeline does a coherent search for circular and elliptical binary orbits (3 or 5 binary parameters) using the Random/Stochastic Template-Bank algorithm. You can read more about this algorithm in Messenger (2008), Harry (2009), Knispel (2011) and Allen (2013).

## How to install this GPU pipeline?

1. Install Ben Barsdell's GPU De-dedispersion library "dedisp" that does an incoherent dedispersion from this repo. https://github.com/ewanbarr/dedisp.git (Edit Makefile.inc and point to your CUDA installation)
2. Install the software from "fast_bt_resampler_working" branch of 5D Peasoup. (Edit Makefile.inc and point to your CUDA and Dedisp installation)

## How to Calculate Required amount of Orbital templates for a Circular Orbit Search.

1. Run the code emcee_plot_codes/calculate_circular_orbit_search_template_distribution.py . Tested on python2.7 and python3.3. This will calculate the total orbital templates you need for a circular orbit search and print to screen.

2. Use the -file option if you would like to generate a template-bank after the required templates is calculated. This is done using the Metropolis Hastings algorithm. Outputs two files. One is a header with your parameter search ranges, and the second file with the suffix *gpu_format needs to be given as input to the GPU pipeline for processing.

3. Alternatively you can also use emcee which uses Goodman & Weare’s Affine Invariant MCMC technique to create a template-bank and plot corner plots for your template density distribution. Check emcee_plot_codes/calculate_circular_orbit_search_template_distribution.py and emcee_plot_codes/make_corner_plots_from_emcee_simulations.py respectively for that. If your going down this route, follow step 1 and whatever total number of templates it outputs (for e.g. 1000), pass that value to --templates.

4. If your only interested to do circular orbit searches, I recommend installing the software from this repo https://github.com/vishnubk/3D_peasoup . 
The installation instructions are identical. This has a faster implementation of the time-domain resampler. It avoids the while loop used to solve Kepler's equation iteratively which is required only if the orbit is elliptical. Combining both these repos into one is a work in progress.

## Calculating template density for circular and elliptical orbit searches using Emcee
Codes to calculate template density using emcee can be found in the emcee_plot_codes directory

## Paper

https://arxiv.org/abs/2112.11991?context=astro-ph
