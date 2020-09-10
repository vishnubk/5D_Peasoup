# This is where common definitions go

INSTALL_DIR = /fred/oz002/vishnu/32bit_installed_softwares

#cuda setup
CUDA_DIR   = /apps/skylake/software/core/cuda/9.0.176

#Thrust 1.7.1 or higher
THRUST_DIR = /fred/oz002/psrhome/opt/thrust-1.7

# dedisp setup
DEDISP_DIR = /fred/oz002/vishnu/software_bck/dedisp

GCC       = gcc
GXX       = g++
AR        = ar
NVCC      = $(CUDA_DIR)/bin/nvcc
SHELL     = /bin/bash
UCFLAGS   = -DUSE_NVTX
