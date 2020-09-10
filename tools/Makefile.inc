# This is where common definitions go

INSTALL_DIR = /p/scratch/cpadc/vishnu/debug_software

#cuda setup
CUDA_DIR   = /gpfs/software/opt/cuda/8.0.61

#Thrust 1.7.1 or higher
THRUST_DIR = /usr/local/cuda/

# dedisp setup
DEDISP_DIR = /p/scratch/cpadc/vishnu/softwares

GCC       = gcc
GXX       = g++
AR        = ar
NVCC      = $(CUDA_DIR)/bin/nvcc
SHELL     = /bin/bash
UCFLAGS   = -DUSE_NVTX
