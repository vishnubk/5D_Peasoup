#pragma once
#include "data_types/fourierseries.hpp"
#include "kernels/kernels.h"
#include "kernels/defaults.h"
#include "utils/utils.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <math.h>
class Template_Bank {
private:
  bool d_mem_allocated;
  std::vector<double> angular_velocity; // Angular Velocity = 2pi/orbital period
  std::vector<double> tau;              // asini in light-seconds
  std::vector<double> phi;              // initial orbital phase (0,2pi)
  std::vector<double> long_periastron;              // Longtitude of Periastron (0,2pi)
  std::vector<double> eccentricity;              // Eccentricity (0,1)

std::vector<std::string> split(std::string const &input) {
    std::stringstream buffer(input);
    std::vector<std::string> ret;
    std::copy(std::istream_iterator<std::string>(buffer),
              std::istream_iterator<std::string>(),
              std::back_inserter(ret));
    return ret;
  }
  

public:
  Template_Bank(std::string template_bank_list)
  {
    d_mem_allocated = false;
    set_template_bank_file(template_bank_list);
  }
 
 
  void set_template_bank_file(std::string template_bank_list){
    std::string line;
    std::ifstream infile(template_bank_list.c_str());
    ErrorChecker::check_file_error(infile, template_bank_list);
    while (std::getline(infile,line)){
      std::vector<std::string> split_line = split(line);
      if (split_line.size()>0){
	angular_velocity.push_back(::atof(split_line[0].c_str()));
	tau.push_back(::atof(split_line[1].c_str()));
	phi.push_back(::atof(split_line[2].c_str()));
    long_periastron.push_back(::atof(split_line[3].c_str()));
    eccentricity.push_back(::atof(split_line[4].c_str()));
      
      }
    }
  
    infile.close();
  }


   std::vector<double> get_angular_velocity(void){
    //angular_velocity = 2.0f * M_PI/(angular_velocity * 3600.0f);
    return angular_velocity;
      }

   std::vector<double> get_tau(void){
    return tau;
     }

    std::vector<double> get_phi(void){
     //phi = phi * 2.0f * M_PI;
     return phi;
      }

     std::vector<double> get_long_periastron(void){
     //long_periastron = long_periastron * M_PI/180.0f;
     return long_periastron;
      }

     std::vector<double> get_eccentricity(void){
     return eccentricity;
      }

  //void zap(DeviceFourierSeries<cufftComplex>& fseries){
  //  float bin_width = fseries.get_bin_width();
  //  unsigned int nbins = fseries.get_nbins();
  //  zap(fseries.get_data(),bin_width,nbins);
  //}
  //
  //void zap(cufftComplex* fseries, float bin_width, unsigned int nbins){
  //  device_zap_birdies(fseries, d_birdies, d_widths,
  //                     bin_width, birdies.size(), nbins,
  //                     MAX_BLOCKS, MAX_THREADS);
  //}
    
};
