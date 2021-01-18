#pragma once
#include <data_types/timeseries.hpp>
#include <kernels/kernels.h>
#include <kernels/defaults.h>
#include <utils/exceptions.hpp>

class TimeDomainResampler {
private:
  unsigned int max_threads;
  unsigned int max_blocks;
  
public:
  TimeDomainResampler(unsigned int max_threads=MAX_THREADS, unsigned int max_blocks=MAX_BLOCKS)
    :max_threads(max_threads),max_blocks(max_blocks)    
  {
  }
  
  //Force float until the kernel gets templated
  void resample(DeviceTimeSeries<float>& input, DeviceTimeSeries<float>& output, 
		unsigned int size, float acc)
  {
    device_resample(input.get_data(), output.get_data(), size,
		    acc, input.get_tsamp(),max_threads,  max_blocks);
  }

  void resampleII(DeviceTimeSeries<float>& input, DeviceTimeSeries<float>& output,
                unsigned int size, float acc)
  {
    device_resampleII(input.get_data(), output.get_data(), size,
                    acc, input.get_tsamp(),max_threads,  max_blocks);
  }


  void binary_timeseries_offset(DeviceTimeSeries<float>& input, DeviceTimeSeries<float>& output,
                     unsigned int size, double omega, double tau, double phi)
  { 
    double inverse_tsamp = 1/input.get_tsamp();
    //printf("Inverse_tsamp: %.6f, tsamp: %.6f \n", inverse_tsamp, input.get_tsamp());
    device_timeseries_offset(input.get_data(), output.get_data(), size, omega, tau, phi, 1/input.get_tsamp(), input.get_tsamp(), max_threads, max_blocks);
  }

  void binary_modulate_time_series_length(DeviceTimeSeries<float>& input, unsigned int  nsamples_unpadded, unsigned int* new_length)

   {
    //unsigned new_length = nsamples_unpadded -1;
    device_modulate_time_series_length(input.get_data(), nsamples_unpadded, new_length);
  }

  void new_binary_modulate_time_series_length(DeviceTimeSeries<float>& input, unsigned int  nsamples_unpadded, unsigned int* new_length)

   {
    //unsigned new_length = nsamples_unpadded -1;
    device_new_modulate_time_series_length(input.get_data(), nsamples_unpadded, new_length);
  }

  void binary_resample_circular_binary(DeviceTimeSeries<float>& input, DeviceTimeSeries<float>& output, DeviceTimeSeries<float>& offset,
                     unsigned int new_length)
  
  {
    device_resample_circular_binary(input.get_data(), output.get_data(), offset.get_data(), new_length, max_threads,  max_blocks);
  }

  void resampler_3D_circular_orbit_large_timeseries(DeviceTimeSeries<float>& input, DeviceTimeSeries<float>& output, unsigned int size, double omega, double tau, double phi)
  {
    double inverse_tsamp = 1/input.get_tsamp();
    double zero_offset = tau * sin(phi) * inverse_tsamp;
    //printf("Inverse_tsamp: %.6f, zero_offset: %.6f, Size: %d \n", inverse_tsamp, zero_offset, size);
    device_resampler_circular_binary_large_timeseries(input.get_data(), output.get_data(), omega, tau, phi, zero_offset, 1/input.get_tsamp(), input.get_tsamp(), size, max_threads, max_blocks);
  }

  void remove_roemer_delay(double* start_timeseries_array, double* roemer_delay_removed_timeseries_array, unsigned int size,\
   double omega, double tau, double phi_normalised, double long_periastron, double eccentricity, double sampling_time)

   {
   
   device_remove_roemer_delay_elliptical(start_timeseries_array, roemer_delay_removed_timeseries_array, omega, tau, phi_normalised, long_periastron, eccentricity, sampling_time, size, max_threads, max_blocks);
   }

    void resample_using_1D_lerp(double* roemer_delay_removed_timeseries_array, DeviceTimeSeries<float>& input,  unsigned long xp_len, 
    unsigned long x_len, double* output_samples_array, DeviceTimeSeries<float>& output)

   {
   
   device_resample_using_1D_lerp(roemer_delay_removed_timeseries_array, input.get_data(), xp_len, x_len, output_samples_array, output.get_data(), max_threads, max_blocks);
   }

   void fast_ellitpical_orbit_resampler_large_timeseries(DeviceTimeSeries<float>& input, DeviceTimeSeries<float>& output, unsigned long size, double omega, double tau, double phi, double long_periastron, double eccentricity, double sampling_time, double inverse_tsamp)
  {
   device_get_barycentered_timeseries_elliptical_orbits(input.get_data(), output.get_data(), size, omega, tau, phi, long_periastron, eccentricity, inverse_tsamp, sampling_time, max_threads, max_blocks);
}
};

