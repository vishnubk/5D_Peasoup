#include </fred/oz002/vishnu/software/template_bank_gpu/include/data_types/filterbank.hpp>
#include <iostream>
#include <stdexcept>
#include <assert.h>

using namespace std;

int main(void){
  
  std::string filename("/fred/oz002/vishnu/LOWLAT/corrected_filterbank_files/2012-09-26-07:30:59/01/2012-09-26-07:30:59_downsampled.fil");
  try {
    SigprocFilterbank filobj(filename);
  } catch (std::runtime_error& e) {
    std::cout << e.what() << std::endl;
  }

  unsigned char* data_ptr;
  unsigned int nsamps = 17000000;
  unsigned int nchans  = 1024;
  unsigned char nbits = 2;
  float fch1 = 1581.804688;
  float foff = -0.390625;
  float tsamp = 0.000256;
  float tstart = 56196.313182870370;
  SigprocFilterbank filobj(data_ptr,nsamps,nchans,nbits,fch1,foff,tsamp,tstart);
  
  assert(nsamps==filobj.get_nsamps());
  assert(nchans==filobj.get_nchans());
  assert(tsamp==filobj.get_tsamp());
  assert(nbits==filobj.get_nbits());
  assert(foff==filobj.get_foff());
  assert(fch1==filobj.get_fch1());
  assert(tstart==filobj.get_tstart());
  assert(data_ptr==filobj.get_data());
  
  unsigned char* new_data_ptr;
  unsigned int new_nsamps = 100000+1;
  unsigned int new_nchans  = 1024+1;
  unsigned char new_nbits = 2+1;
  float new_fch1 = 1560.0+1;
  float new_foff = 0.39+1;
  float new_tsamp = 0.000054+1;
  float tstart = 56196.313182870370+1;
  filobj.set_nsamps(new_nsamps);
  filobj.set_nchans(new_nchans);
  filobj.set_tsamp(new_tsamp);
  filobj.set_data(new_data_ptr);
  filobj.set_nbits(new_nbits);
  filobj.set_foff(new_foff);
  filobj.set_fch1(new_fch1);
  filobj.set_start(new_tstart);
  
  assert(new_nsamps==filobj.get_nsamps());
  assert(new_nchans==filobj.get_nchans());
  assert(new_tsamp==filobj.get_tsamp());
  assert(new_nbits==filobj.get_nbits());
  assert(new_foff==filobj.get_foff());
  assert(new_fch1==filobj.get_fch1());
  assert(new_data_ptr==filobj.get_data());
  
  return 0;
}
