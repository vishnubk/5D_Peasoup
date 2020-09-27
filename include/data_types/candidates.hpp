#pragma once
#include <iostream>
#include <vector>
#include <sstream>
#include "stdio.h"

//Lightweight plain old data struct
//used to combine multiple writes into
//one big write
struct CandidatePOD {
  float dm;
  int dm_idx;
  float acc;
  int nh;
  float snr;
  float freq;
};

struct Candidate {
public:
  float dm;
  int dm_idx;
  float acc;
  int nh;
  float snr;
  float freq;
  float folded_snr;
  double opt_period;
  bool is_adjacent;
  bool is_physical;
  float ddm_count_ratio;
  float ddm_snr_ratio;
  std::vector<Candidate> assoc;
  std::vector<float> fold;
  int nbins;
  int nints;
  
  void append(Candidate& other){
    assoc.push_back(other);
  }

  int count_assoc(void){
    int count = 0;
    for (int ii=0;ii<assoc.size();ii++){
      count ++;
      count += assoc[ii].count_assoc();
    }
    return count;
  }
  
  Candidate(float dm, int dm_idx, float acc, int nh, float snr, float freq)
    :dm(dm),dm_idx(dm_idx),acc(acc),nh(nh),
     snr(snr),folded_snr(0.0),freq(freq),
     opt_period(0.0),is_adjacent(false),is_physical(false),
     ddm_count_ratio(0.0),ddm_snr_ratio(0.0),nints(0),nbins(0){}
  
  Candidate(float dm, int dm_idx, float acc, int nh, float snr, float folded_snr, float freq)
    :dm(dm),dm_idx(dm_idx),acc(acc),nh(nh),snr(snr),
     folded_snr(folded_snr),freq(freq),opt_period(0.0),
     is_adjacent(false),is_physical(false),
     ddm_count_ratio(0.0),ddm_snr_ratio(0.0),nints(0),nbins(0){}

  Candidate()
    :dm(0.0),dm_idx(0.0),acc(0.0),nh(0.0),snr(0.0),
     folded_snr(0.0),freq(0.0),opt_period(0.0),
     is_adjacent(false),is_physical(false),
     ddm_count_ratio(0.0),ddm_snr_ratio(0.0),nints(0),nbins(0){}

  void set_fold(float* ar, int nbins, int nints){
    int size = nbins*nints;
    this->nints = nints;
    this->nbins = nbins;
    fold.resize(size);
    for (int ii=0;ii<size;ii++)
      fold[ii] = ar[ii];
  }
  
  void collect_candidates(std::vector<CandidatePOD>& cands_lite){
    CandidatePOD cand_stats = {dm,dm_idx,acc,nh,snr,freq};
    cands_lite.push_back(cand_stats);    
    for (int ii=0;ii<assoc.size();ii++){
      assoc[ii].collect_candidates(cands_lite);
    }
  }

  void print(FILE* fo=stdout){
    fprintf(fo,"%.15f\t%.15f\t%.15f\t%.2f\t%.2f\t%d\t%.1f\t%.1f\t%d\t%d\t%.4f\t%.4f\t%d\n",
	    1.0/freq,opt_period,freq,dm,acc,
	    nh,snr,folded_snr,is_adjacent,
	    is_physical,ddm_count_ratio,
	    ddm_snr_ratio,assoc.size());
    for (int ii=0;ii<assoc.size();ii++){
      assoc[ii].print(fo);
    }
  }

};



struct CandidatePOD_template_bank {
  float dm;
  int dm_idx;
  float omega;
  float tau;
  float phi;
  float long_periastron;
  float eccentricity;
  int nh;
  float snr;
  float freq;
};

struct Candidate_template_bank {
public:
  float dm;
  int dm_idx;
  float omega;
  float tau;
  float phi;
  float long_periastron;
  float eccentricity;
  int nh;
  float snr;
  float freq;
  float folded_snr;
  double opt_period;
  bool is_adjacent;
  bool is_physical;
  float ddm_count_ratio;
  float ddm_snr_ratio;
  std::vector<Candidate_template_bank> assoc;
  std::vector<float> fold;
  int nbins;
  int nints;
  
  void append(Candidate_template_bank& other){
    assoc.push_back(other);
  }

  int count_assoc(void){
    int count = 0;
    for (int ii=0;ii<assoc.size();ii++){
      count ++;
      count += assoc[ii].count_assoc();
    }
    return count;
  }
  
  Candidate_template_bank(float dm, int dm_idx, float omega, float tau, float phi, float long_periastron,
  float eccentricity, int nh, float snr, float freq)
    :dm(dm),dm_idx(dm_idx),omega(omega),tau(tau),phi(phi),long_periastron(long_periastron),
    eccentricity(eccentricity), nh(nh), snr(snr),folded_snr(0.0),freq(freq),
     opt_period(0.0),is_adjacent(false),is_physical(false),
     ddm_count_ratio(0.0),ddm_snr_ratio(0.0),nints(0),nbins(0){}
  
  Candidate_template_bank(float dm, int dm_idx, float omega, float tau, float phi, float long_periastron, float eccentricity, int nh, float snr, float folded_snr, float freq)
    :dm(dm),dm_idx(dm_idx),omega(omega),tau(tau),phi(phi),long_periastron(long_periastron),
    eccentricity(eccentricity),nh(nh),snr(snr), folded_snr(folded_snr),
    freq(freq),opt_period(0.0),is_adjacent(false),is_physical(false),
    ddm_count_ratio(0.0),ddm_snr_ratio(0.0),nints(0),nbins(0){}

  Candidate_template_bank()
    :dm(0.0),dm_idx(0.0),omega(0.0),tau(0.0),phi(0.0),long_periastron(0.0),eccentricity(0.0),
     nh(0.0),snr(0.0),folded_snr(0.0),freq(0.0),opt_period(0.0),
     is_adjacent(false),is_physical(false),
     ddm_count_ratio(0.0),ddm_snr_ratio(0.0),nints(0),nbins(0){}

  void set_fold(float* ar, int nbins, int nints){
    int size = nbins*nints;
    this->nints = nints;
    this->nbins = nbins;
    fold.resize(size);
    for (int ii=0;ii<size;ii++)
      fold[ii] = ar[ii];
  }
  
  void collect_candidates(std::vector<CandidatePOD_template_bank>& cands_lite){
    CandidatePOD_template_bank cand_stats = {dm,dm_idx,omega,tau,phi,long_periastron,eccentricity,nh,snr,freq};
    cands_lite.push_back(cand_stats);    
    for (int ii=0;ii<assoc.size();ii++){
      assoc[ii].collect_candidates(cands_lite);
    }
  }

  void print(FILE* fo=stdout){
    fprintf(fo,"%.15f\t%.15f\t%.15f\t%.2f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%d\t%.1f\t%.1f\t%d\t%d\t%.4f\t%.4f\t%d\n",
	    1.0/freq,opt_period,freq,dm,omega,tau,phi,long_periastron,eccentricity,
	    nh,snr,folded_snr,is_adjacent,
	    is_physical,ddm_count_ratio,
	    ddm_snr_ratio,assoc.size());
    for (int ii=0;ii<assoc.size();ii++){
      assoc[ii].print(fo);
    }
  }

};



class CandidateCollection {
public:
  std::vector<Candidate> cands;

  CandidateCollection(){}
  
  void append(CandidateCollection& other){
    cands.insert(cands.end(),other.cands.begin(),other.cands.end());
  }
  
  void append(std::vector<Candidate> other){
    cands.insert(cands.end(),other.begin(),other.end());
  }

  void reset(void){
    cands.clear();
  }

  void print(FILE* fo=stdout){
    for (int ii=0;ii<cands.size();ii++)
      cands[ii].print(fo);
  }

  void generate_candidate_binaries(std::string output_directory="./") {
    char filename[80];    
    std::stringstream filepath;
    for (int ii=0;ii<cands.size();ii++){
      filepath.str("");
      sprintf(filename,"cand_%04d_%.5f_%.1f_%.1f.peasoup",
	      ii,1.0/cands[ii].freq,cands[ii].dm,cands[ii].acc);
      filepath << output_directory << "/" << filename;
      FILE* fo = fopen(filepath.str().c_str(),"w");
      
      
      
      
      cands[ii].print(fo);
      fclose(fo);
    }
  }
  
  void write_candidate_file(std::string filepath="./candidates.txt") {
    FILE* fo = fopen(filepath.c_str(),"w");
    fprintf(fo,"#Period...Optimal period...Frequency...DM...Acceleration...Harmonic number...S/N...Folded S/N\n");
    for (int ii=0;ii<cands.size();ii++){
      fprintf(fo,"#Candidate %d\n",ii);
      cands[ii].print(fo);
    }
    fclose(fo);
  }
};

class CandidateCollection_template_bank {
public:
  std::vector<Candidate_template_bank> cands;

  CandidateCollection_template_bank(){}

  void append(CandidateCollection_template_bank& other){
    cands.insert(cands.end(),other.cands.begin(),other.cands.end());
  }

  void append(std::vector<Candidate_template_bank> other){
    cands.insert(cands.end(),other.begin(),other.end());
  }

  void reset(void){
    cands.clear();
  }

  void print(FILE* fo=stdout){
    for (int ii=0;ii<cands.size();ii++)
      cands[ii].print(fo);
  }

  void generate_candidate_binaries(std::string output_directory="./") {
    char filename[80];
    std::stringstream filepath;
    for (int ii=0;ii<cands.size();ii++){
      filepath.str("");
      sprintf(filename,"cand_%04d_%.5f_%.1f_%.1f_%.1f_%.1f.peasoup",
              ii,1.0/cands[ii].freq,cands[ii].dm,cands[ii].omega,cands[ii].tau,cands[ii].phi,cands[ii].long_periastron,cands[ii].eccentricity);
      filepath << output_directory << "/" << filename;
      FILE* fo = fopen(filepath.str().c_str(),"w");




      cands[ii].print(fo);
      fclose(fo);
    }
  }

  void write_candidate_file(std::string filepath="./candidates_from_template_bank.txt") {
    FILE* fo = fopen(filepath.c_str(),"w");
    fprintf(fo,"#Period...Optimal period...Frequency...DM...Angular Velocity...Projected Radius...Initial Orbital Phase...Longtitude of Periastron...Eccentricity...Harmonic number...S/N...Folded S/N\n");
    for (int ii=0;ii<cands.size();ii++){
      fprintf(fo,"#Candidate %d\n",ii);
      cands[ii].print(fo);
    }
    fclose(fo);
  }
};


class SpectrumCandidates: public CandidateCollection {
public:
  float dm;
  int dm_idx;
  float acc;

  SpectrumCandidates(float dm, int dm_idx, float acc)
    :dm(dm),dm_idx(dm_idx),acc(acc){}
  
  void append(float* snrs, float* freqs, int nh, int size){
    cands.reserve(size+cands.size());
    for (int ii=0;ii<size;ii++)
      cands.push_back(Candidate(dm,dm_idx,acc,nh,snrs[ii],freqs[ii]));
  }
};


class SpectrumCandidates_templatebank: public CandidateCollection_template_bank {
public:
  float dm;
  int dm_idx;
  float omega;
  float tau;
  float phi;
  float long_periastron;
  float eccentricity;

  SpectrumCandidates_templatebank(float dm, int dm_idx, float omega, float tau, float phi, float long_periastron, float eccentricity)
    :dm(dm),dm_idx(dm_idx),omega(omega),tau(tau),phi(phi),long_periastron(long_periastron),eccentricity(eccentricity){}

  void append(float* snrs, float* freqs, int nh, int size){
    cands.reserve(size+cands.size());
    for (int ii=0;ii<size;ii++)
      cands.push_back(Candidate_template_bank(dm,dm_idx,omega,tau,phi,long_periastron,eccentricity,nh,snrs[ii],freqs[ii]));
  }
};
