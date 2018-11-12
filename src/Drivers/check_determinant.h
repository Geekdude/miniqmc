#ifndef QMCPLUSPLUS_CHECK_DETERMINANT_H
#define QMCPLUSPLUS_CHECK_DETERMINANT_H
#include "QMCWaveFunctions/future/DeterminantDeviceImp.h"
namespace qmcplusplus
{

class CheckDeterminantTest
{
public:
  // clang-format off
  typedef QMCTraits::RealType           RealType;
  typedef ParticleSet::ParticlePos_t    ParticlePos_t;
  typedef ParticleSet::PosType          PosType;
  // clang-format on

  int na        = 1;
  int nb        = 1;
  int nc        = 1;
  int nsteps    = 5;
  int iseed     = 11;
  int nsubsteps = 1;
  int np        = 1;
  
  bool verbose = false;

  ParticleSet ions;
  Tensor<OHMMS_PRECISION, 3> lattice_b;
  int error;

  void setup(int argc, char** argv);

  int run_test();  
};

template<future::DeterminantDeviceType DT>
class CheckDeterminantHelpers
{
public:
  using QMCT = QMCTraits;
  static void initialize(int arc, char** argv);
  static double runThreads(int np, PrimeNumberSet<uint32_t>& myPrimes,
			  ParticleSet& ions, int& nsteps,
			 int& nsubsteps);
  static void test(int& error, ParticleSet& ions, int& nsteps, int& nsubsteps, int& np);
  static void finalize();
private:
  static void thread_main(const int ip,
			  const PrimeNumberSet<uint32_t>& myPrimes,
			  const ParticleSet& ions,
			  const int& nsteps,
			  const int& nsubsteps,
			  double& accumulated_error);
  static void updateFromDevice(future::DiracDeterminant<future::DeterminantDeviceImp<DT>>& determinant_device);
};


 
}
#endif
