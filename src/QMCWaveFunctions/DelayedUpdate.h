//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_DELAYED_UPDATE_H
#define QMCPLUSPLUS_DELAYED_UPDATE_H

#define USE_FPGA

#include "config.h"
#include <Numerics/OhmmsPETE/OhmmsVector.h>
#include <Numerics/OhmmsPETE/OhmmsMatrix.h>
#include "Numerics/OhmmsBlas.h"
#include "QMCWaveFunctions/DiracMatrix.h"
#include "Numerics/BlasThreadingEnv.h"
#include "xf_blas.hpp"

#define XF_CHECK(status, call)                                             \
    status = call;                                              \
    if (status != XFBLAS_STATUS_SUCCESS) {                           \
      printf("%s:%d Error calling " #call ", error code is: %d\n",  \
              __FILE__,__LINE__, status);                            \
      exit(1);                                           \
    }

namespace qmcplusplus
{
/** implements delayed update on CPU using BLAS
 * @tparam T base precision for most computation
 * @tparam T_FP high precision for matrix inversion, T_FP >= T
 */
template<typename T, typename T_FP>
class DelayedUpdate
{
  /// define real type
  using real_type = typename scalar_traits<T>::real_type;
  /// orbital values of delayed electrons
  Matrix<T> U;
  /// rows of Ainv corresponding to delayed electrons
  Matrix<T> V;
  /// Matrix inverse of B, at maximum KxK
  Matrix<T> Binv;
  /// scratch space, used during inverse update
  Matrix<T> tempMat;
  /// temporal scratch space used by SM-1
  Vector<T> temp;
  /// new column of B
  Vector<T> p;
  /// list of delayed electrons
  std::vector<int> delay_list;
  /// current number of delays, increase one for each acceptance, reset to 0 after updating Ainv
  int delay_count;
  /// matrix inversion engine
  DiracMatrix<T_FP, T> detEng;
  /// FPGA Malloc
  bool fpga_malloc;
  /// Number of FPGA Kernels
  int l_numKernel;

  float *d_V = nullptr;
  float *d_U = nullptr;
  float *d_Binv = nullptr;
  float *d_tempMat = nullptr;

public:
  /// default constructor
  DelayedUpdate() : delay_count(0), fpga_malloc(false), l_numKernel(1) {
    string l_xclbinFile("blas.xclbin");
    string l_configFile("config_info.dat");
    xfblasEngine_t engineName = XFBLAS_ENGINE_GEMM;
    xfblasStatus_t status;

    XF_CHECK(status, xfblasCreate(l_xclbinFile.c_str(), l_configFile, engineName, l_numKernel));
  }

  /// default deconstructor
  ~DelayedUpdate() {
    xfblasDestroy(l_numKernel);
  }

  /** resize the internal storage
   * @param norb number of electrons/orbitals
   * @param delay, maximum delay 0<delay<=norb
   */
  inline void resize(int norb, int delay)
  {
    V.resize(delay, norb);
    U.resize(delay, norb);
    p.resize(delay);
    temp.resize(norb);
    tempMat.resize(norb, delay);
    Binv.resize(delay, delay);
    delay_list.resize(delay);

#ifdef USE_FPGA
    xfblasStatus_t status;
    if (fpga_malloc)
    {
      xfblasFree(d_V, l_numKernel - 1);
      xfblasFree(d_U, l_numKernel - 1);
      xfblasFree(d_Binv, l_numKernel - 1);
      xfblasFree(d_tempMat, l_numKernel - 1);
    }

    printf("In Resize\n");
    fflush(stdout);
    XF_CHECK(status, xfblasMalloc(&d_V, delay, norb, sizeof(*V.data()), l_numKernel - 1));
    printf("Aloc V\n");
    fflush(stdout);
    XF_CHECK(status, xfblasMalloc(&d_U, delay, norb, sizeof(*U.data()), l_numKernel - 1));
    printf("Aloc U\n");
    fflush(stdout);
    XF_CHECK(status, xfblasMalloc(&d_Binv, delay, delay, sizeof(*Binv.data()), l_numKernel - 1));
    printf("Aloc Binv\n");
    fflush(stdout);
    XF_CHECK(status, xfblasMalloc(&d_tempMat, norb, delay, sizeof(*tempMat.data()), l_numKernel - 1));
    printf("Aloc tempMat\n");
    fflush(stdout);
    printf("Leaving Resize\n");
    fflush(stdout);

    fpga_malloc = true;
#endif
  }

  /** compute the inverse of the transpose of matrix A
   * @param logdetT orbital value matrix
   * @param Ainv inverse matrix
   */
  inline void invert_transpose(const Matrix<T>& logdetT, Matrix<T>& Ainv, real_type& LogValue, real_type& PhaseValue)
  {
    detEng.invert_transpose(logdetT, Ainv, LogValue, PhaseValue);
    // safe mechanism
    delay_count = 0;
  }

  /** initialize internal objects when Ainv is refreshed
   * @param Ainv inverse matrix
   */
  inline void initializeInv(const Matrix<T>& Ainv)
  {
    // safe mechanism
    delay_count = 0;
  }

  /** compute the row of up-to-date Ainv
   * @param Ainv inverse matrix
   * @param rowchanged the row id corresponding to the proposed electron
   */
  template<typename VVT>
  inline void getInvRow(const Matrix<T>& Ainv, int rowchanged, VVT& invRow)
  {
    if (delay_count == 0)
    {
      // Ainv is fresh, directly access Ainv
      std::copy_n(Ainv[rowchanged], invRow.size(), invRow.data());
      return;
    }
    const T cone(1);
    const T czero(0);
    const int norb     = Ainv.rows();
    const int lda_Binv = Binv.cols();
    // save Ainv[rowchanged] to invRow
    std::copy_n(Ainv[rowchanged], norb, invRow.data());
    // multiply V (NxK) Binv(KxK) U(KxN) invRow right to the left
    BLAS::gemv('T', norb, delay_count, cone, U.data(), norb, invRow.data(), 1, czero, p.data(), 1);
    BLAS::gemv('N', delay_count, delay_count, cone, Binv.data(), lda_Binv, p.data(), 1, czero, Binv[delay_count], 1);
    BLAS::gemv('N', norb, delay_count, -cone, V.data(), norb, Binv[delay_count], 1, cone, invRow.data(), 1);
  }

  /** accept a move with the update delayed
   * @param Ainv inverse matrix
   * @param rowchanged the row id corresponding to the proposed electron
   * @param psiV new orbital values
   *
   * Before delay_count reaches the maximum delay, only Binv is updated with a recursive algorithm
   */
  template<typename VVT>
  inline void acceptRow(Matrix<T>& Ainv, int rowchanged, const VVT& psiV)
  {
    const T cminusone(-1);
    const T czero(0);
    const int norb     = Ainv.rows();
    const int lda_Binv = Binv.cols();
    std::copy_n(Ainv[rowchanged], norb, V[delay_count]);
    std::copy_n(psiV.data(), norb, U[delay_count]);
    delay_list[delay_count] = rowchanged;
    // the new Binv is [[X Y] [Z x]]
    BLAS::gemv('T', norb, delay_count + 1, cminusone, V.data(), norb, psiV.data(), 1, czero, p.data(), 1);
    // x
    T y = -p[delay_count];
    for (int i = 0; i < delay_count; i++)
      y += Binv[delay_count][i] * p[i];
    Binv[delay_count][delay_count] = y = T(1) / y;
    // Y
    BLAS::gemv('T', delay_count, delay_count, y, Binv.data(), lda_Binv, p.data(), 1, czero, Binv.data() + delay_count,
               lda_Binv);
    // X
    BLAS::ger(delay_count, delay_count, cminusone, Binv[delay_count], 1, Binv.data() + delay_count, lda_Binv,
              Binv.data(), lda_Binv);
    // Z
    for (int i = 0; i < delay_count; i++)
      Binv[delay_count][i] *= -y;
    delay_count++;
    // update Ainv when maximal delay is reached
    if (delay_count == lda_Binv)
      updateInvMat(Ainv);
  }

  /** update the full Ainv and reset delay_count
   * @param Ainv inverse matrix
   */
  inline void updateInvMat(Matrix<T>& Ainv)
  {
    if (delay_count == 0)
      return;
    // update the inverse matrix
    const T cone(1);
    const T czero(0);
    const int norb = Ainv.rows();

    #ifdef USE_FPGA
    xfblasStatus_t status;
    const int lda_Binv     = Binv.cols();

    float *d_Ainv = nullptr;

    XF_CHECK(status, xfblasMalloc(&d_Ainv, Ainv.rows(), Ainv.cols(), sizeof(*Ainv.data()), l_numKernel - 1));
    
    XF_CHECK(status, xfblasSetMatrix(U.rows(), U.cols(), sizeof(*U.data()), U.data(), U.cols(), d_U, l_numKernel - 1));
    XF_CHECK(status, xfblasSetMatrix(Ainv.rows(), Ainv.cols(), sizeof(*Ainv.data()), Ainv.data(), Ainv.cols(), d_Ainv, l_numKernel - 1));
    XF_CHECK(status, xfblasSetMatrix(tempMat.rows(), tempMat.cols(), sizeof(*tempMat.data()), tempMat.data(), tempMat.cols(), d_tempMat, l_numKernel - 1));
    XF_CHECK(status, xfblasGemm(XFBLAS_OP_T, XFBLAS_OP_N, delay_count, norb, norb, cone, U.data(), norb, Ainv.data(), norb, czero, tempMat.data(), lda_Binv, l_numKernel - 1));
    //              BxLAS::gemm(         'T',         'N',delay_count, norb, norb, cone, U.data(), norb, Ainv.data(), norb, czero, tempMat.data(), lda_Binv);
    XF_CHECK(status, xfblasGetMatrix(tempMat.rows(), tempMat.cols(), sizeof(*tempMat.data()), d_tempMat, tempMat.data(), tempMat.cols(), l_numKernel - 1));

    for (int i = 0; i < delay_count; i++)
      tempMat(delay_list[i], i) -= cone;

    XF_CHECK(status, xfblasSetMatrix(V.rows(), V.cols(), sizeof(*V.data()), V.data(), V.cols(), d_V, l_numKernel - 1));
    XF_CHECK(status, xfblasSetMatrix(Binv.rows(), Binv.cols(), sizeof(*Binv.data()), Binv.data(), Binv.cols(), d_Binv, l_numKernel - 1));
    XF_CHECK(status, xfblasGemm(XFBLAS_OP_N, XFBLAS_OP_N, norb, delay_count, delay_count, cone, V.data(), norb, Binv.data(), lda_Binv, czero, U.data(), norb, l_numKernel - 1));
    //              BxLAS::gemm(         'N',         'N',norb, delay_count, delay_count, cone, V.data(), norb, Binv.data(), lda_Binv, czero, U.data(), norb);
    XF_CHECK(status, xfblasGetMatrix(U.rows(), U.cols(), sizeof(*U.data()), d_U, U.data(), U.cols(), l_numKernel - 1));

    XF_CHECK(status, xfblasSetMatrix(tempMat.rows(), tempMat.cols(), sizeof(*tempMat.data()), tempMat.data(), tempMat.cols(), d_tempMat, l_numKernel - 1));
    XF_CHECK(status, xfblasGemm(XFBLAS_OP_N, XFBLAS_OP_N, norb, norb, delay_count, -cone, U.data(), norb, tempMat.data(), lda_Binv, cone, Ainv.data(), norb, l_numKernel - 1));
    //              BxLAS::gemm(         'N',         'N',norb, norb, delay_count, -cone, U.data(), norb, tempMat.data(), lda_Binv, cone, Ainv.data(), norb);
    XF_CHECK(status, xfblasGetMatrix(Ainv.rows(), Ainv.cols(), sizeof(*Ainv.data()), d_Ainv, Ainv.data(), Ainv.cols(), l_numKernel - 1));

    xfblasFree(Ainv.data(), l_numKernel - 1);
    #else
    if (delay_count == 1)
    {
      // this is a special case invoking the Fahy's variant of Sherman-Morrison update.
      // Only use the first norb elements of tempMat as a temporal array
      BLAS::gemv('T', norb, norb, cone, Ainv.data(), norb, U[0], 1, czero, temp.data(), 1);
      temp[delay_list[0]] -= cone;
      BLAS::ger(norb, norb, -Binv[0][0], V[0], 1, temp.data(), 1, Ainv.data(), norb);
    }
    else
    {
      const int lda_Binv     = Binv.cols();
      int num_threads_nested = getNextLevelNumThreads();
      // always use serial when norb is small or only one second level thread
      bool use_serial(norb <= 256 || num_threads_nested == 1);
      if (use_serial || BlasThreadingEnv::NestedThreadingSupported())
      {
        // threading depends on BLAS
        BlasThreadingEnv knob(use_serial ? 1 : num_threads_nested);
        BLAS::gemm('T', 'N', delay_count, norb, norb, cone, U.data(), norb, Ainv.data(), norb, czero, tempMat.data(),
                   lda_Binv);
        for (int i = 0; i < delay_count; i++)
          tempMat(delay_list[i], i) -= cone;
        BLAS::gemm('N', 'N', norb, delay_count, delay_count, cone, V.data(), norb, Binv.data(), lda_Binv, czero,
                   U.data(), norb);
        BLAS::gemm('N', 'N', norb, norb, delay_count, -cone, U.data(), norb, tempMat.data(), lda_Binv, cone,
                   Ainv.data(), norb);
      }
      else
      {
        // manually threaded version of the above GEMM calls
#pragma omp parallel
        {
          const int block_size = getAlignedSize<T>((norb + num_threads_nested - 1) / num_threads_nested);
          int num_block        = (norb + block_size - 1) / block_size;
#pragma omp for
          for (int ix = 0; ix < num_block; ix++)
          {
            int x_offset = ix * block_size;
            BLAS::gemm('T', 'N', delay_count, std::min(norb - x_offset, block_size), norb, cone, U.data(), norb,
                       Ainv[x_offset], norb, czero, tempMat[x_offset], lda_Binv);
          }
#pragma omp master
          for (int i = 0; i < delay_count; i++)
            tempMat(delay_list[i], i) -= cone;
#pragma omp for
          for (int iy = 0; iy < num_block; iy++)
          {
            int y_offset = iy * block_size;
            BLAS::gemm('N', 'N', std::min(norb - y_offset, block_size), delay_count, delay_count, cone,
                       V.data() + y_offset, norb, Binv.data(), lda_Binv, czero, U.data() + y_offset, norb);
          }
#pragma omp for collapse(2) nowait
          for (int iy = 0; iy < num_block; iy++)
            for (int ix = 0; ix < num_block; ix++)
            {
              int x_offset = ix * block_size;
              int y_offset = iy * block_size;
              BLAS::gemm('N', 'N', std::min(norb - y_offset, block_size), std::min(norb - x_offset, block_size),
                         delay_count, -cone, U.data() + y_offset, norb, tempMat[x_offset], lda_Binv, cone,
                         Ainv[x_offset] + y_offset, norb);
            }
        }
      }
    }
    #endif
    delay_count = 0;
  }
};
} // namespace qmcplusplus

#endif // QMCPLUSPLUS_DELAYED_UPDATE_H
