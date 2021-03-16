#!/bin/bash
rm -rf bin
make -j 4 check_wfc
cp ../../Vitis_Libraries/blas/L3/examples/gemm/build_dir.hw_emu.xilinx_u250_gen3x16_xdma_3_1_202020_1/blas.xclbin bin/
cp ../../Vitis_Libraries/blas/L3/examples/gemm/build_dir.hw_emu.xilinx_u250_gen3x16_xdma_3_1_202020_1/config_info.dat bin/
cp ../../Vitis_Libraries/blas/L3/examples/gemm/emconfig.json bin/
