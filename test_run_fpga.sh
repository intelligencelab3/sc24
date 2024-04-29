#!/bin/bash

# Navigate to your project directory.
# cd MergePath-SpMM/ICCAD-Accel-GCN/

echo "Partitioning\n"

python block_level_partitionV8.py

# mkdir build

cd build

cmake ..

make -j10

touch test_results_fpga.txt

echo "Testing 64mul"
# printf "\n\n\n\n">>test_results.txt

./spmm_test mult64_fpga_4lut_shared 32 > test_results_fpga.txt

echo "Testing 128mul"
printf "\n\n\n\n">>test_results_fpga.txt


./spmm_test mult128_fpga_4lut_shared 32 >> test_results_fpga.txt


echo "Testing 192mul"
printf "\n\n\n\n">>test_results_fpga.txt


./spmm_test mult192_fpga_4lut_shared 32 >> test_results_fpga.txt


echo "Testing 256mul"
printf "\n\n\n\n">>test_results_fpga.txt


./spmm_test mult256_fpga_4lut_shared 32 >> test_results_fpga.txt


echo "Testing 320mul"
printf "\n\n\n\n">>test_results_fpga.txt


./spmm_test mult320_fpga_4lut_shared 32 >> test_results_fpga.txt

echo "Testing 384mul"
printf "\n\n\n\n">>test_results_fpga.txt


./spmm_test mult384_fpga_4lut_shared 32 >> test_results_fpga.txt


echo "Testing 448mul"
printf "\n\n\n\n">>test_results_fpga.txt


./spmm_test mult448_fpga_4lut_shared 32 >> test_results_fpga.txt


echo "Testing 512mul"

printf "\n\n\n\n">>test_results_fpga.txt

./spmm_test mult512_fpga_4lut_shared 32 >> test_results_fpga.txt

echo "Test run finished."