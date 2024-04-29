#!/bin/bash

# Navigate to your project directory.
# cd MergePath-SpMM/ICCAD-Accel-GCN/

echo "Partitioning\n"

python block_level_partitionV8.py

# mkdir build

cd build

cmake ..

make -j10

touch test_results_mapped.txt

echo "Testing 64mul"
# printf "\n\n\n\n">>test_results.txt

./spmm_test mult64_7nm_mapped_shared 32 > test_results_mapped.txt

echo "Testing 128mul"
printf "\n\n\n\n">>test_results_mapped.txt

./spmm_test mult128_7nm_mapped_shared 32 >> test_results_mapped.txt

echo "Testing 192mul"
printf "\n\n\n\n">>test_results_mapped.txt

./spmm_test mult192_7nm_mapped_shared 32 >> test_results_mapped.txt

echo "Testing 256mul"
printf "\n\n\n\n">>test_results_mapped.txt

./spmm_test mult256_7nm_mapped_shared 32 >> test_results_mapped.txt

echo "Testing 320mul"
printf "\n\n\n\n">>test_results_mapped.txt

./spmm_test mult320_7nm_mapped_shared 32 >> test_results_mapped.txt

echo "Testing 384mul"
printf "\n\n\n\n">>test_results_mapped.txt

./spmm_test mult384_7nm_mapped_shared 32 >> test_results_mapped.txt

echo "Testing 448mul"
printf "\n\n\n\n">>test_results_mapped.txt

./spmm_test mult448_7nm_mapped_shared 32 >> test_results_mapped.txt

echo "Testing 512mul"

printf "\n\n\n\n">>test_results_mapped.txt

./spmm_test mult512_7nm_mapped_shared 32 >> test_results_mapped.txt

echo "Test run finished."