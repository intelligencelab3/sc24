#!/bin/bash

# Navigate to your project directory.
# cd MergePath-SpMM/ICCAD-Accel-GCN/

echo "Partitioning\n"

python block_level_partitionV8.py

# mkdir build

cd build

cmake ..

make -j10

touch test_results_booth.txt

echo "Testing 64mul"
# printf "\n\n\n\n">>test_results.txt

./spmm_test booth_mult64_shared 32 > test_results_booth.txt

echo "Testing 128mul"
printf "\n\n\n\n">>test_results_booth.txt


./spmm_test booth_mult128_shared 32 >> test_results_booth.txt


echo "Testing 192mul"
printf "\n\n\n\n">>test_results_booth.txt


./spmm_test booth_mult192_shared 32 >> test_results_booth.txt


echo "Testing 256mul"
printf "\n\n\n\n">>test_results_booth.txt


./spmm_test booth_mult256_shared 32 >> test_results_booth.txt


echo "Testing 320mul"
printf "\n\n\n\n">>test_results_booth.txt


./spmm_test booth_mult320_shared 32 >> test_results_booth.txt

echo "Testing 384mul"
printf "\n\n\n\n">>test_results_booth.txt


./spmm_test booth_mult384_shared 32 >> test_results_booth.txt


echo "Testing 448mul"
printf "\n\n\n\n">>test_results_booth.txt


./spmm_test booth_mult448_shared 32 >> test_results_booth.txt


echo "Testing 512mul"

printf "\n\n\n\n">>test_results_booth.txt

./spmm_test booth_mult512_shared 32 >> test_results_booth.txt

echo "Test run finished."