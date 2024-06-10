10_cavity:10_cavity.cpp
	g++ -std=c++11 $^
10_cavity_cuda: 10_cavity.cu
	nvcc -std=c++11 $^
10_cavity_simd: 10_cavity_simd.cpp
	g++ -std=c++11 -march=native -O3 $^
