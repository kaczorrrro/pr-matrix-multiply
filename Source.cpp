#include <iostream>
#include "Matrix.h"
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdexcept>
#include <string>
#include "MM Params.h"


void multiply_matrices_IKJ(Matrix_t & a, Matrix_t & b, Matrix_t & out)
{
#pragma omp parallel for 
	for (int i = 0; i < out.rows; i++) {
		for (int k = 0; k < a.columns; k++) {
			for (int j = 0; j < out.columns; j++) {
				out.at(i, j) += a.at(i, k) * b.at(k, j);
				//No race, since each thread writes to different element
				//mostly no cache invalidation since these elems are far away
			}
		}
	}
}

void multiply_matrices_JKI(Matrix_t & a, Matrix_t & b, Matrix_t & out)
{
#pragma omp parallel for 
	for (int j = 0; j < out.columns; j++) {
		for (int k = 0; k < a.columns; k++) {
			for (int i = 0; i < out.rows; i++) {
				out.at(i, j) += a.at(i, k) * b.at(k, j);
				//No race, since each thread writes to different element
				//Cache invalidation will happen, sice these elems are almost next to each other

			}
		}
	}
}
struct Int1_t
{
	unsigned int a : 1;
};


int main()
{
	int matrix_size = 16;
	Matrix_t h_a(matrix_size, matrix_size); for (FloatT & e : h_a) e = rand() % 5;
	Matrix_t h_b(matrix_size, matrix_size); for (FloatT & e : h_b) e = rand() % 4;
	h_a.print();
	std::cout << std::endl;
	h_b.print();
	std::cout << std::endl;
	Matrix_t h_out_gpu = cuda_matmul(h_a, h_b);
	Matrix_t h_out_cpu(h_out_gpu.rows, h_out_gpu.columns);
	h_out_cpu.setAllTo(0);
	multiply_matrices_IKJ(h_a, h_b, h_out_cpu);

	std::cout << "\nGPU:" << std::endl;
	h_out_gpu.print();
	std::cout << "\nCPU: " << std::endl;
	h_out_cpu.print();

	std::cout << "CPU == GPU: " << (h_out_cpu == h_out_gpu) << std::endl;

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}




