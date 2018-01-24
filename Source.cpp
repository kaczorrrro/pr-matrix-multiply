#include <iostream>
#include "Matrix.h"
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdexcept>
#include <string>
#include <chrono>
#include "MM Params.h"
#include "Windows.h"
#include <cmath>


//Baseline
void multiply_matrices_IJK(Matrix_t & a, Matrix_t & b, Matrix_t & out)
{
	for (int i = 0; i < out.rows; i++) {
		for (int j = 0; j < out.columns; j++) {
			for (int k = 0; k < a.columns; k++) {			
				out.at(i, j) += a.at(i, k) * b.at(k, j);
			}
		}
	}
}

void multiply_matrices_IKJ(Matrix_t & A, Matrix_t & B, Matrix_t & C)
{
#pragma omp parallel
	{
		auto a = A.shallow_copy();
		auto b = B.shallow_copy();
		auto out = C.shallow_copy();
#pragma omp for 
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

}


void multiply_matrices_JKI(Matrix_t & A, Matrix_t & B, Matrix_t & C)
{
#pragma omp parallel
	{
		auto a = A.shallow_copy();
		auto b = B.shallow_copy();
		auto out = C.shallow_copy();
#pragma omp for 
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
}

//void multiply_matrices_IKJ(float* a, float* b, float * out, int n)
//{
//	//#pragma omp parallel for
//	for (int i = 0; i < n; i++) {
//		for (int k = 0; k < n; k++) {
//			for (int j = 0; j < n; j++) {
//				out[i*n + j] += a[i*n + k] * b[k*n + j];
//				//No race, since each thread writes to different element
//				//mostly no cache invalidation since these elems are far away
//			}
//		}
//	}
//}

struct Int1_t
{
	unsigned int a : 1;
};


void cpu_test() {
	using namespace std::chrono;

	int min_size = 1;
	int max_size = 2048;
	long long iters_start = pow(max_size / min_size, 2.0) * 4;

	std::cout << "Algorithm,Size,Total bytes,Total time[us],Tests,Time per test,Time per elem" << std::endl;
	for(double size_double=min_size, iters_double = iters_start; size_double <=max_size; size_double *=sqrt(sqrt(2)), iters_double /=sqrt(2)) {
		int size = static_cast<int>(round(size_double));
		long long iters = static_cast<long long>(iters_double);
		auto start = high_resolution_clock::now();
		auto end = high_resolution_clock::now();

		Matrix_t h_a(size, size); for (FloatT & e : h_a) e = rand();
		Matrix_t h_b(size, size); for (FloatT & e : h_b) e = rand();
		Matrix_t h_out(size, size);

		start = high_resolution_clock::now();
		for (long long i = 0; i < iters; i++)
			multiply_matrices_JKI(h_a, h_b, h_out);
		end = high_resolution_clock::now();
		std::cout
			<< "JKI,"
			<< size << ","
			<< size * size * 3 * sizeof(FloatT) << ","
			<< duration<double>(end - start).count() << ","
			<< iters << ","
			<< duration<double>(end - start).count() / iters << ","
			<< duration<double>(end - start).count() / iters / (size*size) << ","
			<< std::endl;

		start = high_resolution_clock::now();
		for (long long i = 0; i < iters; i++)
			multiply_matrices_IKJ(h_a, h_b, h_out);
		end = high_resolution_clock::now();
		std::cout
			<< "IKJ,"
			<< size << ","
			<< size * size * 3 * sizeof(FloatT) << ","
			<< duration<double>(end - start).count() << ","
			<< iters << ","
			<< duration<double>(end - start).count() / iters << ","
			<< duration<double>(end - start).count() / iters / (size*size) << ","
			<< std::endl;

		//start = high_resolution_clock::now();
		//for (long long i = 0; i < iters; i++)
		//	multiply_matrices_IJK(h_a, h_b, h_out);
		//end = high_resolution_clock::now();
		//std::cout
		//	<< "IJK,"
		//	<< size << ","
		//	<< size * size * 3 * sizeof(FloatT) << ","
		//	<< duration<double>(end - start).count() << ","
		//	<< iters << ","
		//	<< duration<double>(end - start).count()/iters << ","
		//	<< duration<double>(end - start).count()/iters/(size*size) << ","
		//	<< std::endl;
	}
}


void cpu_num_cores() {
	using namespace std::chrono;
	float useless = 0;
	for (int i = 0; i < 100;i++) {
		int size = 1024;
		long long iters = 1 << 0;

		Matrix_t h_a(size, size); for (FloatT & e : h_a) e = rand();
		Matrix_t h_b(size, size); for (FloatT & e : h_b) e = rand();
		Matrix_t h_out(size, size);

		//auto start = high_resolution_clock::now();
		//for (long long i = 0; i < iters; i++)
		//	multiply_matrices_JKI(h_a, h_b, h_out);
		//auto end = high_resolution_clock::now();
		//std::cout
		//	<< "JKI,"
		//	<< size << ","
		//	<< size * size * 3 * sizeof(FloatT) << ","
		//	<< duration_cast<microseconds>(end - start).count() << ","
		//	<< iters << std::endl;

		auto start = high_resolution_clock::now();
		for (long long i = 0; i < iters; i++) {
			//multiply_matrices_IKJ(h_a.begin(), h_b.begin(), h_out.begin(), size);
			multiply_matrices_IKJ(h_a, h_b, h_out);
		}

		auto end = high_resolution_clock::now();
		std::cout
			<< "IKJ,"
			<< size << ","
			<< size * size * 3 * sizeof(FloatT) << ","
			<< duration_cast<microseconds>(end - start).count() << ","
			<< iters << std::endl;
		for (FloatT & e : h_out) 
			useless += e;
	}
	std::cout << useless << std::endl;
}


void page_size() {
	SYSTEM_INFO systemInfo;
	GetSystemInfo(&systemInfo);
	std::cout << systemInfo.dwPageSize;
}

int main()
{
	cpu_test();
	//cpu_num_cores();
	return 0;
	int matrix_size = 4096;
	//for (int i=8; i<1000;i++) {
	//	Matrix_t h_a(i, i); for (FloatT & e : h_a) e = rand();
	//	Matrix_t h_b(i, i); for (FloatT & e : h_b) e = rand();

	//	Matrix_t h_out_gpu = cuda_matmul(h_a, h_b, true);
	//	Matrix_t h_out_cpu(h_out_gpu.rows, h_out_gpu.columns);

	//	multiply_matrices_IKJ(h_a, h_b, h_out_cpu);
	//	std::cout << i << std::endl;
	//	if (!(h_out_gpu == h_out_cpu))
	//		std::cout << std::endl<< i << "Fail" << std::endl;
	//	std::cout << "Done " << std::endl;
	//	break;
	//}


	Matrix_t h_a(matrix_size, matrix_size); for (FloatT & e : h_a) e = rand();
	Matrix_t h_b(matrix_size, matrix_size); for (FloatT & e : h_b) e = rand();
	//h_a.print();
	//std::cout << std::endl;
	//h_b.print();
	//std::cout << std::endl;
	double time;
	Matrix_t h_out_gpu = cuda_matmul_with_benchmark(h_a, h_b, 20, &time, false);
	std::cout << "end" << std::endl;
	std::cout << "Took " << time << std::endl;
	//Matrix_t h_out_cpu(h_out_gpu.rows, h_out_gpu.columns);
	//h_out_cpu.setAllTo(0);
	//multiply_matrices_IKJ(h_a, h_b, h_out_cpu);

	//std::cout << "\nGPU:" << std::endl;
	//h_out_gpu.print();
	//std::cout << "\nCPU: " << std::endl;
	//h_out_cpu.print();

	//std::cout << "CPU == GPU: " << (h_out_cpu == h_out_gpu) << std::endl;

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}




