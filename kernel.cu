#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MM Params.h"

struct Int1_t
{
	unsigned int val:1;
};


__global__ void mm(Matrix_t a, Matrix_t b, Matrix_t out)
{
	//Declare shared blocks
	__shared__ FloatT shared_memory_0[block_size*block_size];
	__shared__ FloatT shared_memory_1[block_size*block_size];
	__shared__ FloatT shared_memory_2[block_size*block_size];
	__shared__ FloatT shared_memory_3[block_size*block_size];

	//Wrap them intro Matrix class
	Matrix_t shared_a[] = { Matrix_t(shared_memory_0, block_size, block_size), Matrix_t(shared_memory_1, block_size, block_size) };
	Matrix_t shared_b[] = { Matrix_t(shared_memory_2, block_size, block_size), Matrix_t(shared_memory_3, block_size, block_size) };

	//Fetch first block
	shared_a[0].at(threadIdx.y, threadIdx.x) = a.at(blockIdx.y*block_size + threadIdx.y, 0*block_size + threadIdx.x);
	shared_b[0].at(threadIdx.y, threadIdx.x) = b.at(0*block_size + threadIdx.y, blockIdx.x*block_size + threadIdx.x);
	__syncthreads();

	Int1_t fetch_idx{ 1 };
	FloatT acc = 0;
	for (int block_num = 0, blocks_to_process = a.columns / block_size; block_num < blocks_to_process; block_num++) {//TODO CEIL
		//Fetch next block
		if (block_num+1 != blocks_to_process) {
			int next_block_num = block_num + 1;
			shared_a[fetch_idx.val].at(threadIdx.y, threadIdx.x) = a.at(blockIdx.y*block_size + threadIdx.y, next_block_num*block_size + threadIdx.x);
			shared_b[fetch_idx.val].at(threadIdx.y, threadIdx.x) = b.at(next_block_num*block_size + threadIdx.y, blockIdx.x*block_size + threadIdx.x);
		}

		//Now we will make use of previously fetched block (and later fetch again to it's index)
		fetch_idx.val++;

		//To accumulate result
		for (int i = 0; i < block_size; i++) {
			acc += shared_a[fetch_idx.val].at(threadIdx.y, i)*shared_b[fetch_idx.val].at(i, threadIdx.x);
		}

		__syncthreads();
	}

	out.at(blockIdx.y*block_size + threadIdx.y, blockIdx.x*block_size + threadIdx.x) = acc;
}

Matrix_t cuda_matmul(Matrix_t & h_a, Matrix_t & h_b) {
	if (h_a.columns != h_b.rows)
		throw std::runtime_error("Sizes don't match");

	cudaError_t cudaStatus;
	FloatT * d_a_memory;
	FloatT * d_b_memory;
	FloatT * d_out_memory;

	Matrix_t h_out(h_a.rows, h_b.columns);

	try {
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess)
			throw std::runtime_error("cudaSetDevice failed!Do you have a CUDA - capable GPU installed ? ");

		cudaStatus = cudaMalloc((void**)&d_a_memory, h_a.num_elems() * sizeof(FloatT));
		if (cudaStatus != cudaSuccess)
			throw std::runtime_error("cudaMalloc A failed");

		cudaStatus = cudaMalloc((void**)&d_b_memory, h_b.num_elems() * sizeof(FloatT));
		if (cudaStatus != cudaSuccess)
			throw std::runtime_error("cudaMalloc B failed");

		cudaStatus = cudaMalloc((void**)&d_out_memory, h_out.num_elems() * sizeof(FloatT));
		if (cudaStatus != cudaSuccess)
			throw std::runtime_error("cudaMalloc Out failed");

		cudaStatus = cudaMemcpy(d_a_memory, h_a.begin(), h_a.num_elems() * sizeof(FloatT), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
			throw std::runtime_error("cudaMemcpy A failed");

		cudaStatus = cudaMemcpy(d_b_memory, h_b.begin(), h_a.num_elems() * sizeof(FloatT), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
			throw std::runtime_error("cudaMemcpy B failed");

		Matrix_t d_a(d_a_memory, h_a.rows, h_a.columns);
		Matrix_t d_b(d_b_memory, h_b.rows, h_b.columns);
		Matrix_t d_out(d_out_memory, h_out.rows, h_out.columns);

		//TODO ceil
		mm << <dim3(d_out.columns/block_size, d_out.rows/block_size), dim3(block_size, block_size) >> >(d_a.shallow_copy(), d_b.shallow_copy(), d_out.shallow_copy());

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			throw std::runtime_error("MM kernel launch failed");

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
			throw std::runtime_error("cudaDeviceSynchronize returned error code %d after launching addKernel!\n");

		cudaStatus = cudaMemcpy(h_out.begin(), d_out.begin(), d_out.num_elems() * sizeof(FloatT), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
			throw std::runtime_error("cudaMemcpy to host failed");

		return h_out;
	}
	catch (std::exception & e) {
		std::cerr << e.what() << std::endl;
		std::cerr << "Cuda status: " << cudaStatus << std::endl;
		cudaFree(d_a_memory);
		cudaFree(d_b_memory);
		cudaFree(d_out_memory);
		throw std::runtime_error("Cuda mm failed");
	}
}







//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//    {
//		int matrix_size = 4;
//		Matrix_t h_a(matrix_size, matrix_size); for (FloatT & e : h_a) e = 1;
//		Matrix_t h_b(matrix_size, matrix_size); for (FloatT & e : h_b) e = 1;
//		Matrix_t h_out(matrix_size, matrix_size);
//
//		FloatT* d_a;
//		cudaStatus = cudaMalloc((void**)&d_a, matrix_size * matrix_size * sizeof(FloatT));
//		cudaMemcpy(d_a, h_a.begin(), h_a.num_elems() * sizeof(FloatT), cudaMemcpyHostToDevice);
//		FloatT* d_b;
//		cudaStatus = cudaMalloc((void**)&d_b, matrix_size * matrix_size * sizeof(FloatT));
//		cudaMemcpy(d_b, h_b.begin(), h_a.num_elems() * sizeof(FloatT), cudaMemcpyHostToDevice);
//		FloatT* d_out;
//		cudaStatus = cudaMalloc((void**)&d_out, matrix_size * matrix_size * sizeof(FloatT));
//
//
//		Matrix_t a(d_a, matrix_size, matrix_size);
//		Matrix_t b(d_b, matrix_size, matrix_size);
//		Matrix_t out(d_out, matrix_size, matrix_size);
//		// Launch a kernel on the GPU with one thread for each element.
//		mm <<<dim3(1,1), dim3(block_size, block_size)>>>(a.shallow_copy(),b.shallow_copy(),out.shallow_copy());
//		cudaDeviceSynchronize();
//		cudaMemcpy(h_out.begin(), out.begin(), h_out.num_elems() * sizeof(FloatT), cudaMemcpyDeviceToHost);
//		h_out.print();
//    }
//
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
