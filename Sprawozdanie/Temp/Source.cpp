#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdexcept>
#include <string>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>

#define CUDA_CALLABLE_MEMBER 

template <class type>
class Matrix {
public:
	using value_type = type;
	const int rows, columns;

	CUDA_CALLABLE_MEMBER Matrix(const int rows, const int columns) :rows(rows), columns(columns), shoudl_delete(true){
		//Zero initialize
		matrix = new type[rows*columns]();
	}

	CUDA_CALLABLE_MEMBER Matrix(type* data, const int rows, const int columns) :
		rows(rows), 
		columns(columns), 
		shoudl_delete(false),
		matrix(data){}

	CUDA_CALLABLE_MEMBER Matrix shallow_copy() {
		return Matrix(matrix, rows, columns);
	}


	CUDA_CALLABLE_MEMBER Matrix (Matrix && other): rows(other.rows), columns(other.columns), matrix(other.matrix), shoudl_delete(other.shoudl_delete){
		other.matrix = nullptr;
	}

	CUDA_CALLABLE_MEMBER ~Matrix() {
		if (shoudl_delete)
			delete[] matrix;
	}

	CUDA_CALLABLE_MEMBER type* begin() {
		return matrix;
	}

	CUDA_CALLABLE_MEMBER type* end() {
		return matrix + columns*rows;
	}

	CUDA_CALLABLE_MEMBER type & at(const int & row, const int & column) {
#ifndef NDEBUG
		rangeCheck(row, column);
#endif
		return matrix[row*columns + column];
	}

	CUDA_CALLABLE_MEMBER size_t num_elems() {
		return columns * rows;
	}

	 void print() {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				std::cout << at(i, j) << " ";
			}
			std::cout << std::endl;
		}
	}

	void setAllTo(const type & value) {
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < columns; j++)
				at(i, j) = value;
	}

	Matrix T() {
		Matrix transposed(rows, columns);
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < columns; j++)
				transposed.at(i, j) = at(j, i);
		return transposed;
	}

	bool operator==(Matrix & m1) {
		if (columns != m1.columns || rows != m1.rows)
			return false;

		type* a = this->begin();
		type* b = m1.begin();

		while (a != this->end())
			if (*a++ != *b++)
				return false;
		return true;
	}




private:
	type * matrix;
	const bool shoudl_delete;
	CUDA_CALLABLE_MEMBER void rangeCheck(int row, int column) {
		assert(row >= 0 && row < rows && column >= 0 && column < columns);
		//if (row < 0 || row >= rows || column < 0 || column >= columns) {
		//	//std::cout << "Row: " << row << ", col: " << column << " out of index" << std::endl;
		//	//throw std::invalid_argument("Wrong index");
		//}
	}

};


using FloatT = float;
using Matrix_t = Matrix<FloatT>;


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


void cpu_test() {
	using namespace std::chrono;

	int min_size = 1;
	int max_size = 2048;
	long long iters_start = pow(max_size / min_size, 2.0) * 4;
	int skip = false;
	std::cout << "Algorithm,Size,Total bytes,Total time[us],Tests,Time per test,Time per elem" << std::endl;
	for(double size_double=min_size, iters_double = iters_start; size_double <=max_size; size_double *=sqrt(sqrt(2)), iters_double /=sqrt(2)) {
		int size = static_cast<int>(round(size_double));
		if (size == 1 && skip)continue;
		skip = true;
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

		start = high_resolution_clock::now();
		for (long long i = 0; i < iters; i++)
			multiply_matrices_IJK(h_a, h_b, h_out);
		end = high_resolution_clock::now();
		std::cout
			<< "IJK,"
			<< size << ","
			<< size * size * 3 * sizeof(FloatT) << ","
			<< duration<double>(end - start).count() << ","
			<< iters << ","
			<< duration<double>(end - start).count()/iters << ","
			<< duration<double>(end - start).count()/iters/(size*size) << ","
			<< std::endl;
	}
}

int main()
{
	cpu_test();
	return 0;
}