#pragma once
#include "Matrix.h"

constexpr int block_size = 30;
using FloatT = float;
using Matrix_t = Matrix<FloatT>;

Matrix_t cuda_matmul(Matrix_t & h_a, Matrix_t & h_b, double * time = nullptr, bool use_mm2 = true);
Matrix_t cuda_matmul_with_benchmark(Matrix_t & h_a, Matrix_t & h_b, int, double * time = nullptr, bool use_mm2 = true);