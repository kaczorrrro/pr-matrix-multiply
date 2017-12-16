#pragma once
#include "Matrix.h"

constexpr int block_size = 2;
using FloatT = int;
using Matrix_t = Matrix<FloatT>;

Matrix_t cuda_matmul(Matrix_t & h_a, Matrix_t & h_b);