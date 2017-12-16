#pragma once
#include <iostream>
#include <algorithm>
#include "cuda_compat.h"
#include <cassert>

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
