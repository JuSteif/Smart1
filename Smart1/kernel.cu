#include "Matrix.cuh"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
	int error;
	Matrix A(10, 5, &error, 2, 0.1);
	if (error != cudaSuccess) return;
	Matrix B(1, 10, &error, 2, 1);
	if (error != cudaSuccess) return;

	//*
	A.printMatrix();
	printf("\n");
	B.printMatrix();
	//*/

	Matrix C = A * B;
	
	printf("\nResult: %d\n", C.getHeight());
	//printf("Value: %f", C.getData(0, 0));
	C.printMatrix();

	A.deleteMatrix();
	B.deleteMatrix();
	C.deleteMatrix();

	return 0;
}