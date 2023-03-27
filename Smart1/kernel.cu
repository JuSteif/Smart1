#include "Matrix.cuh"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

class Layer {
private:
	bool outputNeuron;
public:
	Matrix Weights;
	Matrix Outputs;
	Matrix* Inputs;

	Layer(int sizeInput, int sizeOutput, bool outputNeuron, Matrix *Inputs): outputNeuron(outputNeuron), Inputs(Inputs) {
		int error;
		Weights = Matrix(sizeInput, sizeOutput, &error, 2, 1);
		if (error != cudaSuccess) return;
	}

	void deleteLayer() {
		Weights.deleteMatrix();
		Outputs.deleteMatrix();
	}

	void forward() {
		Outputs = Weights * *Inputs;
	}
};

int main(int argc, char** argv) {
	int error;
	Matrix Inputs(1, 10, &error, 2, 0.00000001);
	if (error != cudaSuccess) return;

	Layer layer(10, 5, true, &Inputs);
	

	//*
	Inputs.printMatrix();
	printf("\n");
	layer.Weights.printMatrix();
	//*/

	layer.forward();
	
	printf("\nResult: %d\n", layer.Outputs.getHeight());
	//printf("Value: %f", C.getData(0, 0));
	layer.Outputs.printMatrix();

	Inputs.deleteMatrix();
	layer.deleteLayer();

	return 0;
}