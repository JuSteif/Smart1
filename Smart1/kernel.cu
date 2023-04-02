#include "Matrix.cuh"
#include "Layer.h"
#include "Network.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

int main(int argc, char** argv) {

	Network network(2);

	network.addLayer(1);
	
	network.prepareNetwork();
	
	network.forward();

	//*
	printf("Inputs:\n");
	network.Inputs.printMatrix();
	printf("\nWeights:\n");
	network.printNetwork();
	
	printf("\nOutputs:\n");
	network.getNetworkOutput()->printMatrix();
	//*/

	int error;
	Matrix Target = Matrix(1, 1, &error, 2, 1);
	network.Backpropogation(0.5, Target);

	network.deleteNetwork();


	/*int error;
	Matrix Ins(1, 2, &error, 2, 1);
	Matrix Outs(1, 1, &error, 2, 1);
	Layer layer = Layer(2, 1, true, &Ins, NULL, &Outs);

	layer.forward();
	layer.calculatErrorSignal();

	printf("\nResult:\n");
	layer.Outputs.printMatrix();
	printf("\n___________________________________\nError:\n");
	layer.ErrorSignal.printMatrix();

	layer.calculateNewWeights(0.5);
	printf("\n___________________________________\nnew Weights:\n");
	layer.Weights.printMatrix();*/

	return 0;
}