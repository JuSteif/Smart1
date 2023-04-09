#include "Matrix.cuh"
#include "Layer.h"
#include "Network.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

int main(int argc, char** argv) {

	Network network(2);

	network.addLayer(2);
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
	printf("\n___________________________________________________________\n\n\n\n\n");
	//*/

	int error;
	Matrix Target = Matrix(1, 1, &error, 2, 1);
	network.Backpropogation(0.5, Target);


	network.deleteNetwork();

	return 0;
}