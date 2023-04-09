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
	
	/*
	printf("Inputs:\n");
	network.Inputs.printMatrix();
	printf("\nWeights:\n");
	network.printNetwork();
	printf("\nOutputs:\n");
	network.getNetworkOutput()->printMatrix();
	printf("\n___________________________________________________________\n\n\n\n\n");
	//*/

	bool networkError = false;
	while(!networkError){
		networkError = true;
		for(int i = 0; i < 4; i++){
			int i1 = i % 2;
			int i2 = i / 2;
			network.Inputs.setData(0, 0, i1);
			network.Inputs.setData(0, 1, i2);

			int r = 0;
			if((i1 == 1 && i2 == 0) || (i1 == 0 && i2 == 1)){
				r = 1;
			}
			network.forward();
			int error;
			Matrix Target = Matrix(1, 1, &error, 0);
			Target.setData(0, 0, r);
			network.Backpropogation(0.5, Target);

			printf("i1: %d i2: %d r: %d\n", i1, i2, r);

			if(network.getNetworkOutput()->getData(0, 0) != r){
				networkError = false;
			}
		}
	} 

	network.printNetwork();

	network.deleteNetwork();

	return 0;
}