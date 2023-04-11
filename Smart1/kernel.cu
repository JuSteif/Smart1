#include "Matrix.cuh"
#include "Layer.h"
#include "Network.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <time.h>

int main(int argc, char** argv) {

	printf("XOR Test \n");

	Network network(2);

	network.addLayer(4);
	network.addLayer(1);
	
	network.prepareNetwork();
	srand(time(0));
	
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
	int count = 0;
	while(!networkError){
		count++;
		networkError = true;

		float remember[4];
		for(int j = 0; j < 4; j++){
			int j1 = j % 2;
			int j2 = j / 2;
			if (j1 == 0) j1 = 0;
			if (j2 == 0) j2 = 0;
			int r2 = 0;
			if((j1 == 1 && j2 == 0) || (j1 == 0 && j2 == 1)){
				r2 = 1;
			}

			network.Inputs.setData(0, 0, j1);
			network.Inputs.setData(0, 1, j2);

			network.forward();
			remember[j] = network.getNetworkOutput()->getData(0, 0);

			if(round(network.getNetworkOutput()->getData(0, 0)) != r2){
				networkError = false;
			}
		}
		if(networkError){
			printf("0 0 %f\n", remember[0]);
			printf("0 1 %f\n", remember[1]);
			printf("1 0 %f\n", remember[2]);
			printf("1 1 %f\n", remember[3]);
			break;
		}

		int i = rand() % 4;

		int i1 = i % 2;
		int i2 = i / 2;
		if (i1 == 0) i1 = 0;
		if (i2 == 0) i2 = 0;
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

		/*printf("i1: %d i2: %d r: %d\n", i1, i2, r);
		network.printNetwork();*/

		printf("\n\n________________________________\nRound %d:\n", i);
		printf("i %d i1: %d i2: %d r: %d count %d\n", i, i1, i2, r, count);
		network.printNetwork();

		/*char con;
		scanf("%d", &con);*/
	} 

	network.deleteNetwork();

	return 0;
}