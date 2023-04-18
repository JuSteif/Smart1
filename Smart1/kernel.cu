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

	network.addLayer(2, STEP_FUNCTION);
	network.addLayer(1, STEP_FUNCTION);
	
	network.prepareNetwork();
	srand(time(0));

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

			float test2[2] = {j1, j2};
			network.setInput(test2);

			network.forward();
			remember[j] = network.getNetworkOutput()->getData(0, 0);

			int size;
			float* res = network.getOutputArray(&size);

			if(round(res[0]) != r2){
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

		float test[2] = {i1, i2};
		network.setInput(test);

		float r = 0;
		if((i1 == 1 && i2 == 0) || (i1 == 0 && i2 == 1)){
			r = 1;
		}
		network.forward();
		network.setTarget(&r);
		network.Backpropogation(0.05);

		/*printf("i1: %d i2: %d r: %d\n", i1, i2, r);
		network.printNetwork();*/

		printf("\n\n________________________________\n\n");
		printf("i %d i1: %d i2: %d r: %d count %d\n", i, i1, i2, r, count);
		network.printNetwork();

	} 
	network.safeNetwork("C:\\Users\\seife\\OneDrive\\Desktop\\KI\\weights");

	network.deleteNetwork();

	return 0;
}