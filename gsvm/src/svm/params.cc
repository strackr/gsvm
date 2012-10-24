#include "params.h"

TrainParams::TrainParams(fvalue epsilon, quantity drawNumber, fvalue gmdmk, quantity bucketNumber) :
		epsilon(epsilon),
		drawNumber(drawNumber) {
	gmdm.k = gmdmk;
	generator.bucketNumber = bucketNumber;
}
