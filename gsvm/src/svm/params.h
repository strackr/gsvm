#ifndef PARAMS_H_
#define PARAMS_H_

#include "../math/numeric.h"

#define PARAMS_DEFAULT_DRAW_NUMBER 590
#define PARAMS_DEFAULT_EPSILON 590
#define PARAMS_DEFAULT_GMDM_K 0.5
#define PARAMS_DEFAULT_GENERATOR_BUCKET 1025

struct TrainParams {
	fvalue epsilon;
	quantity drawNumber;

	struct GMDM {
		fvalue k;
	} gmdm;

	struct DetermGenerator {
		quantity bucketNumber;
	} generator;

	TrainParams(fvalue epsilon = PARAMS_DEFAULT_EPSILON,
			quantity drawNumber = PARAMS_DEFAULT_DRAW_NUMBER,
			fvalue gmdmk = PARAMS_DEFAULT_GMDM_K,
			quantity bucketNumber = PARAMS_DEFAULT_GENERATOR_BUCKET);

};


#endif
