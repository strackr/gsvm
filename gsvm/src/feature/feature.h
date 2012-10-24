#ifndef FEATURE_H_
#define FEATURE_H_

#include "../math/numeric.h"
#include "../math/matrix.h"
#include "../math/random.h"

#define MAX_EIG_PROB_SIZE 1000
#define DEFAULT_PCA_THRESH 0.99

template<typename Matrix>
class FeatureProcessor {

public:
	void normalize(Matrix *samples);
	void randomize(Matrix *samples, label_id *labels);
	void reduceDimensionality(Matrix *samples, fvalue threshold = DEFAULT_PCA_THRESH);

};

#endif
