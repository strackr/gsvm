#ifndef VIOLATION_H_
#define VIOLATION_H_

#include "../math/numeric.h"
#include "params.h"

enum ViolationCriterion {
	MDM,
	IMDM,
	GMDM
};

template<ViolationCriterion Type>
class ViolationEstimator {

public:
	ViolationEstimator(TrainParams &params);

	fvalue evaluateGoodness(fvalue au, fvalue av, fvalue uv, fvalue uw, fvalue vw, fvalue tau);

};

template<>
class ViolationEstimator<GMDM> {

	fvalue k;

public:
	ViolationEstimator(TrainParams &params);

	fvalue evaluateGoodness(fvalue au, fvalue av, fvalue uv, fvalue uw, fvalue vw, fvalue tau);

};


template<ViolationCriterion Type>
inline ViolationEstimator<Type>::ViolationEstimator(TrainParams& params) {
}

inline ViolationEstimator<GMDM>::ViolationEstimator(TrainParams& params) {
	k = params.gmdm.k;
}

template<>
inline fvalue ViolationEstimator<MDM>::evaluateGoodness(fvalue au, fvalue av,
		fvalue uv, fvalue uw, fvalue vw, fvalue tau) {
	return uw;
}

template<>
inline fvalue ViolationEstimator<IMDM>::evaluateGoodness(fvalue au, fvalue av,
		fvalue uv, fvalue uw, fvalue vw, fvalue tau) {
	fvalue beta = min(0.5 * (uw - vw) / (tau - uv), au);
	return -beta * (vw - uw + beta * (tau - uv));
}

inline fvalue ViolationEstimator<GMDM>::evaluateGoodness(fvalue au, fvalue av,
		fvalue uv, fvalue uw, fvalue vw, fvalue tau) {
	double k = 1.0;
	return (uw - vw) / pow(sqrt(tau - uv), k);
}

#endif
