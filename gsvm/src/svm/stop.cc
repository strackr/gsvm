#include "stop.h"

StopCriterionStrategy::~StopCriterionStrategy() {
}


fvalue DefaultStopStrategy::getThreshold(fvalue w2, fvalue tau, fvalue eps) {
	return w2 * pow2(1.0 - eps);
}

fvalue DefaultStopStrategy::getEpsilon(fvalue tau, fvalue c) {
	return DEFAULT_EPSILON;
}

DefaultStopStrategy::~DefaultStopStrategy() {
}


fvalue MebStopStrategy::getThreshold(fvalue w2, fvalue tau, fvalue eps) {
//	return w2 + (tau * (1 - pow2(1 + eps))) / 2.0;
	return (w2 + tau * (1 - pow2(1 + eps))) / 2.0;
}

fvalue MebStopStrategy::getEpsilon(fvalue tau, fvalue c) {
	fvalue d1dc = 1.0 / c;
	return (2e-6 / (tau - d1dc) + d1dc / EXPECTED_SV_NUMBER) / tau;
}

MebStopStrategy::~MebStopStrategy() {
}
