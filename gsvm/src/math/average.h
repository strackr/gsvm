#ifndef AVERAGE_H_
#define AVERAGE_H_

#include "numeric.h"

#define DEFAULT_EXP 0.9

class ExpMovingAverage {

	fvalue value;
	fvalue exp;

public:
	ExpMovingAverage(fvalue value, fvalue exp);

	void addObservation(fvalue obs);
	fvalue getValue();

};

inline ExpMovingAverage::ExpMovingAverage(fvalue value = 0, fvalue exp = DEFAULT_EXP) :
		value(value),
		exp(exp) {
}

inline void ExpMovingAverage::addObservation(fvalue obs) {
	value = (1.0 - exp) * value + exp * obs;
}

inline fvalue ExpMovingAverage::getValue() {
	return value;
}

#endif
