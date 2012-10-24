#ifndef STOP_H_
#define STOP_H_

#include "../math/numeric.h"

#define DEFAULT_EPSILON 0.001

#define EXPECTED_SV_NUMBER 15000

class StopCriterionStrategy {

public:
	virtual fvalue getThreshold(fvalue w2, fvalue tau, fvalue eps) = 0;
	virtual fvalue getEpsilon(fvalue tau, fvalue c) = 0;

	virtual ~StopCriterionStrategy();

};


class DefaultStopStrategy: public StopCriterionStrategy {

public:
	virtual fvalue getThreshold(fvalue w2, fvalue tau, fvalue eps);
	virtual fvalue getEpsilon(fvalue tau, fvalue c);

	virtual ~DefaultStopStrategy();

};

class MebStopStrategy: public StopCriterionStrategy {

public:
	virtual fvalue getThreshold(fvalue w2, fvalue tau, fvalue eps);
	virtual fvalue getEpsilon(fvalue tau, fvalue c);

	virtual ~MebStopStrategy();

};

#endif
