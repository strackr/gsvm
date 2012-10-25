/**************************************************************************
 * This file is part of gsvm, a Support Vector Machine solver.
 * Copyright (C) 2012 Robert Strack (strackr@vcu.edu), Vojislav Kecman
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 **************************************************************************/

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
