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
