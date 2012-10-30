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

#include "stop.h"

StopCriterionStrategy::~StopCriterionStrategy() {
}


fvalue MnStopStrategy::getThreshold(fvalue w2, fvalue tau, fvalue eps) {
	return w2 * pow2(1.0 - eps);
}

fvalue MnStopStrategy::getEpsilon(fvalue tau, fvalue c) {
	return DEFAULT_EPSILON;
}

MnStopStrategy::~MnStopStrategy() {
}


fvalue AdjustedMnStopStrategy::getThreshold(fvalue w2, fvalue tau, fvalue eps) {
	return w2 - eps * sqrt(w2 * tau);
}

AdjustedMnStopStrategy::~AdjustedMnStopStrategy() {
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
