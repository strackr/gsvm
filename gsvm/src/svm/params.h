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
