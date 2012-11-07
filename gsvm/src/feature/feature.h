/**************************************************************************
 * This file is part of gsvm, a Support Vector Machine solver.
 * Copyright (C) 2012 Robert Strack (strackr@vcu.edu), Vojislav Kecman
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 **************************************************************************/

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
