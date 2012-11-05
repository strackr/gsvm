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

#include <gtest/gtest.h>

#include "../../src/math/matrix.h"

#include "../../src/io/dataset.h"
#include "../../src/io/solver_factory.h"

TEST(SparseMatrixSuite, shouldEvaluateDotProductForOneSample) {
	// given
	list<map<feature_id, fvalue> > samples;
	map<feature_id, fvalue> sample;
	sample[0] = 0.5;
	samples.push_back(sample);

	map<feature_id, feature_id> mappings;
	mappings[0] = 0;

	FeatureMatrixBuilder<sfmatrix> builder;
	sfmatrix* matrix = builder.getFeatureMatrix(samples, mappings);
	MatrixEvaluator<sfmatrix> evaluator(matrix);

	// when
	fvalue dot = evaluator.dot(0, 0);

	// then
	ASSERT_DOUBLE_EQ(0.25, dot);
}
