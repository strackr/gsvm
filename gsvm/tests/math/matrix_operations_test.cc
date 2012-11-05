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
#include <boost/smart_ptr.hpp>

#include "matrix_utils.h"

#include "../../src/math/matrix.h"
#include "../../src/io/dataset.h"
#include "../../src/io/solver_factory.h"

#include "matrix_utils.h"

using namespace boost;
using namespace testing;

template<typename Matrix>
class MatrixOperationsSuite: public Test {

public:
	SimpleMatrixFactory<Matrix> sfmFactory;

};

typedef Types<sfmatrix, dfmatrix> MatrixTypes;
TYPED_TEST_CASE(MatrixOperationsSuite, MatrixTypes);

TYPED_TEST(MatrixOperationsSuite, shouldEvaluateDotProductForOneSample) {
	// given
	string desc = "label 1:0.5";

	scoped_ptr<TypeParam> matrix(this->sfmFactory.create(desc));
	MatrixEvaluator<TypeParam> evaluator(matrix.get());

	// when
	fvalue dot = evaluator.dot(0, 0);

	// then
	ASSERT_DOUBLE_EQ(0.25, dot);
}

TYPED_TEST(MatrixOperationsSuite, shouldEvaluateDotProductForTwoSamples) {
	// given
	string desc =
			"label1 1:0.5\n"
			"label2 1:0.2";

	scoped_ptr<TypeParam> matrix(this->sfmFactory.create(desc));
	MatrixEvaluator<TypeParam> evaluator(matrix.get());

	// when
	fvalue dot10 = evaluator.dot(1, 0);
	fvalue dot01 = evaluator.dot(0, 1);

	// then
	ASSERT_DOUBLE_EQ(0.1, dot01);
	ASSERT_DOUBLE_EQ(0.1, dot10);
}

TYPED_TEST(MatrixOperationsSuite, shouldEvaluateSelfDistance) {
	// given
	string desc = "label 1:-1.5";

	scoped_ptr<TypeParam> matrix(this->sfmFactory.create(desc));
	MatrixEvaluator<TypeParam> evaluator(matrix.get());

	// when
	fvalue dot = evaluator.dist(0, 0);

	// then
	ASSERT_DOUBLE_EQ(0.0, dot);
}
