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

TYPED_TEST(MatrixOperationsSuite, ShouldEvaluateDotProductForOneSample) {
	// given
	string desc = "label 1:0.5";

	scoped_ptr<TypeParam> matrix(this->sfmFactory.create(desc));
	MatrixEvaluator<TypeParam> evaluator(matrix.get());

	// when
	fvalue dot = evaluator.dot(0, 0);

	// then
	ASSERT_DOUBLE_EQ(0.25, dot);
}

TYPED_TEST(MatrixOperationsSuite, ShouldEvaluateDotProductForTwoSamples) {
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

TYPED_TEST(MatrixOperationsSuite, ShouldEvaluateSelfDistance) {
	// given
	string desc = "label 1:-1.5";

	scoped_ptr<TypeParam> matrix(this->sfmFactory.create(desc));
	MatrixEvaluator<TypeParam> evaluator(matrix.get());

	// when
	fvalue dist = evaluator.dist(0, 0);

	// then
	ASSERT_DOUBLE_EQ(0.0, dist);
}

TYPED_TEST(MatrixOperationsSuite, ShouldSwapTwoSamples) {
	// given
	string desc =
			"label1 1:0.5\n"
			"label1 1:0.7\n"
			"label2 1:0.2";

	scoped_ptr<TypeParam> matrix(this->sfmFactory.create(desc));
	MatrixEvaluator<TypeParam> evaluator(matrix.get());

	// when
	fvalue norm00Before = evaluator.dot(0, 0);
	fvalue norm11Before = evaluator.dot(1, 1);
	fvalue norm22Before = evaluator.dot(2, 2);
	evaluator.swapSamples(0, 2);
	fvalue norm00After = evaluator.dot(0, 0);
	fvalue norm11After = evaluator.dot(1, 1);
	fvalue norm22After = evaluator.dot(2, 2);

	// then
	ASSERT_TRUE(norm00Before == norm22After);
	ASSERT_TRUE(norm11Before == norm11After);
	ASSERT_TRUE(norm22Before == norm00After);
}

TYPED_TEST(MatrixOperationsSuite, ShouldCalculateMultipleDist) {
	// given
	string desc =
			"label1 1:0.3\n"
			"label2 2:0.5\n"
			"label1 1:0.6 2:0.4";

	fvector* buffer = fvector_alloc(3);
	scoped_ptr<TypeParam> matrix(this->sfmFactory.create(desc));
	MatrixEvaluator<TypeParam> evaluator(matrix.get());

	// when
	evaluator.dist(2, 0, 2, buffer);
	fvalue dist20 = fvector_get(buffer, 0);
	fvalue dist21 = fvector_get(buffer, 1);

	fvector_free(buffer);

	// then
	ASSERT_DOUBLE_EQ(0.25, dist20);
	ASSERT_DOUBLE_EQ(0.37, dist21);
}

TYPED_TEST(MatrixOperationsSuite, ShouldCalculateMultipleDistWithMappings) {
	// given
	string desc =
			"label1 1:0.6 2:0.4\n"
			"label2 2:0.5\n"
			"label1 1:0.3";
	sample_id mappings[] = {0, 2, 1};

	fvector* buffer = fvector_alloc(3);
	scoped_ptr<TypeParam> matrix(this->sfmFactory.create(desc));
	MatrixEvaluator<TypeParam> evaluator(matrix.get());

	// when
	evaluator.dist(0, 1, 3, mappings, buffer);
	fvalue dist1 = fvector_get(buffer, 1);
	fvalue dist2 = fvector_get(buffer, 2);

	fvector_free(buffer);

	// then
	ASSERT_DOUBLE_EQ(0.25, dist1);
	ASSERT_DOUBLE_EQ(0.37, dist2);
}
