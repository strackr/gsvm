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

#include "../../src/data/dataset.h"
#include "../../src/data/solver_factory.h"
#include "../../src/svm/kernel.h"
#include "../../src/svm/strategy.h"
#include "../../src/svm/stop.h"

using namespace boost;
using namespace testing;

typedef AbstractSolver<GaussKernel, sfmatrix, SolverStrategy<MDM, FAIR> > DefaultSolver;
typedef BaseSolverFactory<sfmatrix, SolverStrategy<MDM, FAIR> > DefaultSolverFactory;
typedef Classifier<GaussKernel, sfmatrix> DefaultClassifier;


class ClassificationSuite: public TestWithParam<MulticlassApproach> {

};

INSTANTIATE_TEST_CASE_P(InstantiationName,
		ClassificationSuite, Values(ALL_AT_ONCE, PAIRWISE));

TEST_P(ClassificationSuite, ShouldPerformSimpleMulticlassClassification) {
	// given
	string desc =
			"class1 1:-2 2:1\n"
			"class1 1:-1 2:2\n"
			"class2 1:1 2:2\n"
			"class2 1:2 2:1\n"
			"class3 1:-1 2:-1\n"
			"class3 1:1 2:-1";

	TrainParams params;
	GaussKernel kernel(1.0);
	fvalue c = 1;

	StopCriterion stopCrit = ADJMNORM;
	MulticlassApproach multiApproach = GetParam();

	istringstream input(desc);
	DefaultSolverFactory reader(input, params, stopCrit, multiApproach);

	// when
	scoped_ptr<DefaultSolver> solver(reader.getSolver());
	solver->setKernelParams(c, kernel);
	solver->train();

	scoped_ptr<DefaultClassifier> classifier(solver->getClassifier());

	// then
	label_id *labels = solver->getLabels();
	ASSERT_TRUE(labels[0] == classifier->classify(0));
	ASSERT_TRUE(labels[1] == classifier->classify(1));
	ASSERT_TRUE(labels[2] == classifier->classify(2));
	ASSERT_TRUE(labels[3] == classifier->classify(3));
	ASSERT_TRUE(labels[4] == classifier->classify(4));
	ASSERT_TRUE(labels[5] == classifier->classify(5));

	multiset<label_id> labelSet(labels, labels + 6);
	ASSERT_EQ(2, labelSet.count(0));
	ASSERT_EQ(2, labelSet.count(1));
	ASSERT_EQ(2, labelSet.count(2));
}
