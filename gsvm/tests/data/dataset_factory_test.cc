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

#include "../math/matrix_utils.h"

#include "../../src/data/dataset.h"
#include "../../src/data/solver_factory.h"

using namespace boost;

TEST(DataSetFactorySuite, ShouldCreateCompleteDataSet) {
	// given
	string desc =
			"label_a 1:0.5\n"
			"label_b 10:-1\n"
			"label_a 1:1 2:0.1";

	istringstream stream(desc);
	SparseFormatDataSetFactory factory(stream);

	// when
	DataSet dataSet = factory.createDataSet();

	// then
	vector<label_id> labels(dataSet.labels.begin(), dataSet.labels.end());
	vector<map<feature_id, fvalue> > features(
			dataSet.features.begin(), dataSet.features.end());

	ASSERT_EQ(2, dataSet.labelNames.size());
	ASSERT_EQ("label_a", dataSet.labelNames[labels[0]]);
	ASSERT_EQ("label_b", dataSet.labelNames[labels[1]]);

	ASSERT_EQ(3, features.size());

	ASSERT_EQ(1, features[0].size());
	ASSERT_EQ(0.5, features[0][1]);

	ASSERT_EQ(1, features[1].size());
	ASSERT_EQ(-1.0, features[1][10]);

	ASSERT_EQ(2, features[2].size());
	ASSERT_EQ(1.0, features[2][1]);
	ASSERT_EQ(0.1, features[2][2]);

	ASSERT_EQ(3, labels.size());
	ASSERT_TRUE(labels[0] == labels[2]);
	ASSERT_TRUE(labels[0] != labels[1]);
}

TEST(DataSetFactorySuite, ShouldCreateCorrectSparseMatrix) {
	// given
	string desc =
			"label1 1:0.6 2:0.4\n"
			"label2 2:0.5\n"
			"label1 1:0.3";

	istringstream stream(desc);
	SparseFormatDataSetFactory factory(stream);
	FeatureMatrixBuilder<sfmatrix> builder;

	map<feature_id, feature_id> mappings;
	mappings[1] = 0;
	mappings[2] = 1;

	// when
	DataSet dataSet = factory.createDataSet();
	scoped_ptr<sfmatrix> matrix(
			builder.getFeatureMatrix(dataSet.features, mappings));

	// then
	ASSERT_EQ(2, matrix->width);
	ASSERT_EQ(3, matrix->height);

	ASSERT_EQ(0.6, matrix->values[0]);
	ASSERT_EQ(0.4, matrix->values[1]);
	ASSERT_EQ(0.5, matrix->values[3]);
	ASSERT_EQ(0.3, matrix->values[5]);

	ASSERT_EQ(0, matrix->features[0]);
	ASSERT_EQ(1, matrix->features[1]);
	ASSERT_EQ(INVALID_FEATURE_ID, matrix->features[2]);
	ASSERT_EQ(1, matrix->features[3]);
	ASSERT_EQ(INVALID_FEATURE_ID, matrix->features[4]);
	ASSERT_EQ(0, matrix->features[5]);
	ASSERT_EQ(INVALID_FEATURE_ID, matrix->features[6]);

	ASSERT_EQ(0, matrix->offsets[0]);
	ASSERT_EQ(3, matrix->offsets[1]);
	ASSERT_EQ(5, matrix->offsets[2]);
}
