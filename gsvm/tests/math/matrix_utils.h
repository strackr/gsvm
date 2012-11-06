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

#ifndef MATRIX_UTILS_H_
#define MATRIX_UTILS_H_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

#include <set>
#include <map>
#include <list>
#include <vector>

#include <algorithm>

#include "../../src/math/numeric.h"
#include "../../src/data/dataset.h"
#include "../../src/data/solver_factory.h"

template<typename Matrix>
class SimpleMatrixFactory {

public:
	Matrix* create(string& desc);

};

template<typename Matrix>
Matrix* SimpleMatrixFactory<Matrix>::create(string& desc) {
	istringstream stream(desc);
	SparseFormatDataSetFactory factory(stream);
	DataSet dataSet = factory.createDataSet();
	FeatureMatrixBuilder<Matrix> matrixBuilder;
	map<feature_id, feature_id> identity;
	map<feature_id, string>::iterator it;
	for (it = dataSet.labelNames.begin(); it != dataSet.labelNames.end(); it++) {
		identity[it->first] = it->first;
	}
	return matrixBuilder.getFeatureMatrix(dataSet.features, identity);
}

#endif
