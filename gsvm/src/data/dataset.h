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

#ifndef DATASET_H_
#define DATASET_H_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

#include <set>
#include <map>
#include <list>
#include <vector>

#include "../math/numeric.h"

/**
 * Structure containing training samples for SVM problem.
 */
struct DataSet {

	map<label_id, string> labelNames;
	list<label_id> labels;
	list<map<feature_id, fvalue> > features;

	DataSet(map<label_id, string> labelNames, list<label_id> labels,
			list<map<feature_id, fvalue> > features) :
			labelNames(labelNames),
			labels(labels),
			features(features) {
	}

};

/**
 * Interface for data set factories.
 */
class DataSetFactory {

public:
	virtual ~DataSetFactory() {}

	virtual DataSet createDataSet() = 0;

};

/**
 * Data set factory capable of parsing data file in sparse format (libSVM format).
 */
class SparseFormatDataSetFactory: public DataSetFactory {

	istream& input;

	map<feature_id, fvalue> readFeatures(istream& lineStream);
	map<label_id, string> getLabelMap(map<string, label_id>& labelIds);

public:
	SparseFormatDataSetFactory(istream& input) :
			input(input) {
	}

	DataSet createDataSet();

};

#endif

