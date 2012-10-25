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

#ifndef DATA_HPP_
#define DATA_HPP_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

#include <set>
#include <map>
#include <list>
#include <vector>

#include "../logging/log.h"
#include "../svm/solver.h"
#include "../svm/cv.h"
#include "../feature/feature.h"

using namespace std;


enum StopCriterion {
	DEFAULT,
	MEB
};


template<typename Matrix>
class FeatureMatrixBuilder {

public:
	Matrix* getFeatureMatrix(list<map<feature_id, fvalue>*> &features,
			map<feature_id, feature_id> mappings);

};


template<typename Kernel, typename Matrix, typename Strategy>
class SolverBuilder {

public:
	virtual Solver<Kernel, Matrix, Strategy>* getSolver() = 0;
	virtual ~SolverBuilder();

};

template<typename Kernel, typename Matrix, typename Strategy>
SolverBuilder<Kernel, Matrix, Strategy>::~SolverBuilder() {
}


template<typename Matrix, typename Strategy>
class DefaultSolverBuilder: public SolverBuilder<GaussKernel, Matrix, Strategy> {

private:
	istream *input;
	TrainParams params;

	StopCriterion strategy;

	quantity innerFolds;
	quantity outerFolds;

	bool reduceDim;

	FeatureMatrixBuilder<Matrix> *matrixBuilder;

	map<feature_id, fvalue>* readFeatures(istream &lineStream);

	map<label_id, string> getLabelMap(map<string, label_id> &labelIds);
	map<feature_id, feature_id> findOptimalFeatureMappings(list<map<feature_id, fvalue>*> &features);
	Matrix* getFeatureMatrix(list<map<feature_id, fvalue>*> &features,
			map<feature_id, feature_id> mappings);
	label_id* getLabelVector(list<label_id> &labels);
	StopCriterionStrategy* getStopCriterion();

	Matrix* preprocess(Matrix *x, label_id *y);

public:
	DefaultSolverBuilder(istream &input, TrainParams &params, StopCriterion strategy,
			quantity innerFolds, quantity outerFolds, bool reduceDim);
	virtual ~DefaultSolverBuilder();

	virtual Solver<GaussKernel, Matrix, Strategy>* getSolver();

};

template<typename Matrix, typename Strategy>
DefaultSolverBuilder<Matrix, Strategy>::DefaultSolverBuilder(istream &input, TrainParams &params,
		StopCriterion strategy = DEFAULT, quantity innerFolds = 1, quantity outerFolds = 1,
		bool reduceDim = false) :
		input(&input),
		params(params),
		strategy(strategy),
		innerFolds(innerFolds),
		outerFolds(outerFolds),
		reduceDim(reduceDim),
		matrixBuilder(new FeatureMatrixBuilder<Matrix>()) {
}

template<typename Matrix, typename Strategy>
DefaultSolverBuilder<Matrix, Strategy>::~DefaultSolverBuilder() {
}

template<typename Matrix, typename Strategy>
Solver<GaussKernel, Matrix, Strategy>* DefaultSolverBuilder<Matrix, Strategy>::getSolver() {
	map<string, label_id> labelIds;
	list<label_id> sampleLabels;
	list<map<feature_id, fvalue>*> sampleFeatures;

	label_id labelCounter = 0;

	while (!input->eof()) {
		string line;
		getline(*input, line);
		stringstream lineStream(line);
		if (!line.empty()) {
			// read label
			string label;
			lineStream >> label;
			if (!labelIds.count(label)) {
				labelIds[label] = labelCounter++;
			}
			sampleLabels.push_back(labelIds[label]);

			// read features
			sampleFeatures.push_back(readFeatures(lineStream));
		}
	}

	map<label_id, string> labels = getLabelMap(labelIds);
	map<feature_id, feature_id> mappings = findOptimalFeatureMappings(sampleFeatures);
	Matrix *x = getFeatureMatrix(sampleFeatures, mappings);
	label_id *y = getLabelVector(sampleLabels);

	// free the memory
	list<map<feature_id, fvalue>*>::iterator lit;
	for (lit = sampleFeatures.begin(); lit != sampleFeatures.end(); lit++) {
		delete *lit;
	}

	x = preprocess(x, y);

	StopCriterionStrategy *strategy = getStopCriterion();

	return new CrossSolver<GaussKernel, Matrix, Strategy>(
			labels, x, y, params, strategy, innerFolds, outerFolds);
//	return new Solver<GaussKernel>(labels, x, y, params);
}

template<typename Matrix, typename Strategy>
map<label_id, string> DefaultSolverBuilder<Matrix, Strategy>::getLabelMap(
		map<string, label_id> &labelIds) {
	map<label_id, string> labels;
	map<string, label_id>::iterator it;
	for (it = labelIds.begin(); it != labelIds.end(); it++) {
		labels[it->second] = it->first;
	}
	return labels;
}

template<typename Matrix, typename Strategy>
map<feature_id, feature_id> DefaultSolverBuilder<Matrix, Strategy>::findOptimalFeatureMappings(
		list<map<feature_id, fvalue>*> &features) {
	quantity dimension = 0;
	map<feature_id, fvalue> fmax;
	map<feature_id, fvalue> fmin;

	list<map<feature_id, fvalue>*>::iterator lit;
	for (lit = features.begin(); lit != features.end(); lit++) {
		map<feature_id, fvalue>::iterator mit;
		for (mit = (*lit)->begin(); mit != (*lit)->end(); mit++) {
			dimension = max((quantity) mit->first, dimension);
			fmax[mit->first] = max(fmax[mit->first], mit->second);
			fmin[mit->first] = min(fmin[mit->first], mit->second);
		}
	}

	feature_id available = 0;
	map<feature_id, feature_id> mappings;
	for (feature_id feat = 0; feat <= dimension; feat++) {
		if (fmax[feat] == fmin[feat]) {
			for (lit = features.begin(); lit != features.end(); lit++) {
				(*lit)->erase(feat);
			}
		} else {
			mappings[feat] = available++;
		}
	}

	quantity featureNum = 0;
	for (lit = features.begin(); lit != features.end(); lit++) {
		featureNum += (*lit)->size();
	}

	double density = 100.0 * featureNum / (features.size() * mappings.size());
	logger << format("data density: %.2f[%%]") % density << endl;
	return mappings;
}

template<typename Matrix, typename Strategy>
Matrix* DefaultSolverBuilder<Matrix, Strategy>::getFeatureMatrix(list<map<feature_id, fvalue>*> &features,
		map<feature_id, feature_id> mappings) {
	return matrixBuilder->getFeatureMatrix(features, mappings);
}


template<typename Matrix, typename Strategy>
label_id* DefaultSolverBuilder<Matrix, Strategy>::getLabelVector(list<label_id> &labels) {
	label_id *dataLabels = new label_id[labels.size()];
	quantity id = 0;

	list<label_id>::iterator lit;
	for (lit = labels.begin(); lit != labels.end(); lit++) {
		dataLabels[id++] = *lit;
	}
	return dataLabels;
}

template<typename Matrix, typename Strategy>
StopCriterionStrategy* DefaultSolverBuilder<Matrix, Strategy>::getStopCriterion() {
	StopCriterionStrategy *stop;
	if (strategy == MEB) {
		stop = new MebStopStrategy();
	} else {
		stop = new DefaultStopStrategy();
	}
	return stop;
}

template<typename Matrix, typename Strategy>
map<feature_id, fvalue>* DefaultSolverBuilder<Matrix, Strategy>::readFeatures(
		istream &lineStream) {
	map<feature_id, fvalue> *features = new map<feature_id, fvalue>();
	while (!lineStream.eof()) {
		feature_id featureId;
		char delim;
		fvalue value;
		lineStream >> featureId >> delim >> value;
		if (value != 0.0) {
			(*features)[featureId] = value;
		}
	}
	return features;
}

template<typename Matrix, typename Strategy>
Matrix* DefaultSolverBuilder<Matrix, Strategy>::preprocess(Matrix *x, label_id *y) {
	FeatureProcessor<Matrix> proc;
	proc.normalize(x);
	proc.randomize(x, y);
	if (reduceDim) {
		proc.reduceDimensionality(x);
	}
	return x;
}

#endif
