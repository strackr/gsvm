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
#include "../svm/solver_universal.h"
#include "../svm/validation.h"
#include "../feature/feature.h"

using namespace std;


enum StopCriterion {
	ADJMNORM,
	MNORM,
	MEB
};


template<typename Matrix>
class FeatureMatrixBuilder {

public:
	Matrix* getFeatureMatrix(list<map<feature_id, fvalue>*> &features,
			map<feature_id, feature_id> mappings);

};


template<typename Matrix, typename Strategy>
class DefaultSolverFactory {

private:
	istream *input;
	TrainParams params;

	StopCriterion strategy;

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
	DefaultSolverFactory(istream &input, TrainParams &params, StopCriterion strategy, bool reduceDim);

	UniversalSolver<GaussKernel, Matrix, Strategy>* getUniversalSolver();
	CrossValidationSolver<GaussKernel, Matrix, Strategy>* getCrossValidationSolver(
			quantity innerFolds, quantity outerFolds);

};

template<typename Matrix, typename Strategy>
DefaultSolverFactory<Matrix, Strategy>::DefaultSolverFactory(istream &input, TrainParams &params,
		StopCriterion strategy = ADJMNORM, bool reduceDim = false) :
		input(&input),
		params(params),
		strategy(strategy),
		reduceDim(reduceDim),
		matrixBuilder(new FeatureMatrixBuilder<Matrix>()) {
}

template<typename Matrix, typename Strategy>
UniversalSolver<GaussKernel, Matrix, Strategy>* DefaultSolverFactory<Matrix, Strategy>::getUniversalSolver() {
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

	return new UniversalSolver<GaussKernel, Matrix, Strategy>(
			labels, x, y, params, strategy);
}

template<typename Matrix, typename Strategy>
CrossValidationSolver<GaussKernel, Matrix, Strategy>* DefaultSolverFactory<Matrix, Strategy>::getCrossValidationSolver(
		quantity innerFolds, quantity outerFolds) {
	UniversalSolver<GaussKernel, Matrix, Strategy> *solver = getUniversalSolver();
	return new CrossValidationSolver<GaussKernel, Matrix, Strategy>(solver, innerFolds, outerFolds);
}

template<typename Matrix, typename Strategy>
map<label_id, string> DefaultSolverFactory<Matrix, Strategy>::getLabelMap(
		map<string, label_id> &labelIds) {
	map<label_id, string> labels;
	map<string, label_id>::iterator it;
	for (it = labelIds.begin(); it != labelIds.end(); it++) {
		labels[it->second] = it->first;
	}
	return labels;
}

template<typename Matrix, typename Strategy>
map<feature_id, feature_id> DefaultSolverFactory<Matrix, Strategy>::findOptimalFeatureMappings(
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
Matrix* DefaultSolverFactory<Matrix, Strategy>::getFeatureMatrix(list<map<feature_id, fvalue>*> &features,
		map<feature_id, feature_id> mappings) {
	return matrixBuilder->getFeatureMatrix(features, mappings);
}


template<typename Matrix, typename Strategy>
label_id* DefaultSolverFactory<Matrix, Strategy>::getLabelVector(list<label_id> &labels) {
	label_id *dataLabels = new label_id[labels.size()];
	quantity id = 0;

	list<label_id>::iterator lit;
	for (lit = labels.begin(); lit != labels.end(); lit++) {
		dataLabels[id++] = *lit;
	}
	return dataLabels;
}

template<typename Matrix, typename Strategy>
StopCriterionStrategy* DefaultSolverFactory<Matrix, Strategy>::getStopCriterion() {
	switch (strategy) {
	case MEB:
		return new MebStopStrategy();
	case MNORM:
		return new MnStopStrategy();
	case ADJMNORM:
	default:
		return new AdjustedMnStopStrategy();
	}
}

template<typename Matrix, typename Strategy>
map<feature_id, fvalue>* DefaultSolverFactory<Matrix, Strategy>::readFeatures(
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
Matrix* DefaultSolverFactory<Matrix, Strategy>::preprocess(Matrix *x, label_id *y) {
	FeatureProcessor<Matrix> proc;
	proc.normalize(x);
	proc.randomize(x, y);
	if (reduceDim) {
		proc.reduceDimensionality(x);
	}
	return x;
}

#endif
