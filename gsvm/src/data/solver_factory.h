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

#ifndef SOLVER_FACTORY_H_
#define SOLVER_FACTORY_H_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

#include <set>
#include <map>
#include <list>
#include <vector>

#include "dataset.h"
#include "../logging/log.h"
#include "../svm/universal_solver.h"
#include "../svm/pairwise_solver.h"
#include "../svm/validation.h"
#include "../svm/strategy.h"
#include "../feature/feature.h"

using namespace std;


enum StopCriterion {
	ADJMNORM,
	MNORM,
	MEB
};

enum MulticlassApproach {
	ALL_AT_ONCE,
	PAIRWISE
};


template<typename Matrix>
class FeatureMatrixBuilder {

public:
	Matrix* getFeatureMatrix(list<map<feature_id, fvalue> >& features,
			map<feature_id, feature_id>& mappings);

	Matrix* getFeatureMatrix(list<map<feature_id, fvalue> >& features);

};

template<typename Matrix>
inline Matrix* FeatureMatrixBuilder<Matrix>::getFeatureMatrix(
		list<map<feature_id, fvalue> >& features) {
	set<feature_id> uniqFeatures;
	list<map<feature_id, fvalue> >::iterator fit;
	for (fit = features.begin(); fit != features.end(); fit++) {
		map<feature_id, fvalue>::iterator sit;
		for (sit = fit->begin(); sit != fit->end(); sit++) {
			uniqFeatures.insert(sit->first);
		}
	}

	map<feature_id, feature_id> optimalMapping;

	feature_id current = 0;
	set<feature_id>::iterator ufit;
	for (ufit = uniqFeatures.begin(); ufit != uniqFeatures.end(); ufit++) {
		optimalMapping[*ufit] = current++;
	}
	return getFeatureMatrix(features, optimalMapping);
}


template<typename Matrix = sfmatrix, typename Strategy = SolverStrategy<MDM, FAIR> >
class BaseSolverFactory {

private:
	istream& input;

	TrainParams params;
	StopCriterion strategy;
	MulticlassApproach multiclass;

	bool reduceDim;

	FeatureMatrixBuilder<Matrix> *matrixBuilder;

	map<feature_id, feature_id> findOptimalFeatureMappings(
			list<map<feature_id, fvalue> >& features);
	label_id* getLabelVector(list<label_id>& labels);
	StopCriterionStrategy* getStopCriterion();

	Matrix* preprocess(Matrix *x, label_id *y);

	AbstractSolver<GaussKernel, Matrix, Strategy>* createSolver(
			MulticlassApproach type, map<label_id,string> labels,
			Matrix* x, label_id* y, TrainParams& params,
			StopCriterionStrategy* strategy);

public:
	BaseSolverFactory(istream& input, TrainParams& params,
			StopCriterion strategy, MulticlassApproach multiclass, bool reduceDim);

	AbstractSolver<GaussKernel, Matrix, Strategy>* getSolver();
	CrossValidationSolver<GaussKernel, Matrix, Strategy>* getCrossValidationSolver(
			quantity innerFolds, quantity outerFolds);

};

template<typename Matrix, typename Strategy>
BaseSolverFactory<Matrix, Strategy>::BaseSolverFactory(istream& input,
		TrainParams& params= TrainParams(),
		StopCriterion strategy = ADJMNORM,
		MulticlassApproach multiclass = ALL_AT_ONCE,
		bool reduceDim = false) :
		input(input),
		params(params),
		strategy(strategy),
		multiclass(multiclass),
		reduceDim(reduceDim),
		matrixBuilder(new FeatureMatrixBuilder<Matrix>()) {
}

template<typename Matrix, typename Strategy>
AbstractSolver<GaussKernel, Matrix, Strategy>* BaseSolverFactory<Matrix, Strategy>::createSolver(
		MulticlassApproach type, map<label_id, string> labels,
		Matrix* x, label_id* y, TrainParams& params, StopCriterionStrategy* strategy) {
	AbstractSolver<GaussKernel, Matrix, Strategy> *solver = NULL;
	if (type == PAIRWISE) {
		solver = new PairwiseSolver<GaussKernel, Matrix, Strategy>(
					labels, x, y, params, strategy);
	} else {
		solver = new UniversalSolver<GaussKernel, Matrix, Strategy>(
					labels, x, y, params, strategy);
	}
	return solver;
}

template<typename Matrix, typename Strategy>
AbstractSolver<GaussKernel, Matrix, Strategy>* BaseSolverFactory<Matrix, Strategy>::getSolver() {
	SparseFormatDataSetFactory dataSetFactory(input);
	DataSet dataSet = dataSetFactory.createDataSet();

	map<feature_id, feature_id> mappings = findOptimalFeatureMappings(dataSet.features);
	Matrix *x = matrixBuilder->getFeatureMatrix(dataSet.features, mappings);
	label_id *y = getLabelVector(dataSet.labels);

	x = preprocess(x, y);

	StopCriterionStrategy *strategy = getStopCriterion();

	return createSolver(multiclass, dataSet.labelNames, x, y, params, strategy);
}

template<typename Matrix, typename Strategy>
CrossValidationSolver<GaussKernel, Matrix, Strategy>* BaseSolverFactory<Matrix, Strategy>::getCrossValidationSolver(
		quantity innerFolds, quantity outerFolds) {
	AbstractSolver<GaussKernel, Matrix, Strategy> *solver = getSolver();
	return new CrossValidationSolver<GaussKernel, Matrix, Strategy>(
			solver, innerFolds, outerFolds);
}

template<typename Matrix, typename Strategy>
map<feature_id, feature_id> BaseSolverFactory<Matrix, Strategy>::findOptimalFeatureMappings(
		list<map<feature_id, fvalue> >& features) {
	quantity dimension = 0;
	map<feature_id, fvalue> fmax;
	map<feature_id, fvalue> fmin;

	list<map<feature_id, fvalue> >::iterator lit;
	for (lit = features.begin(); lit != features.end(); lit++) {
		map<feature_id, fvalue>::iterator mit;
		for (mit = lit->begin(); mit != lit->end(); mit++) {
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
				lit->erase(feat);
			}
		} else {
			mappings[feat] = available++;
		}
	}

	quantity featureNum = 0;
	for (lit = features.begin(); lit != features.end(); lit++) {
		featureNum += lit->size();
	}

	double density = 100.0 * featureNum / (features.size() * mappings.size());
	logger << format("data density: %.2f[%%]") % density << endl;
	return mappings;
}

template<typename Matrix, typename Strategy>
label_id* BaseSolverFactory<Matrix, Strategy>::getLabelVector(
		list<label_id>& labels) {
	label_id *dataLabels = new label_id[labels.size()];
	quantity id = 0;

	list<label_id>::iterator lit;
	for (lit = labels.begin(); lit != labels.end(); lit++) {
		dataLabels[id++] = *lit;
	}
	return dataLabels;
}

template<typename Matrix, typename Strategy>
StopCriterionStrategy* BaseSolverFactory<Matrix, Strategy>::getStopCriterion() {
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
Matrix* BaseSolverFactory<Matrix, Strategy>::preprocess(Matrix *x, label_id *y) {
	FeatureProcessor<Matrix> proc;
	proc.normalize(x);
	proc.randomize(x, y);
	if (reduceDim) {
		proc.reduceDimensionality(x);
	}
	return x;
}

#endif
