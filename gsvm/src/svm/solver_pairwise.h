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

#ifndef SOLVER_PAIRWISE_H_
#define SOLVER_PAIRWISE_H_

#include "solver_common.h"

template<typename K, typename V>
struct PairValueComparator {

	bool operator()(pair<K, V> pair1, pair<K, V> pair2) {
		return pair1.second > pair2.second;
	}

};


struct PairwiseTrainingResult {

	pair<label_id, label_id> labels;
	vector<fvalue> alphas;
	fvalue bias;
	vector<sample_id> samples;
	quantity size;

	PairwiseTrainingResult(const pair<label_id, label_id>& labels, quantity size) :
			labels(labels),
			alphas(vector<fvalue>()),
			bias(0),
			samples(vector<sample_id>()),
			size(0) {
		alphas.reserve(size);
		samples.reserve(size);
	}

	void clear() {
		alphas.clear();
		bias = 0;
		samples.clear();
	}

};


template<typename Kernel, typename Matrix, typename Strategy>
class PairwiseSolver: public AbstractSolver<Kernel, Matrix, Strategy> {

	vector<PairwiseTrainingResult> trainings;

	void sortLabels(label_id *sampleLabels, quantity size, pair<label_id, label_id>& labels);

public:
	PairwiseSolver(map<label_id, string> labelNames, Matrix *samples,
			label_id *labels, TrainParams &params,
			StopCriterionStrategy *stopStrategy);
	virtual ~PairwiseSolver();

	void train();
	UniversalClassifier<Kernel, Matrix>* getClassifier();

};

template<typename Kernel, typename Matrix, typename Strategy>
PairwiseSolver<Kernel, Matrix, Strategy>::PairwiseSolver(
		map<label_id, string> labelNames, Matrix *samples,
		label_id *labels, TrainParams &params,
		StopCriterionStrategy *stopStrategy) :
		AbstractSolver<Kernel, Matrix, Strategy>(labelNames, samples, labels, params, stopStrategy),
		trainings(vector<PairwiseTrainingResult>()) {
	label_id maxLabel = labelNames.size();
	vector<quantity> labelSizes(maxLabel, 0);
	for (sample_id sample = 0; sample < this->size; sample++) {
		labelSizes[this->labels[sample]]++;
	}

	// sort
	vector<pair<label_id, quantity> > sizes(maxLabel);
	for (label_id label = 0; label < maxLabel; label++) {
		sizes[label] = pair<label_id, quantity>(label, labelSizes[label]);
	}
	sort(sizes.begin(), sizes.end(), PairValueComparator<label_id, quantity>());

	vector<pair<label_id, quantity> >::iterator it1;
	for (it1 = sizes.begin(); it1 < sizes.end(); it1++) {
		vector<pair<label_id, quantity> >::iterator it2;
		for (it2 = it1 + 1; it2 < sizes.end(); it2++) {
			pair<label_id, label_id> labels(it1->first, it2->first);
			quantity size = it1->second + it2->second;
			trainings.push_back(PairwiseTrainingResult(labels, size));
		}
	}
}

template<typename Kernel, typename Matrix, typename Strategy>
PairwiseSolver<Kernel, Matrix, Strategy>::~PairwiseSolver() {
}

template<typename Kernel, typename Matrix, typename Strategy>
UniversalClassifier<Kernel, Matrix>* PairwiseSolver<Kernel, Matrix, Strategy>::getClassifier() {
	// XXX create pairwise classifier
	return this->buildClassifier();
}

template<typename Kernel, typename Matrix, typename Strategy>
void PairwiseSolver<Kernel, Matrix, Strategy>::train() {
	vector<PairwiseTrainingResult>::iterator it;
	for (it = trainings.begin(); it != trainings.end(); it++) {
		pair<label_id, label_id> trainingPair = it->labels;
		sortLabels(this->labels, this->currentSize, trainingPair);
		this->trainForCache(this->cache);

		fvalue* resultAlphas = it->alphas.data();
		fvalue* cacheAlphas = this->cache->getAlphas()->data;
		fvalue bias = 0;
		sample_id* cacheSamples = this->cache->getBackwardOrder();
		sample_id* resultSamples = it->samples.data();
		quantity svNumber  = this->cache->getSVNumber();
		for (quantity i = 0; i < svNumber; i++) {
			fvalue alpha = cacheAlphas[i];
			resultAlphas[i] = alpha;
			resultSamples[i] = cacheSamples[i];
			bias += alpha * (this->labels[i] == trainingPair.first ? 1.0 : -1.0);
		}
		it->bias = bias;
		it->size = svNumber;
	}
}

template<typename Kernel, typename Matrix, typename Strategy>
void PairwiseSolver<Kernel, Matrix, Strategy>::sortLabels(
		label_id *sampleLabels, quantity size, pair<label_id, label_id>& labels) {
	label_id first = labels.first;
	label_id second = labels.second;
	id train = 0;
	id test = size;
	while (train < test) {
		while (sampleLabels[train] == first || sampleLabels[train] == second) {
			train++;
		}
		while (sampleLabels[test] != first && sampleLabels[test] != second) {
			test--;
		}
		if (train < test) {
			this->swapSamples(train++, test--);
		}
	}
}

#endif
