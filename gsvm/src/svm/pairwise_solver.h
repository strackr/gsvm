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

#ifndef SOLVER_PAIRWISE_H_
#define SOLVER_PAIRWISE_H_

#include "solver.h"


struct PairwiseTrainingResult {

	pair<label_id, label_id> trainingLabels;
	vector<fvalue> yalphas;
	fvalue bias;
	vector<sample_id> samples;
	quantity size;

	PairwiseTrainingResult(pair<label_id, label_id>& trainingLabels, quantity size) :
			trainingLabels(trainingLabels),
			yalphas(vector<fvalue>(size, 0.0)),
			bias(0),
			samples(vector<sample_id>(size, INVALID_SAMPLE_ID)),
			size(0) {
	}

	void clear() {
		yalphas.clear();
		bias = 0;
		samples.clear();
	}

};


struct PairwiseTrainingState {

	vector<PairwiseTrainingResult> models;
	quantity svNumber;
	quantity labelNumber;

};


/**
 * Pairwise classifier perform classification based on SVM models created by
 * pairwise solver.
 */
template<typename Kernel, typename Matrix>
class PairwiseClassifier: public Classifier<Kernel, Matrix> {

	RbfKernelEvaluator<Kernel, Matrix>* evaluator;
	PairwiseTrainingState* state;
	fvector *buffer;

	vector<quantity> votes;
	vector<fvalue> evidence;

protected:
	fvalue getDecisionForModel(sample_id sample,
			PairwiseTrainingResult* model, fvector* buffer);
	fvalue convertDecisionToEvidence(fvalue decision);

public:
	PairwiseClassifier(RbfKernelEvaluator<Kernel, Matrix> *evaluator,
			PairwiseTrainingState* state, fvector *buffer);
	virtual ~PairwiseClassifier();

	virtual label_id classify(sample_id sample);
	virtual quantity getSvNumber();

};

template<typename Kernel, typename Matrix>
PairwiseClassifier<Kernel, Matrix>::PairwiseClassifier(
		RbfKernelEvaluator<Kernel, Matrix> *evaluator,
		PairwiseTrainingState* state, fvector *buffer) :
		evaluator(evaluator),
		state(state),
		buffer(buffer),
		votes(state->labelNumber),
		evidence(state->labelNumber){
}

template<typename Kernel, typename Matrix>
PairwiseClassifier<Kernel, Matrix>::~PairwiseClassifier() {
}

template<typename Kernel, typename Matrix>
label_id PairwiseClassifier<Kernel, Matrix>::classify(sample_id sample) {
	fill(votes.begin(), votes.end(), 0);
	fill(evidence.begin(), evidence.end(), 0.0);

	evaluator->evalInnerKernel(sample, 0, state->svNumber, buffer);

	vector<PairwiseTrainingResult>::iterator it;
	for (it = state->models.begin(); it != state->models.end(); it++) {
		PairwiseTrainingResult* result = it.base();
		fvalue dec = getDecisionForModel(sample, result, buffer);
		label_id label = dec > 0
				? result->trainingLabels.first
				: result->trainingLabels.second;
		votes[label]++;
		fvalue evidValue = convertDecisionToEvidence(dec);
		evidence[result->trainingLabels.first] += evidValue;
		evidence[result->trainingLabels.second] += evidValue;
	}

	label_id maxLabelId = 0;
	quantity maxVotes = 0;
	quantity maxEvidence = 0.0;
	for (label_id i = 0; i < state->labelNumber; i++) {
		if (votes[i] > maxVotes
			|| (votes[i] == maxVotes && evidence[i] > maxEvidence)) {
			maxLabelId = i;
			maxVotes = votes[i];
			maxEvidence = evidence[i];
		}
	}

	return maxLabelId;
}

template<typename Kernel, typename Matrix>
fvalue PairwiseClassifier<Kernel, Matrix>::getDecisionForModel(sample_id sample,
		PairwiseTrainingResult* model, fvector* buffer) {
	fvalue dec = model->bias;
	label_id positiveLabel = model->trainingLabels.first;
	fvalue* kernels = buffer->data;
	for (sample_id i = 0; i < model->size; i++) {
		dec += model->yalphas[i] * kernels[model->samples[i]];
	}
	return dec;
}

template<typename Kernel, typename Matrix>
inline fvalue PairwiseClassifier<Kernel, Matrix>::convertDecisionToEvidence(
		fvalue decision) {
	return decision;
}

template<typename Kernel, typename Matrix>
quantity PairwiseClassifier<Kernel, Matrix>::getSvNumber() {
	return state->svNumber;
}


/**
 * Pairwise solver performs SVM training by generating SVM state for all
 * two-element combinations of the class trainingLabels.
 */
template<typename Kernel, typename Matrix, typename Strategy>
class PairwiseSolver: public AbstractSolver<Kernel, Matrix, Strategy> {

	template<typename K, typename V>
	struct PairValueComparator {

		bool operator()(pair<K, V> pair1, pair<K, V> pair2) {
			return pair1.second > pair2.second;
		}

	};

	PairwiseTrainingState state;

	quantity reorderSamples(label_id *labels, quantity size,
			pair<label_id, label_id>& labelPair);

protected:
	CachedKernelEvaluator<Kernel, Matrix, Strategy>* buildCache(
			fvalue c, Kernel &gparams);

public:
	PairwiseSolver(map<label_id, string> labelNames, Matrix *samples,
			label_id *labels, TrainParams &params,
			StopCriterionStrategy *stopStrategy);
	virtual ~PairwiseSolver();

	void train();
	Classifier<Kernel, Matrix>* getClassifier();

	quantity getSvNumber();

};


template<typename Kernel, typename Matrix, typename Strategy>
PairwiseSolver<Kernel, Matrix, Strategy>::PairwiseSolver(
		map<label_id, string> labelNames, Matrix *samples,
		label_id *labels, TrainParams &params,
		StopCriterionStrategy *stopStrategy) :
		AbstractSolver<Kernel, Matrix, Strategy>(labelNames,
				samples, labels, params, stopStrategy),
		state(PairwiseTrainingState()) {
	label_id maxLabel = labelNames.size();
	vector<quantity> classSizes(maxLabel, 0);
	for (sample_id sample = 0; sample < this->size; sample++) {
		classSizes[this->labels[sample]]++;
	}

	// sort
	vector<pair<label_id, quantity> > sizes(maxLabel);
	for (label_id label = 0; label < maxLabel; label++) {
		sizes[label] = pair<label_id, quantity>(label, classSizes[label]);
	}
	sort(sizes.begin(), sizes.end(), PairValueComparator<label_id, quantity>());

	vector<pair<label_id, quantity> >::iterator it1;
	for (it1 = sizes.begin(); it1 < sizes.end(); it1++) {
		vector<pair<label_id, quantity> >::iterator it2;
		for (it2 = it1 + 1; it2 < sizes.end(); it2++) {
			pair<label_id, label_id> labels(it1->first, it2->first);
			quantity size = it1->second + it2->second;
			state.models.push_back(PairwiseTrainingResult(labels, size));
		}
	}
	state.labelNumber = maxLabel;
}

template<typename Kernel, typename Matrix, typename Strategy>
PairwiseSolver<Kernel, Matrix, Strategy>::~PairwiseSolver() {
}

template<typename Kernel, typename Matrix, typename Strategy>
Classifier<Kernel, Matrix>* PairwiseSolver<Kernel, Matrix, Strategy>::getClassifier() {
	return new PairwiseClassifier<Kernel, Matrix>(this->cache->getEvaluator(),
			&state, this->cache->getBuffer());
}

template<typename Kernel, typename Matrix, typename Strategy>
void PairwiseSolver<Kernel, Matrix, Strategy>::train() {
	quantity totalSize = this->currentSize;
	vector<PairwiseTrainingResult>::iterator it;
	for (it = state.models.begin(); it != state.models.end(); it++) {
		pair<label_id, label_id> trainPair = it->trainingLabels;
		quantity size = reorderSamples(this->labels, totalSize, trainPair);
		this->setCurrentSize(size);
		this->reset();
		this->trainForCache(this->cache);

		fvalue bias = 0;
		fvalue* cacheAlphas = this->cache->getAlphas()->data;
		sample_id* cacheSamples = this->cache->getBackwardOrder();
		quantity svNumber  = this->cache->getSVNumber();
		for (quantity i = 0; i < svNumber; i++) {
			fvalue yy = this->labels[i] == trainPair.first ? 1.0 : -1.0;
			fvalue yalpha = yy * cacheAlphas[i];
			it->yalphas[i] = yalpha;
			it->samples[i] = cacheSamples[i];
			bias += yalpha;
		}
		it->bias = bias;
		it->size = svNumber;
	}

	id freeOffset = state.models.back().size;
	sample_id* mapping = this->cache->getForwardOrder();
	for (it = state.models.begin(); it != state.models.end(); it++) {
		for (id i = 0; i < it->size; i++) {
			id realOffset = mapping[it->samples[i]];
			if (realOffset >= freeOffset) {
				this->swapSamples(realOffset, freeOffset);
				realOffset = freeOffset++;
			}
			it->samples[i] = realOffset;
		}
	}
	state.svNumber = freeOffset;

	this->setCurrentSize(totalSize);
}

template<typename Kernel, typename Matrix, typename Strategy>
quantity PairwiseSolver<Kernel, Matrix, Strategy>::reorderSamples(
		label_id *labels, quantity size, pair<label_id, label_id>& labelPair) {
	label_id first = labelPair.first;
	label_id second = labelPair.second;
	id train = 0;
	id test = size - 1;
	while (train <= test) {
		while (train < size && (labels[train] == first || labels[train] == second)) {
			train++;
		}
		while (test >= 0 && (labels[test] != first && labels[test] != second)) {
			test--;
		}
		if (train < test) {
			this->swapSamples(train++, test--);
		}
	}
	return train;
}

template<typename Kernel, typename Matrix, typename Strategy>
CachedKernelEvaluator<Kernel, Matrix, Strategy>* PairwiseSolver<Kernel, Matrix, Strategy>::buildCache(
		fvalue c, Kernel &gparams) {
	RbfKernelEvaluator<GaussKernel, Matrix> *rbf = new RbfKernelEvaluator<GaussKernel, Matrix>(
			this->samples, this->labels, 2, c, gparams);
	return new CachedKernelEvaluator<GaussKernel, Matrix, Strategy>(
			rbf, &this->strategy, this->size, this->params.cache.size, this->params.eta, NULL);
}

template<typename Kernel, typename Matrix, typename Strategy>
quantity PairwiseSolver<Kernel, Matrix, Strategy>::getSvNumber() {
	return state.svNumber;
}

#endif
