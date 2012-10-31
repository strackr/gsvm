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

#ifndef SELECTION_H_
#define SELECTION_H_

#include "../time/timer.h"
#include "../logging/log.h"
#include "../svm/validation.h"
#include "../svm/kernel.h"

#define LOG_STEP(f, t, s) ((t > f && s > 1) ? (exp(log((t) / (f)) / ((s) - 1))) : 1.0)

struct SearchRange {

	fvalue cLow;
	fvalue cHigh;
	quantity cResolution;

	fvalue gammaLow;
	fvalue gammaHigh;
	quantity gammaResolution;

};


struct ModelSelectionResults {

	fvalue c;
	fvalue gamma;

	TestingResult bestResult;

};


template<typename Matrix, typename Strategy>
class GridGaussianModelSelector {

protected:
	TestingResult validate(CrossValidationSolver<GaussKernel, Matrix, Strategy>& solver,
			fvalue c, fvalue gamma);

public:
	virtual ~GridGaussianModelSelector();

	virtual ModelSelectionResults selectParameters(
			CrossValidationSolver<GaussKernel, Matrix, Strategy> &solver, SearchRange &range);
	TestingResult doNestedCrossValidation(CrossValidationSolver<GaussKernel, Matrix, Strategy> &solver,
			SearchRange &range);

};

template<typename Matrix, typename Strategy>
GridGaussianModelSelector<Matrix, Strategy>::~GridGaussianModelSelector() {
}

template<typename Matrix, typename Strategy>
TestingResult GridGaussianModelSelector<Matrix, Strategy>::validate(
		CrossValidationSolver<GaussKernel, Matrix, Strategy>& solver, fvalue c, fvalue gamma) {
	Timer timer;
	GaussKernel param(gamma);
	solver.setKernelParams(c, param);

	timer.start();
	TestingResult result = solver.doCrossValidation();
	timer.stop();

	logger << format("outer fold %d CV: time=%.2f[s], accuracy=%.2f[%%], C=%.4g, G=%.4g\n")
			% solver.getOuterFold() % timer.getTimeElapsed() % (100.0 * result.accuracy) % c % gamma;

	return result;
}

template<typename Matrix, typename Strategy>
ModelSelectionResults GridGaussianModelSelector<Matrix, Strategy>::selectParameters(
		CrossValidationSolver<GaussKernel, Matrix, Strategy> &solver, SearchRange &range) {
	fvalue cRatio = LOG_STEP(range.cLow, range.cHigh, range.cResolution);
	fvalue gammaRatio = LOG_STEP(range.gammaLow, range.gammaHigh, range.gammaResolution);

	ModelSelectionResults results;
	results.bestResult.accuracy = 0.0;

	for (quantity cIter = 0; cIter < range.cResolution; cIter++) {
		fvalue c = range.cLow * pow(cRatio, cIter);
		for (quantity gammaIter = 0; gammaIter < range.gammaResolution; gammaIter++) {
			fvalue gamma = range.gammaLow * pow(gammaRatio, gammaIter);

			TestingResult result = validate(solver, c, gamma);

			if (result.accuracy > results.bestResult.accuracy) {
				results.bestResult = result;
				results.c = c;
				results.gamma = gamma;
			}
		}
	}

	return results;
}

template<typename Matrix, typename Strategy>
TestingResult GridGaussianModelSelector<Matrix, Strategy>::doNestedCrossValidation(
		CrossValidationSolver<GaussKernel, Matrix, Strategy> &solver, SearchRange &range) {
	Timer timer;

	TestingResult result;
	GaussKernel initial(range.gammaLow);
	solver.setKernelParams(range.cLow, initial);
	for (fold_id fold = 0; fold < solver.getOuterFoldsNumber(); fold++) {
		solver.resetOuterFold(fold);

		timer.restart();
		ModelSelectionResults params = selectParameters(solver, range);
		timer.stop();

		logger << format("outer fold %d model selection: time=%.2f[s], accuracy=%.2f[%%], C=%.4g, G=%.4g\n")
				% fold % timer.getTimeElapsed() % (100.0 * params.bestResult.accuracy) % params.c % params.gamma;

		GaussKernel kernel(params.gamma);
		solver.setKernelParams(params.c, kernel);

		timer.restart();
		solver.trainOuter();
		timer.stop();

		StateHolder<Matrix> &stateHolder = solver.getStateHolder();
		logger << format("outer fold %d final training: time=%.2f[s], sv=%d/%d\n")
				% fold % timer.getTimeElapsed() % stateHolder.getSvNumber() % solver.getOuterProblemSize();

		timer.restart();
		TestingResult current = solver.testOuter();
		timer.stop();

		logger << format("outer fold %d final testing: time=%.2f[s], accuracy=%.2f[%%]\n")
				% fold % timer.getTimeElapsed() % (100.0 * current.accuracy);

		result.accuracy += current.accuracy / solver.getOuterFoldsNumber();
	}
	return result;
}


typedef unsigned int offset;

#define INVALID_OFFSET ((offset) -1)

struct TrainingCoord {

	offset c;
	offset gamma;

	TrainingCoord(offset c = 0, offset gamma = 0) :
			c(c), gamma(gamma) {
	}

	bool operator<(const TrainingCoord &crd) const {
		return c > crd.c || (c == crd.c && gamma > crd.gamma);
	}

};


struct Pattern {

	TrainingCoord *coords;

	quantity size;
	quantity spread;

	~Pattern();

};


class PatternFactory {

public:
	Pattern* createCross();

};


template<typename Matrix, typename Strategy>
class PatternGaussianModelSelector: public GridGaussianModelSelector<Matrix, Strategy> {

	Pattern *pattern;

	map<TrainingCoord, TestingResult> results;

protected:
	void registerResult(TestingResult result, offset c, offset gamma);

	TrainingCoord findStartingPoint(SearchRange &range);
	quantity evaluateDistance(offset c, offset gamma, SearchRange &range);

	void printTrainingMatrix(SearchRange &range);

public:
	PatternGaussianModelSelector(Pattern *pattern);
	virtual ~PatternGaussianModelSelector();

	virtual ModelSelectionResults selectParameters(
			CrossValidationSolver<GaussKernel, Matrix, Strategy> &solver,
			SearchRange &range);

};

template<typename Matrix, typename Strategy>
PatternGaussianModelSelector<Matrix, Strategy>::PatternGaussianModelSelector(Pattern *pattern) :
		pattern(pattern) {
}

template<typename Matrix, typename Strategy>
PatternGaussianModelSelector<Matrix, Strategy>::~PatternGaussianModelSelector() {
	delete pattern;
}

template<typename Matrix, typename Strategy>
void PatternGaussianModelSelector<Matrix, Strategy>::registerResult(TestingResult result, offset c, offset gamma) {
	results[TrainingCoord(c, gamma)] = result;
}

template<typename Matrix, typename Strategy>
quantity PatternGaussianModelSelector<Matrix, Strategy>::evaluateDistance(offset c, offset gamma, SearchRange &range) {
//	quantity dist = min(min(c, gamma), min(range.cResolution - c, range.gammaResolution - gamma) - 1);
	quantity dist = range.cResolution + range.gammaResolution;
	map<TrainingCoord, TestingResult>::iterator it;
	for (it = results.begin(); it != results.end(); it++) {
		TrainingCoord coord = it->first;
		quantity cdiff = (c > coord.c) ? (c - coord.c) : (coord.c - c);
		quantity sdiff = (gamma > coord.gamma) ? (gamma - coord.gamma) : (coord.gamma - gamma);
		dist = min(dist, cdiff + sdiff);
	}
	return dist;
}

template<typename Matrix, typename Strategy>
TrainingCoord PatternGaussianModelSelector<Matrix, Strategy>::findStartingPoint(SearchRange &range) {
	TrainingCoord startingPoint(INVALID_OFFSET, INVALID_OFFSET);

	offset cCenterOffset = (range.cResolution - 1) / 2;
	offset gammaCenterOffset = (range.gammaResolution - 1) / 2;

	quantity maxDist = 0;

	if (!results.empty()) {
		quantity rangeSpread = min(range.cResolution, range.gammaResolution);
		quantity scale = max(exp2(floor(log2((rangeSpread - 1) / pattern->spread))), 1.0);
		quantity minDist = ceil(sqrt(rangeSpread) / 2);
		do {
			for (offset c = cCenterOffset % scale; c < range.cResolution; c += scale) {
				for (offset s = gammaCenterOffset % scale; s < range.gammaResolution; s += scale) {
					quantity dist = evaluateDistance(c, s, range);

					if (dist > maxDist) {
						maxDist = dist;
						if (dist >= minDist) {
							startingPoint.c = c;
							startingPoint.gamma = s;
						}
					}
				}
			}
			scale /= 2;
		} while (scale > minDist);
	} else {
		startingPoint.c = cCenterOffset;
		startingPoint.gamma = gammaCenterOffset;
	}
	return startingPoint;
}

template<typename Matrix, typename Strategy>
ModelSelectionResults PatternGaussianModelSelector<Matrix, Strategy>::selectParameters(
		CrossValidationSolver<GaussKernel, Matrix, Strategy> &solver, SearchRange &range) {
	results.clear();

	ModelSelectionResults globalRes;
	globalRes.bestResult.accuracy = 0.0;

	TrainingCoord trnCoords = findStartingPoint(range);
	while (trnCoords.c != INVALID_OFFSET && trnCoords.gamma != INVALID_OFFSET) {
		quantity rangeSpread = min(range.cResolution, range.gammaResolution);
		quantity scale = max(exp2(floor(log2((rangeSpread - 1) / pattern->spread))), 1.0);
		offset cOffset = trnCoords.c;
		offset gammaOffset = trnCoords.gamma;

		fvalue cRatio = LOG_STEP(range.cLow, range.cHigh, range.cResolution);
		fvalue gammaRatio = LOG_STEP(range.gammaLow, range.gammaHigh, range.gammaResolution);

		while (scale > 0) {
			TrainingCoord bestPosition;
			fvalue bestAccuracy = -MAX_FVALUE;

			for (quantity i = 0; i < pattern->size; i++) {
				TrainingCoord shift = pattern->coords[i];
				TrainingCoord current(cOffset + scale * shift.c, gammaOffset + scale * shift.gamma);

				if (current.c >= 0 && current.c < range.cResolution
						&& current.gamma >= 0 && current.gamma < range.gammaResolution) {
					TestingResult result;

					if (results.find(current) == results.end()) {
						fvalue c = range.cLow * pow(cRatio, current.c);
						fvalue gamma = range.gammaLow * pow(gammaRatio, current.gamma);

						result = this->validate(solver, c, gamma);
						registerResult(result, current.c, current.gamma);
					} else {
						result = results[current];
					}

					if (result.accuracy > bestAccuracy) {
						bestPosition = current;
						bestAccuracy = result.accuracy;
					}
				}
			}

			if (bestAccuracy > globalRes.bestResult.accuracy) {
				globalRes.bestResult.accuracy = bestAccuracy;
				globalRes.c = range.cLow * pow(cRatio, bestPosition.c);
				globalRes.gamma = range.gammaLow * pow(gammaRatio, bestPosition.gamma);
			}

			if (cOffset == bestPosition.c && gammaOffset == bestPosition.gamma) {
				scale /= 2;
			} else {
				cOffset = bestPosition.c;
				gammaOffset = bestPosition.gamma;
			}
		}
		trnCoords = findStartingPoint(range);
	}
//	printTrainingMatrix(range);
	return globalRes;
}

template<typename Matrix, typename Strategy>
void PatternGaussianModelSelector<Matrix, Strategy>::printTrainingMatrix(
		SearchRange &range) {
	cout << "training matrix size: " << results.size() << endl;
	map<TrainingCoord, TestingResult>::iterator it;
	int table[range.cResolution][range.gammaResolution];
	for (quantity i = 0; i < range.cResolution; i++) {
		for (quantity j = 0; j < range.gammaResolution; j++) {
			table[i][j] = 0;
		}
	}
	for (it = results.begin(); it != results.end(); it++) {
		TrainingCoord coord = it->first;
		table[coord.c][coord.gamma] = 1;
	}
	for (quantity i = 0; i < range.cResolution; i++) {
		for (quantity j = 0; j < range.gammaResolution; j++) {
			cout << (table[i][j] ? 'x' : '.');
		}
		cout << endl;
	}
}

#endif
