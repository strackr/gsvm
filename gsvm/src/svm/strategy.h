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

#ifndef UPDATE_H_
#define UPDATE_H_

#include "violation.h"
#include "generator.h"
#include "params.h"

template<ViolationCriterion Violation, GeneratorType Generator>
class SolverStrategy {

	ViolationEstimator<Violation> estimator;
	CandidateIdGenerator<Generator> generator;

public:
	SolverStrategy(TrainParams &params, quantity labelNumber,
			label_id *labels, quantity sampleNumber);

	fvalue evaluateGoodness(fvalue au, fvalue av, fvalue uv,
			fvalue uw, fvalue vw, fvalue tau);

	id generateNextId();
	void markGeneratedIdAsCorrect();
	void markGeneratedIdAsFailed();
	void resetGenerator(label_id *labels, id maxId);
	void notifyExchange(id u, id v);

};

template<ViolationCriterion Violation, GeneratorType Generator>
inline SolverStrategy<Violation, Generator>::SolverStrategy(TrainParams& params,
		quantity labelNumber, label_id *labels, quantity sampleNumber) :
		estimator(params),
		generator(params, labelNumber, labels, sampleNumber) {
}

template<ViolationCriterion Violation, GeneratorType Generator>
inline fvalue SolverStrategy<Violation, Generator>::evaluateGoodness(fvalue au, fvalue av,
		fvalue uv, fvalue uw, fvalue vw, fvalue tau) {
	return estimator.evaluateGoodness(au, av, uv, uw, vw, tau);
}

template<ViolationCriterion Violation, GeneratorType Generator>
inline id SolverStrategy<Violation, Generator>::generateNextId() {
	return generator.nextId();
}

template<ViolationCriterion Violation, GeneratorType Generator>
inline void SolverStrategy<Violation, Generator>::markGeneratedIdAsCorrect() {
	generator.markCorrect();
}

template<ViolationCriterion Violation, GeneratorType Generator>
inline void SolverStrategy<Violation, Generator>::markGeneratedIdAsFailed() {
	generator.markFailed();
}

template<ViolationCriterion Violation, GeneratorType Generator>
inline void SolverStrategy<Violation, Generator>::resetGenerator(label_id *labels, id maxId) {
	generator.reset(labels, maxId);
}

template<ViolationCriterion Violation, GeneratorType Generator>
inline void SolverStrategy<Violation, Generator>::notifyExchange(id u, id v) {
	generator.exchange(u, v);
}

#endif
