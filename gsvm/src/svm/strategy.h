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
