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

#ifndef LAUCHER_H_
#define LAUCHER_H_

#include "gsvm.h"


class InvalidConfigurationException {

	string message;

public:
	InvalidConfigurationException(string message);

	string getMessage();

};


class ApplicationLauncher {

	variables_map &vars;

protected:
	void selectMatrixTypeAndRun(variables_map &vars, SearchRange& range, quantity outer, quantity inner,
			TrainParams& params, StopCriterion& stop, ifstream& input);

	template<typename Matrix>
	void selectViolationCriterionAndRun(variables_map &vars, SearchRange& range, quantity outer, quantity inner,
			TrainParams& params, StopCriterion& stop, ifstream& input);
	template<typename Matrix, ViolationCriterion Violation>
	void selectGeneratorTypeAndRun(variables_map &vars, SearchRange& range, quantity outer, quantity inner,
			TrainParams& params, StopCriterion& stop, ifstream& input);
	template<typename Matrix, ViolationCriterion Violation, GeneratorType Generator>
	void run(variables_map &vars, SearchRange& range, quantity outerFolds, quantity innerFolds,
			TrainParams& params, StopCriterion& stop, ifstream& input);

	template<typename Matrix, typename Strategy>
	CrossSolver<GaussKernel, Matrix, Strategy>* createSolver(ifstream& input,
			TrainParams &params, StopCriterion &crit, quantity innerFolds, quantity outerFolds);

	template<typename Matrix, typename Strategy>
	void performTraining(ifstream& input, TrainParams& params,
			SearchRange& range, StopCriterion& crit);
	template<typename Matrix, typename Strategy>
	void performCrossValidation(ifstream& input, TrainParams &params,
			SearchRange& range, StopCriterion& crit, quantity innerFolds);
	template<typename Matrix, typename Strategy>
	void performNestedCrossValidation(ifstream& input, TrainParams& params,
			SearchRange& range, StopCriterion& crit, quantity innerFolds, quantity outerFolds, bool useGrid);

public:
	ApplicationLauncher(variables_map &vars) : vars(vars) {
	}

	void launch();

};

template<typename Matrix, typename Strategy>
CrossSolver<GaussKernel, Matrix, Strategy>* ApplicationLauncher::createSolver(ifstream& input,
		TrainParams &params, StopCriterion &crit, quantity innerFolds, quantity outerFolds) {
	DefaultSolverBuilder<Matrix, Strategy> reader(input, params, crit, innerFolds, outerFolds, false);

	Timer timer;

	timer.start();
	CrossSolver<GaussKernel, Matrix, Strategy> *solver = (CrossSolver<GaussKernel, Matrix, Strategy>*) reader.getSolver();
	timer.stop();

	logger << format("input reading time: %.2f[s]\n") % timer.getTimeElapsed();

	return solver;
}

template<typename Matrix, typename Strategy>
void ApplicationLauncher::performNestedCrossValidation(ifstream& input, TrainParams& params,
		SearchRange& range, StopCriterion& crit, quantity innerFolds, quantity outerFolds, bool useGrid) {
	CrossSolver<GaussKernel, Matrix, Strategy> *solver = createSolver<Matrix, Strategy>(
			input, params, crit, innerFolds, outerFolds);

	Timer timer;
	timer.start();
	GridGaussianModelSelector<Matrix, Strategy> *selector;
	if (useGrid) {
		selector = new GridGaussianModelSelector<Matrix, Strategy>();
	} else {
		PatternFactory factory;
		selector = new PatternGaussianModelSelector<Matrix, Strategy>(factory.createCross());
	}
	TestingResult res = selector->doNestedCrossValidation(*solver, range);
	timer.stop();

	logger << format("final result: time=%.2f[s], accuracy=%.2f[%%]\n")
			% timer.getTimeElapsed() % (100.0 * res.accuracy);

	delete solver;
	delete selector;
}

template<typename Matrix, typename Strategy>
void ApplicationLauncher::performCrossValidation(ifstream& input, TrainParams &params,
		SearchRange& range, StopCriterion& crit, quantity innerFolds) {
	CrossSolver<GaussKernel, Matrix, Strategy> *solver = createSolver<Matrix, Strategy>(
				input, params, crit, innerFolds, 1);

	Timer timer;
	timer.start();
	GaussKernel param(range.gammaLow);
	solver->setKernelParams(range.cLow, param);
	TestingResult result = solver->doCrossValidation();
	timer.stop();

	logger << format("final result: time=%.2f[s], accuracy=%.2f[%%]\n")
			% timer.getTimeElapsed() % (100.0 * result.accuracy);

	delete solver;
}

template<typename Matrix, typename Strategy>
void ApplicationLauncher::performTraining(ifstream& input, TrainParams& params,
		SearchRange& range, StopCriterion& crit) {
	CrossSolver<GaussKernel, Matrix, Strategy> *solver = createSolver<Matrix, Strategy>(
				input, params, crit, 1, 1);

	Timer timer;
	timer.start();
	GaussKernel param(range.gammaLow);
	solver->setKernelParams(range.cLow, param);
	CrossClassifier<GaussKernel, Matrix, Strategy>* classifier = solver->getClassifier();
	timer.stop();

	Matrix *samples = solver->getSamples();
	label_id *labels = solver->getLabels();
	quantity correct = 0;
	quantity total = samples->height;
	for (sample_id v = 0; v < total; v++) {
		label_id lbl = classifier->classify(v);
		if (lbl == labels[v]) {
			correct++;
		}
	}

	logger << format("final result: time=%.2f[s], accuracy=%.2f[%%], sv=%d\n")
			% timer.getTimeElapsed() % (100.0 * correct / total) % classifier->getSvNumber();

	delete solver;
	delete classifier;
}

template<typename Matrix, ViolationCriterion Violation, GeneratorType Generator>
void ApplicationLauncher::run(variables_map &vars, SearchRange& range, quantity outerFolds, quantity innerFolds,
		TrainParams& params, StopCriterion& stop, ifstream& input) {
	if (outerFolds > 1) {
		bool useGrid = SEL_TYPE_GRID == vars[PR_KEY_SEL_TYPE].as<string>();
		performNestedCrossValidation<Matrix, SolverStrategy<Violation, Generator> >(
				input, params, range, stop, innerFolds, outerFolds, useGrid);
	} else {
		if (innerFolds > 1) {
			performCrossValidation<Matrix, SolverStrategy<Violation, Generator> >(
					input, params, range, stop, innerFolds);
		} else {
			performTraining<Matrix, SolverStrategy<Violation, Generator> >(
					input, params, range, stop);
		}
	}
}

template<typename Matrix, ViolationCriterion Violation>
void ApplicationLauncher::selectGeneratorTypeAndRun(variables_map &vars, SearchRange& range, quantity outer, quantity inner,
		TrainParams& params, StopCriterion& stop, ifstream& input) {
	string random = vars[PR_KEY_ID_RANDOMIZER].as<string>();
	if (ID_RANDOMIZER_PLAIN == random) {
		run<Matrix, Violation, PLAIN>(vars, range, outer, inner, params, stop, input);
	} else if (ID_RANDOMIZER_FAIR == random) {
		run<Matrix, Violation, FAIR>(vars, range, outer, inner, params, stop, input);
	} else if (ID_RANDOMIZER_DETERM == random) {
		run<Matrix, Violation, DETERMINISTIC>(vars, range, outer, inner, params, stop, input);
	} else {
		throw InvalidConfigurationException("invalid id generator type: " + random);
	}
}

template<typename Matrix>
void ApplicationLauncher::selectViolationCriterionAndRun(variables_map &vars, SearchRange& range, quantity outer, quantity inner,
		TrainParams& params, StopCriterion& stop, ifstream& input) {
	string optimization = vars[PR_KEY_OPTIMIZATION].as<string>();
	if (OPTIMIZATION_MDM == optimization) {
		selectGeneratorTypeAndRun<sfmatrix, MDM>(vars, range, outer, inner, params, stop, input);
	} else if (OPTIMIZATION_IMDM == optimization) {
		selectGeneratorTypeAndRun<sfmatrix, IMDM>(vars, range, outer, inner, params, stop, input);
	} else if (OPTIMIZATION_GMDM == optimization) {
		selectGeneratorTypeAndRun<sfmatrix, GMDM>(vars, range, outer, inner, params, stop, input);
	} else {
		throw InvalidConfigurationException("invalid optimization criterion: " + optimization);
	}
}

#endif
