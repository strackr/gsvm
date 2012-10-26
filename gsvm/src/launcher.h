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

#include "configuration.h"


class ApplicationLauncher {

	Configuration &conf;

	void selectMatrixTypeAndRun();

	template<typename Matrix>
	void selectViolationCriterionAndRun();

	template<typename Matrix, ViolationCriterion Violation>
	void selectGeneratorTypeAndRun();

	template<typename Matrix, ViolationCriterion Violation, GeneratorType Generator>
	void run();

protected:
	template<typename Matrix, typename Strategy>
	CrossSolver<GaussKernel, Matrix, Strategy>* createSolver();

	template<typename Matrix, typename Strategy>
	void performTraining();

	template<typename Matrix, typename Strategy>
	void performCrossValidation();

	template<typename Matrix, typename Strategy>
	void performNestedCrossValidation();

public:
	ApplicationLauncher(Configuration &conf) : conf(conf) {
	}

	void launch();

};

template<typename Matrix, typename Strategy>
CrossSolver<GaussKernel, Matrix, Strategy>* ApplicationLauncher::createSolver() {
	ifstream input(conf.dataFile.c_str());
	DefaultSolverBuilder<Matrix, Strategy> reader(input,
			conf.trainingParams,
			conf.stopCriterion,
			conf.validation.innerFolds,
			conf.validation.outerFolds);

	Timer timer(true);
	CrossSolver<GaussKernel, Matrix, Strategy> *solver
			= (CrossSolver<GaussKernel, Matrix, Strategy>*) reader.getSolver();
	timer.stop();

	logger << format("input reading time: %.2f[s]\n") % timer.getTimeElapsed();
	input.close();
	return solver;
}

template<typename Matrix, typename Strategy>
void ApplicationLauncher::performNestedCrossValidation() {
	CrossSolver<GaussKernel, Matrix, Strategy> *solver = createSolver<Matrix, Strategy>();

	Timer timer(true);
	GridGaussianModelSelector<Matrix, Strategy> *selector;
	if (conf.validation.modelSelection == GRID) {
		selector = new GridGaussianModelSelector<Matrix, Strategy>();
	} else {
		PatternFactory factory;
		selector = new PatternGaussianModelSelector<Matrix, Strategy>(factory.createCross());
	}
	TestingResult res = selector->doNestedCrossValidation(*solver, conf.searchRange);
	timer.stop();

	logger << format("final result: time=%.2f[s], accuracy=%.2f[%%]\n")
			% timer.getTimeElapsed() % (100.0 * res.accuracy);

	delete solver;
	delete selector;
}

template<typename Matrix, typename Strategy>
void ApplicationLauncher::performCrossValidation() {
	CrossSolver<GaussKernel, Matrix, Strategy> *solver = createSolver<Matrix, Strategy>();

	Timer timer(true);
	GaussKernel param(conf.searchRange.gammaLow);
	solver->setKernelParams(conf.searchRange.cLow, param);
	TestingResult result = solver->doCrossValidation();
	timer.stop();

	logger << format("final result: time=%.2f[s], accuracy=%.2f[%%]\n")
			% timer.getTimeElapsed() % (100.0 * result.accuracy);

	delete solver;
}

template<typename Matrix, typename Strategy>
void ApplicationLauncher::performTraining() {
	CrossSolver<GaussKernel, Matrix, Strategy> *solver = createSolver<Matrix, Strategy>();

	Timer timer(true);
	GaussKernel param(conf.searchRange.gammaLow);
	solver->setKernelParams(conf.searchRange.cLow, param);
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
void ApplicationLauncher::run() {
	if (conf.validation.outerFolds > 1) {
		performNestedCrossValidation<Matrix, SolverStrategy<Violation, Generator> >();
	} else {
		if (conf.validation.innerFolds > 1) {
			performCrossValidation<Matrix, SolverStrategy<Violation, Generator> >();
		} else {
			performTraining<Matrix, SolverStrategy<Violation, Generator> >();
		}
	}
}

template<typename Matrix, ViolationCriterion Violation>
void ApplicationLauncher::selectGeneratorTypeAndRun() {
	switch (conf.randomization) {
	case PLAIN:
		run<Matrix, Violation, PLAIN>();
		break;
	case FAIR:
		run<Matrix, Violation, FAIR>();
		break;
	case DETERMINISTIC:
		run<Matrix, Violation, DETERMINISTIC>();
		break;
	default:
		throw InvalidConfigurationException("unknown randomization type");
	}
}

template<typename Matrix>
void ApplicationLauncher::selectViolationCriterionAndRun() {
	switch (conf.optimizationProcedure) {
	case MDM:
		selectGeneratorTypeAndRun<Matrix, MDM>();
		break;
	case IMDM:
		selectGeneratorTypeAndRun<Matrix, IMDM>();
		break;
	case GMDM:
		selectGeneratorTypeAndRun<Matrix, GMDM>();
		break;
	default:
		throw InvalidConfigurationException("unknown optimization criterion");
	}
}

#endif
