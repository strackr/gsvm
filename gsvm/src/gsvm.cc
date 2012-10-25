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

#include "gsvm.h"


template<typename Matrix, typename Strategy>
void performNestedCrossValidation(ifstream& input, TrainParams params,
		SearchRange range, StopCriterion crit, quantity innerFolds, quantity outerFolds, bool useGrid) {
	DefaultSolverBuilder<Matrix, Strategy> reader(input, params, crit, innerFolds, outerFolds, false);

	Timer timer;

	timer.start();
	CrossSolver<GaussKernel, Matrix, Strategy> *solver = (CrossSolver<GaussKernel, Matrix, Strategy>*) reader.getSolver();
	timer.stop();

	logger << format("input reading time: %.2f[s]\n") % timer.getTimeElapsed();

	timer.restart();
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
void performCrossValidation(ifstream& input, TrainParams params,
		SearchRange range, StopCriterion crit, quantity innerFolds) {
	DefaultSolverBuilder<Matrix, Strategy> reader(input, params, crit, innerFolds);

	Timer timer;

	timer.start();
	CrossSolver<GaussKernel, Matrix, Strategy> *solver = (CrossSolver<GaussKernel, Matrix, Strategy>*) reader.getSolver();
	timer.stop();

	logger << format("input reading time: %.2f[s]\n") % timer.getTimeElapsed();

	timer.restart();
	GaussKernel param(range.gammaLow);
	solver->setKernelParams(range.cLow, param);
	TestingResult result = solver->doCrossValidation();
	timer.stop();

	logger << format("final result: time=%.2f[s], accuracy=%.2f[%%]\n")
			% timer.getTimeElapsed() % (100.0 * result.accuracy);

	delete solver;
}

template<typename Matrix, typename Strategy>
void performTraining(ifstream& input, TrainParams params,
		SearchRange range, StopCriterion crit) {
	DefaultSolverBuilder<Matrix, Strategy> reader(input, params, crit);

	Timer timer;

	timer.start();
	CrossSolver<GaussKernel, Matrix, Strategy> *solver = (CrossSolver<GaussKernel, Matrix, Strategy>*) reader.getSolver();
	timer.stop();

	logger << format("input reading time: %.2f[s]\n") % timer.getTimeElapsed();

	timer.restart();
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
void run(variables_map &vars, SearchRange& range, quantity outerFolds, quantity innerFolds,
		TrainParams& params, StopCriterion stop, ifstream& input) {
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
void selectGeneratorTypeAndRun(variables_map &vars, SearchRange& range, quantity outer, quantity inner,
		TrainParams& params, StopCriterion stop, ifstream& input) {
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
void selectViolationCriterionAndRun(variables_map &vars, SearchRange& range, quantity outer, quantity inner,
		TrainParams& params, StopCriterion stop, ifstream& input) {
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

void selectMatrixTypeAndRun(variables_map &vars, SearchRange& range, quantity outer, quantity inner,
		TrainParams& params, StopCriterion stop, ifstream& input) {
	string matrix = vars[PR_KEY_MATRIX_TYPE].as<string>();
	if (MAT_TYPE_SPARSE == matrix) {
		selectViolationCriterionAndRun<sfmatrix>(vars, range, outer, inner, params, stop, input);
	} else if (MAT_TYPE_DENSE == matrix) {
		selectViolationCriterionAndRun<dfmatrix>(vars, range, outer, inner, params, stop, input);
	} else {
		throw InvalidConfigurationException("invalid matrix type: " + matrix);
	}
}


int main(int argc, char *argv[]) {
	options_description desc("Allowed options");
	desc.add_options()
		(PR_HELP, "produce help message")
		(PR_C_LOW, value<double>()->default_value(0.0625), "C value (lower bound)")
		(PR_C_HIGH, value<double>()->default_value(1024), "C value (upper bound)")
		(PR_G_LOW, value<double>()->default_value(0.0009765625), "gamma value (lower bound)")
		(PR_G_HIGH, value<double>()->default_value(16), "gamma value (upper bound)")
		(PR_RES, value<int>()->default_value(8), "resolution (for C and gamma)")
		(PR_OUTER_FLD, value<int>()->default_value(1), "outer folds")
		(PR_INNER_FLD, value<int>()->default_value(10), "inner folds")
		(PR_EPSILON, value<string>()->default_value(EPSILON_DEFAULT), "epsilon value")
		(PR_DRAW_NUM, value<int>()->default_value(600), "draw number")
		(PR_SEL_TYPE, value<string>()->default_value(SEL_TYPE_PATTERN), "model selection type (grid or pattern)")
		(PR_MATRIX_TYPE, value<string>()->default_value(MAT_TYPE_SPARSE), "data representation (sparse or dense)")
		(PR_STOP_CRIT, value<string>()->default_value(STOP_CRIT_DEFAULT), "stopping criterion (meb or default)")
		(PR_OPTIMIZATION, value<string>()->default_value(OPTIMIZATION_MDM), "optimization strategy (mdm or imdm)")
		(PR_ID_RANDOMIZER, value<string>()->default_value(ID_RANDOMIZER_FAIR), "id generator (simple, fair, determ)")
		(PR_INPUT, value<string>(), "input file");

	positional_options_description opt;
	opt.add(PR_KEY_INPUT, -1);

	variables_map vars;
	store(command_line_parser(argc, argv).
	          options(desc).positional(opt).run(), vars);
	notify(vars);

	if (!vars.count(PR_KEY_HELP) && vars.count(PR_KEY_INPUT)) {
		map<string, variable_value>::iterator it;
		for (it = vars.begin(); it != vars.end(); it++) {
			if (typeid(string) == it->second.value().type()) {
				logger << format("%-20s%s\n") % it->first % it->second.as<string>();
			} else if (typeid(double) == it->second.value().type()) {
				logger << format("%-20s%g\n") % it->first % it->second.as<double>();
			} else if (typeid(int) == it->second.value().type()) {
				logger << format("%-20s%d\n") % it->first % it->second.as<int>();
			}
		}
		logger << endl;

		string fileName = vars[PR_KEY_INPUT].as<string>();
		ifstream input(fileName.c_str());

		try {
			SearchRange range;
			range.cResolution = vars[PR_KEY_RES].as<int>();
			range.cLow = vars[PR_KEY_C_LOW].as<double>();
			range.cHigh = vars[PR_KEY_C_HIGH].as<double>();
			range.gammaResolution = vars[PR_KEY_RES].as<int>();
			range.gammaLow = vars[PR_KEY_G_LOW].as<double>();
			range.gammaHigh = vars[PR_KEY_G_HIGH].as<double>();

			fvalue epsilon = 0.0;
			if (vars[PR_KEY_EPSILON].as<string>() != EPSILON_DEFAULT) {
				istringstream epstr(vars[PR_KEY_EPSILON].as<string>());
				epstr >> epsilon;
			}
			quantity drawNumber = vars[PR_KEY_DRAW_NUM].as<int>();

			TrainParams params(epsilon, drawNumber);

			string stopStr = vars[PR_KEY_STOP_CRIT].as<string>();
			StopCriterion stop;
			if (STOP_CRIT_DEFAULT == stopStr) {
				stop = DEFAULT;
			} else if (STOP_CRIT_MEB == stopStr) {
				stop = MEB;
			} else {
				throw InvalidConfigurationException("invalid stopping criterion: " + stopStr);
			}

			quantity innerFolds = vars[PR_KEY_INNER_FLD].as<int>();
			quantity outerFolds = vars[PR_KEY_OUTER_FLD].as<int>();

			selectMatrixTypeAndRun(vars, range, outerFolds, innerFolds, params, stop, input);
		} catch (InvalidConfigurationException &e) {
			cerr << e.getMessage() << endl;
			cerr << desc;
		}

		input.close();
	} else {
		cerr << desc;
	}
	return 0;
}
