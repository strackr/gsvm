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

#include "launcher.h"


InvalidConfigurationException::InvalidConfigurationException(string message) : message(message) {
}

string InvalidConfigurationException::getMessage() {
	return message;
}


void ApplicationLauncher::selectMatrixTypeAndRun(variables_map &vars, SearchRange& range, quantity outer, quantity inner,
		TrainParams& params, StopCriterion& stop, ifstream& input) {
	string matrix = vars[PR_KEY_MATRIX_TYPE].as<string>();
	if (MAT_TYPE_SPARSE == matrix) {
		selectViolationCriterionAndRun<sfmatrix>(vars, range, outer, inner, params, stop, input);
	} else if (MAT_TYPE_DENSE == matrix) {
		selectViolationCriterionAndRun<dfmatrix>(vars, range, outer, inner, params, stop, input);
	} else {
		throw InvalidConfigurationException("invalid matrix type: " + matrix);
	}
}

void ApplicationLauncher::launch() {
	string fileName = vars[PR_KEY_INPUT].as<string>();
	ifstream input(fileName.c_str());

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

	input.close();
}
