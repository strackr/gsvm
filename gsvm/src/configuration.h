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

#ifndef CONFIGURATION_H_
#define CONFIGURATION_H_

#include <iostream>

#include <boost/program_options.hpp>

#include "data/solver_factory.h"
#include "svm/kernel.h"
#include "svm/strategy.h"
#include "time/timer.h"
#include "model/selection.h"
#include "logging/log.h"
#include "svm/stop.h"

using namespace std;
using namespace boost::program_options;


#define PR_HELP "help,h"
#define PR_C_LOW "c-low,c"
#define PR_C_HIGH "c-high,C"
#define PR_G_LOW "gamma-low,g"
#define PR_G_HIGH "gamma-high,G"
#define PR_RES "resolution,r"
#define PR_OUTER_FLD "outer-folds,o"
#define PR_INNER_FLD "inner-folds,i"
#define PR_EPSILON "epsilon,e"
#define PR_ETA "eta,E"
#define PR_DRAW_NUM "draw-number,d"
#define PR_INPUT "input,I"
#define PR_SEL_TYPE "model-selection,m"
#define PR_MATRIX_TYPE "matrix-type,t"
#define PR_STOP_CRIT "stop-criterion,p"
#define PR_OPTIMIZATION "optimization,z"
#define PR_ID_RANDOMIZER "randomizer,s"
#define PR_MULTICLASS "multiclass,u"
#define PR_CACHE_SIZE "cache-size,S"
#define PR_WITH_BIAS "with-bias,b"
#define PR_WITHOUT_BIAS "without-bias,B"

#define PR_KEY_HELP "help"
#define PR_KEY_C_LOW "c-low"
#define PR_KEY_C_HIGH "c-high"
#define PR_KEY_G_LOW "gamma-low"
#define PR_KEY_G_HIGH "gamma-high"
#define PR_KEY_RES "resolution"
#define PR_KEY_OUTER_FLD "outer-folds"
#define PR_KEY_INNER_FLD "inner-folds"
#define PR_KEY_EPSILON "epsilon"
#define PR_KEY_ETA "eta"
#define PR_KEY_DRAW_NUM "draw-number"
#define PR_KEY_INPUT "input"
#define PR_KEY_SEL_TYPE "model-selection"
#define PR_KEY_MATRIX_TYPE "matrix-type"
#define PR_KEY_STOP_CRIT "stop-criterion"
#define PR_KEY_OPTIMIZATION "optimization"
#define PR_KEY_ID_RANDOMIZER "randomizer"
#define PR_KEY_MULTICLASS "multiclass"
#define PR_KEY_CACHE_SIZE "cache-size"
#define PR_KEY_WITH_BIAS "with-bias"
#define PR_KEY_WITHOUT_BIAS "without-bias"

#define SEL_TYPE_GRID "grid"
#define SEL_TYPE_PATTERN "pattern"

#define MAT_TYPE_SPARSE "sparse"
#define MAT_TYPE_DENSE "dense"

#define EPSILON_DEFAULT "default"

#define STOP_CRIT_ADJMN "adjmn"
#define STOP_CRIT_MN "mn"
#define STOP_CRIT_MEB "meb"

#define OPTIMIZATION_MDM "mdm"
#define OPTIMIZATION_IMDM "imdm"
#define OPTIMIZATION_GMDM "flex-imdm"

#define ID_RANDOMIZER_PLAIN "simple"
#define ID_RANDOMIZER_FAIR "fair"
#define ID_RANDOMIZER_DETERM "determ"

#define MULTICLASS_ALL_AT_ONCE "allatonce"
#define MULTICLASS_PAIRWISE "pairwise"


/**
 * Exception representing configuration error (like invalid program options).
 */
class invalid_configuration: public exception {

	string message;

public:
	invalid_configuration(string message);
	virtual ~invalid_configuration() throw();

	virtual const char* what() const throw();

};


enum MatrixType {
	SPARSE,
	DENSE
};

enum ModelSelectionType {
	GRID,
	PATTERN
};


struct Configuration {

	string dataFile;

	SearchRange searchRange;
	TrainParams trainingParams;

	struct CrossValidationParams {
		quantity innerFolds;
		quantity outerFolds;
		ModelSelectionType modelSelection;
	} validation;

	MatrixType matrixType;
	ViolationCriterion optimizationProcedure;
	StopCriterion stopCriterion;
	MulticlassApproach multiclass;

	GeneratorType randomization;
};


class ParametersParser {

	variables_map& vars;

public:
	ParametersParser(variables_map& vars);

	Configuration getConfiguration();
};


#endif
