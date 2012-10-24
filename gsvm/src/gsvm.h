#ifndef USVM_H_
#define USVM_H_

#include <iostream>

#include <boost/program_options.hpp>

#include "io/data.h"
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
#define PR_DRAW_NUM "draw-number,d"
#define PR_INPUT "input,I"
#define PR_SEL_TYPE "model-selection,m"
#define PR_MATRIX_TYPE "matrix-type,t"
#define PR_STOP_CRIT "stop-criterion,p"
#define PR_OPTIMIZATION "optimization,z"
#define PR_ID_RANDOMIZER "randomizer,s"

#define PR_KEY_HELP "help"
#define PR_KEY_C_LOW "c-low"
#define PR_KEY_C_HIGH "c-high"
#define PR_KEY_G_LOW "gamma-low"
#define PR_KEY_G_HIGH "gamma-high"
#define PR_KEY_RES "resolution"
#define PR_KEY_OUTER_FLD "outer-folds"
#define PR_KEY_INNER_FLD "inner-folds"
#define PR_KEY_EPSILON "epsilon"
#define PR_KEY_DRAW_NUM "draw-number"
#define PR_KEY_INPUT "input"
#define PR_KEY_SEL_TYPE "model-selection"
#define PR_KEY_MATRIX_TYPE "matrix-type"
#define PR_KEY_STOP_CRIT "stop-criterion"
#define PR_KEY_OPTIMIZATION "optimization"
#define PR_KEY_ID_RANDOMIZER "randomizer"

#define SEL_TYPE_GRID "grid"
#define SEL_TYPE_PATTERN "pattern"

#define MAT_TYPE_SPARSE "sparse"
#define MAT_TYPE_DENSE "dense"

#define EPSILON_DEFAULT "default"

#define STOP_CRIT_DEFAULT "default"
#define STOP_CRIT_MEB "meb"

#define OPTIMIZATION_MDM "mdm"
#define OPTIMIZATION_IMDM "imdm"
#define OPTIMIZATION_GMDM "flex-imdm"

#define ID_RANDOMIZER_PLAIN "simple"
#define ID_RANDOMIZER_FAIR "fair"
#define ID_RANDOMIZER_DETERM "determ"


class InvalidConfigurationException {

	string message;

public:
	InvalidConfigurationException(string message) : message(message) {
	}

	string getMessage() {
		return message;
	}

};

#endif
