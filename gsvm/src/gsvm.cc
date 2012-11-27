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

#include "gsvm.h"

ostream& operator<< (ostream& os, variables_map& vars) {
	map<string, variable_value>::iterator it;
	for (it = vars.begin(); it != vars.end(); it++) {
		if (typeid(string) == it->second.value().type()) {
			os << format("%-20s%s\n") % it->first % it->second.as<string>();
		} else if (typeid(double) == it->second.value().type()) {
			os << format("%-20s%g\n") % it->first % it->second.as<double>();
		} else if (typeid(int) == it->second.value().type()) {
			os << format("%-20s%d\n") % it->first % it->second.as<int>();
		}
	}
	os << endl;
	return os;
}

int main(int argc, char *argv[]) {
	string usage = (format("Usage: %s [OPTION]... [FILE]\n") % PACKAGE).str();
	string descr = "Perform SVM training for the given data set [FILE].\n";
	string options = "Available options";
	options_description desc(usage + descr + options);
	desc.add_options()
		(PR_HELP, "produce help message")
		(PR_C_LOW, value<fvalue>()->default_value(0.0625),
				"C value (lower bound)")
		(PR_C_HIGH, value<fvalue>()->default_value(1024),
				"C value (upper bound)")
		(PR_G_LOW, value<fvalue>()->default_value(0.0009765625),
				"gamma value (lower bound)")
		(PR_G_HIGH, value<fvalue>()->default_value(16),
				"gamma value (upper bound)")
		(PR_RES, value<int>()->default_value(8), "resolution (for C and gamma)")
		(PR_OUTER_FLD, value<int>()->default_value(1), "outer folds")
		(PR_INNER_FLD, value<int>()->default_value(10), "inner folds")
		(PR_EPSILON, value<string>()->default_value(EPSILON_DEFAULT),
				"epsilon value")
		(PR_ETA, value<fvalue>()->default_value(DEFAULT_ETA), "eta value")
		(PR_DRAW_NUM, value<int>()->default_value(600), "draw number")
		(PR_MULTICLASS, value<string>()->default_value(MULTICLASS_ALL_AT_ONCE),
				"multiclass training approach (allatonce or pairwise)")
		(PR_SEL_TYPE, value<string>()->default_value(SEL_TYPE_PATTERN),
				"model selection type (grid or pattern)")
		(PR_MATRIX_TYPE, value<string>()->default_value(MAT_TYPE_SPARSE),
				"data representation (sparse or dense)")
		(PR_STOP_CRIT, value<string>()->default_value(STOP_CRIT_ADJMN),
				"stopping criterion (adjmn, mn or meb)")
		(PR_OPTIMIZATION, value<string>()->default_value(OPTIMIZATION_MDM),
				"optimization strategy (mdm or imdm)")
		(PR_ID_RANDOMIZER, value<string>()->default_value(ID_RANDOMIZER_FAIR),
				"id generator (simple, fair or determ)")
		(PR_CACHE_SIZE, value<int>()->default_value(DEFAULT_CACHE_SIZE),
				"cache size (in MB)")
		(PR_INPUT, value<string>(), "input file");

	positional_options_description opt;
	opt.add(PR_KEY_INPUT, -1);

	try {
		variables_map vars;
		store(command_line_parser(argc, argv).
	          options(desc).positional(opt).run(), vars);
		notify(vars);

		if (!vars.count(PR_KEY_HELP)) {
			ParametersParser parser(vars);
			Configuration conf = parser.getConfiguration();

			logger << vars;

			ApplicationLauncher launcher(conf);
			launcher.launch();
		} else {
			cerr << desc;
		}
	} catch (exception& e) {
		cerr << e.what() << "\n" << endl;
		cerr << desc;
	}
	return 0;
}
