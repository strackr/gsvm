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

#ifndef RANDOM_H_
#define RANDOM_H_

#include "../math/numeric.h"

//#define DEFAULT_RNDGEN gsl_rng_ranlux389
//#define DEFAULT_RNDGEN gsl_rng_taus
//#define DEFAULT_RNDGEN gsl_rng_ran3
#define DEFAULT_RNDGEN gsl_rng_default



class IdGenerator {

protected:
	rng *random;

public:
	IdGenerator(rng *random);
	~IdGenerator();

	id nextId(id limit);

};


class Generators {

	static rng *random;

	static void initializeIfNecessary();

public:

	static IdGenerator create();

};


#endif

