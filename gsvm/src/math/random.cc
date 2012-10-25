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

#include "random.h"

rng *Generators::random = NULL;


IdGenerator::IdGenerator(rng *random) : random(random) {
}

IdGenerator::~IdGenerator() {
}

id IdGenerator::nextId(id limit) {
	return rng_next_int(random, limit);
}


void Generators::initializeIfNecessary() {
	if (random == NULL) {
		rng_setup();
		random = rng_alloc(DEFAULT_RNDGEN);
	}
}

IdGenerator Generators::create() {
	initializeIfNecessary();
	return IdGenerator(random);
}
