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

#include "selection.h"

Pattern::~Pattern() {
	delete [] coords;
}

Pattern* PatternFactory::createCross() {
	Pattern *cross = new Pattern();
	cross->spread = 2;
	cross->size = 5;
	TrainingCoord *coords = new TrainingCoord[5];
	coords[0].c = 0;
	coords[0].gamma = 0;
	coords[1].c = -1;
	coords[1].gamma = 0;
	coords[2].c = 0;
	coords[2].gamma = -1;
	coords[3].c = 1;
	coords[3].gamma = 0;
	coords[4].c = 0;
	coords[4].gamma = 1;
	cross->coords = coords;
	return cross;
}
