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
