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
