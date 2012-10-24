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

