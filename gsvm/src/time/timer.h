#ifndef TIMER_H_
#define TIMER_H_

#include <ctime>

class Timer {

	clock_t total;

	clock_t started;
	clock_t finished;

public:

	Timer();

	void start();
	void stop();

	void reset();
	void restart();

	double getTimeElapsed();

};

#endif
