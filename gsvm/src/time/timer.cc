#include "timer.h"

Timer::Timer() :
		total(0), started(0), finished(0) {
}

void Timer::start() {
	started = clock();
}

void Timer::stop() {
	finished = clock();
	total += finished - started;
	started = finished;
}

void Timer::reset() {
	total = 0;
	started = 0;
	finished = 0;
}

void Timer::restart() {
	reset();
	start();
}

double Timer::getTimeElapsed() {
	return (double) total / CLOCKS_PER_SEC;
}
