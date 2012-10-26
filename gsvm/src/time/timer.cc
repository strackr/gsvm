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

#include "timer.h"

Timer::Timer(bool autostart) :
		total(0), started(0), finished(0) {
	if (autostart) {
		start();
	}
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
