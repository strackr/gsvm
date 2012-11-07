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

#include "generator.h"

ClassDistribution::ClassDistribution(quantity labelNum, label_id *smplMemb, quantity smplNum) :
		labelNumber(labelNum) {
	bufferSizes = new quantity[labelNum];
	for (quantity i = 0; i < labelNum; i++) {
		bufferSizes[i] = 0;
	}
	buffers = new id*[labelNum];
	bufferHolder = new id[smplNum];
	offsets = new id[smplNum];
	for (quantity i = 0; i < smplNum; i++) {
		bufferSizes[smplMemb[i]]++;
		offsets[i] = i;
	}
	buffers[0] = bufferHolder;
	for (quantity i = 1; i < labelNum; i++) {
		buffers[i] = bufferHolder + bufferSizes[i - 1];
	}
}

ClassDistribution::~ClassDistribution() {
	delete [] bufferSizes;
	delete [] buffers;
	delete [] offsets;
}

void ClassDistribution::refresh(label_id* smplMemb, quantity smplNum) {
	for (quantity i = 0; i < labelNumber; i++) {
		bufferSizes[i] = 0;
	}
	for (quantity i = 0; i < smplNum; i++) {
		label_id label = smplMemb[i];
		buffers[label][bufferSizes[label]] = i;
		bufferSizes[label]++;
		offsets[i] = buffers[label] - bufferHolder;
	}
}

void ClassDistribution::exchange(sample_id u, sample_id v) {
	bufferHolder[offsets[u]] = v;
	bufferHolder[offsets[v]] = u;
	swap(offsets[u], offsets[v]);
}
