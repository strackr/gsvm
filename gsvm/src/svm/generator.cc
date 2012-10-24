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
