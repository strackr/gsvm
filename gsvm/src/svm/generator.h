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

#ifndef GENERATOR_H_
#define GENERATOR_H_

#include "../math/numeric.h"
#include "../math/random.h"
#include "params.h"


enum GeneratorType {
	PLAIN,
	FAIR,
	DETERMINISTIC
};

template<GeneratorType Type>
class CandidateIdGenerator {

public:
	CandidateIdGenerator(TrainParams &params, quantity labelNumber, label_id *labels, quantity sampleNumber);

	id nextId();
	void markCorrect();
	void markFailed();

	void exchange(id u, id v);
	void reset(id maxId);

};

struct IdNode {

	id value;
	id next;
	quantity succeeded;
	quantity failed;

	IdNode() : value(INVALID_ID), next(INVALID_ID), succeeded(1), failed(0) {
	}

};

template<>
class CandidateIdGenerator<DETERMINISTIC> {

	id bucketNumber;
	id currentBucket;

	quantity *bucketSizes;
	IdNode *nodes;
	id *heads;
	id *tails;

	void move(id from, id to);
	void initialize(quantity sampleNumber);

public:
	CandidateIdGenerator(TrainParams &params, quantity labelNumber, label_id *labels, quantity sampleNumber);
	~CandidateIdGenerator();

	id nextId();
	void markCorrect();
	void markFailed();

	void exchange(id u, id v);
	void reset(label_id *labels, id maxId);

};

inline CandidateIdGenerator<DETERMINISTIC>::CandidateIdGenerator(TrainParams& params,
		quantity labelNumber, label_id *labels, quantity sampleNumber) {
	bucketNumber = params.generator.bucketNumber;
	bucketSizes = new quantity[bucketNumber];
	heads = new id[bucketNumber];
	tails = new id[bucketNumber];
	nodes = new IdNode[sampleNumber];

	initialize(sampleNumber);
}

inline CandidateIdGenerator<DETERMINISTIC>::~CandidateIdGenerator() {
	delete [] bucketSizes;
	delete [] nodes;
	delete [] heads;
}

inline void CandidateIdGenerator<DETERMINISTIC>::initialize(quantity sampleNumber) {
	// TODO do not reset the number of failures
	currentBucket = 0;
	bucketSizes[0] = sampleNumber;
	for (quantity i = 1; i < bucketNumber; i++) {
		bucketSizes[i] = 0;
		heads[i] = INVALID_ID;
		tails[i] = INVALID_ID;
	}
	heads[0] = 0;
	tails[0] = sampleNumber - 1;
	for (id i = 0; i < sampleNumber; i++) {
		nodes[i].next = i + 1;
		nodes[i].value = i;
		nodes[i].succeeded = 1;
		nodes[i].failed = 0;
	}
	nodes[sampleNumber - 1].next = INVALID_ID;
}

inline void CandidateIdGenerator<DETERMINISTIC>::move(id from, id to) {
	id lastId = heads[from];
	id nextId = nodes[lastId].next;
	heads[from] = nextId;
	if (nextId == INVALID_ID) {
		tails[from] = INVALID_ID;
	}
	if (tails[to] == INVALID_ID) {
		heads[to] = lastId;
	} else {
		nodes[tails[to]].next = lastId;
	}
	tails[to] = lastId;
	nodes[lastId].next = INVALID_ID;
}

inline id CandidateIdGenerator<DETERMINISTIC>::nextId() {
	while (heads[currentBucket] == INVALID_ID) {
		currentBucket = (currentBucket + 1) % bucketNumber;
	}
	return nodes[heads[currentBucket]].value;
}

inline void CandidateIdGenerator<DETERMINISTIC>::markCorrect() {
	IdNode &node = nodes[heads[currentBucket]];
	node.succeeded++;

	quantity offset = min(node.failed / node.succeeded + 1, bucketNumber - 1);
	move(currentBucket, (currentBucket + offset) % bucketNumber);
}

inline void CandidateIdGenerator<DETERMINISTIC>::markFailed() {
	IdNode &node = nodes[heads[currentBucket]];
	node.failed++;

	quantity offset = min(node.failed / node.succeeded + 1, bucketNumber - 1);
	move(currentBucket, (currentBucket + offset) % bucketNumber);
}

inline void CandidateIdGenerator<DETERMINISTIC>::exchange(id u, id v) {
	swap(nodes[u].value, nodes[v].value);
}

inline void CandidateIdGenerator<DETERMINISTIC>::reset(label_id *labels, id maxId) {
	initialize(maxId);
}


struct ClassDistribution {

	quantity labelNumber;
	quantity *bufferSizes;
	id **buffers;
	id *bufferHolder;
	id *offsets;

	ClassDistribution(quantity labelNum, label_id *smplMemb, quantity smplNum);
	~ClassDistribution();

	void refresh(label_id *smplMemb, quantity smplNum);
	void exchange(sample_id u, sample_id v);

};

template<>
class CandidateIdGenerator<FAIR> {

	IdGenerator generator;
	ClassDistribution distr;

public:
	CandidateIdGenerator(TrainParams &params, quantity labelNumber, label_id *labels, quantity sampleNumber);

	id nextId();
	void markCorrect();
	void markFailed();

	void exchange(id u, id v);
	void reset(label_id *labels, id maxId);

};

inline CandidateIdGenerator<FAIR>::CandidateIdGenerator(TrainParams& params,
		quantity labelNumber, label_id *labels, quantity sampleNumber) :
		generator(Generators::create()),
		distr(ClassDistribution(labelNumber, labels, sampleNumber)) {
}

inline id CandidateIdGenerator<FAIR>::nextId() {
	label_id label = generator.nextId(distr.labelNumber);
	id offset = generator.nextId(distr.bufferSizes[label]);
	return distr.buffers[label][offset];
}

inline void CandidateIdGenerator<FAIR>::markCorrect() {
	// do nothing
}

inline void CandidateIdGenerator<FAIR>::markFailed() {
	// do nothing
}

inline void CandidateIdGenerator<FAIR>::exchange(id u, id v) {
	distr.exchange(u, v);
}

inline void CandidateIdGenerator<FAIR>::reset(label_id *labels, id maxId) {
	distr.refresh(labels, maxId);
}


template<>
class CandidateIdGenerator<PLAIN> {

	IdGenerator generator;
	id maxId;

public:
	CandidateIdGenerator(TrainParams &params, quantity labelNumber, label_id *labels, quantity sampleNumber);

	id nextId();
	void markCorrect();
	void markFailed();

	void exchange(id u, id v);
	void reset(label_id *labels, id maxId);

};

inline CandidateIdGenerator<PLAIN>::CandidateIdGenerator(TrainParams& params,
		quantity labelNumber, label_id *labels, quantity sampleNumber) :
		generator(Generators::create()),
		maxId(0) {
}

inline id CandidateIdGenerator<PLAIN>::nextId() {
	return generator.nextId(maxId);
}

inline void CandidateIdGenerator<PLAIN>::markCorrect() {
	// do nothing
}

inline void CandidateIdGenerator<PLAIN>::markFailed() {
	// do nothing
}

inline void CandidateIdGenerator<PLAIN>::exchange(id u, id v) {
	// do nothing
}

inline void CandidateIdGenerator<PLAIN>::reset(label_id *labels, id maxId) {
	this->maxId = maxId;
}

#endif
