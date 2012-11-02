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

#ifndef CACHE_H_
#define CACHE_H_

#include "kernel.h"
#include "../math/random.h"

// uncomment to disable statistics
//#define ENABLE_STATS

#ifdef ENABLE_STATS
#define STATS(c) c
#else
#define STATS(c)
#endif

#define DEFAULT_ETA 1.0

#define VIOLATOR_DISCARD_PROB 0.5
#define VIOLATOR_PROBE_SIZE 200
#define VIOLATOR_MIN_EST_PROP 1.4

#define INITIAL_CACHE_DEPTH 1024
#define INITIAL_ID 0

typedef sample_id row_id;

typedef short_id fold_id;

typedef unsigned long iteration;


struct EntryMapping {

	entry_id cacheEntry;

	EntryMapping() :
		cacheEntry(INVALID_ENTRY_ID) {
	}

};


struct CacheEntry {

	entry_id prev;
	entry_id next;

	entry_id vector;
	sample_id mapping;

	CacheEntry() :
		prev(INVALID_ENTRY_ID),
		next(INVALID_ENTRY_ID),
		vector(INVALID_ENTRY_ID),
		mapping(INVALID_SAMPLE_ID) {
	}

};


struct Stats {

	struct KernelStats {
		quantity kernelEvaluations;
	} kernel;

	Stats() {
		kernel.kernelEvaluations = 0;
	}

};


class SwapListener {

public:
	virtual ~SwapListener() {};

	virtual void notify(sample_id u, sample_id v) = 0;

};


template<typename Kernel, typename Matrix, typename Strategy>
class CachedKernelEvaluator {

	fvalue *kernelValues;
	fvectorv kernelValuesView;

	fvalue *alphas;
	fvectorv alphasView;
	quantity svnumber;

	fvector *fbuffer;
	fvectorv fbufferView;

	fvector *probeBuffer;

	quantity problemSize;
	quantity cacheSize;
	quantity cacheDepth;
	fvalue *cache;

	fvectorv *views;
	sample_id *forwardOrder;
	sample_id *backwardOrder;

	EntryMapping *mappings;
	CacheEntry *entries;

	entry_id lruEntry;

	fvalue w2;

	fvalue eta;

	quantity probeSize;
	quantity minEstimationSize;
	fvalue discardThreshold;

	RbfKernelEvaluator<Kernel, Matrix> *evaluator;
	Strategy *strategy;

	SwapListener *listener;

	Stats stats;

protected:
	CacheEntry& initializeEntry(sample_id v);
	void refreshEntry(sample_id v);

	void updateKernelValues(sample_id u, sample_id v, fvalue beta);
	fvector& evalKernelVector(sample_id v);

	void resizeCache();

	fvalue evalKernel(sample_id uid, sample_id vid);
	void evalKernel(sample_id id, sample_id rangeFrom, sample_id rangeTo, fvector *result);

public:
	CachedKernelEvaluator(RbfKernelEvaluator<Kernel, Matrix> *evaluator, Strategy *strategy,
			quantity probSize, quantity cchSize, SwapListener *listener);
	~CachedKernelEvaluator();

	fvalue evalKernelUV(sample_id u, sample_id v);
	fvalue evalKernelAXV(sample_id v);
	fvalue evalKernelAXVCached(sample_id v);
	bool checkViolation(sample_id v, fvalue threshold);

	fvalue getTau();
	fvalue getVectorWeight(sample_id v);
	quantity getSVNumber();

	sample_id findMaxSVKernelVal(sample_id v);
	sample_id findMinSVKernelVal();

	void performUpdate(sample_id u, sample_id v);
	fvalue getWNorm();
	bool performSvUpdate(fvalue threshold, quantity minViol);

	void setSwapListener(SwapListener *listener);
	void swapSamples(sample_id u, sample_id v);
	void reset();
	void shrink();
	void releaseSupportVectors(fold_id *membership, fold_id fold);
	void setKernelParams(fvalue c, Kernel params);

	Kernel getParams();
	fvalue getC();
	RbfKernelEvaluator<Kernel, Matrix>* getEvaluator();
	fvector* getAlphas();
	fvector* getBuffer();
	sample_id* getBackwardOrder();
	sample_id* getForwardOrder();

	void structureCheck();
	void reportStatistics();
};

template<typename Kernel, typename Matrix, typename Strategy>
CachedKernelEvaluator<Kernel, Matrix, Strategy>::CachedKernelEvaluator(
		RbfKernelEvaluator<Kernel, Matrix> *evaluator, Strategy *strategy,
		quantity probSize, quantity cchSize, SwapListener *listener = NULL) :
		evaluator(evaluator),
		strategy(strategy),
		listener(listener) {
	problemSize = probSize;
	cacheSize = cchSize <= probSize ? cchSize : probSize;
	cacheDepth = INITIAL_CACHE_DEPTH;
	cache = new fvalue[cacheDepth * cacheSize];

	// initialize alphas and kernel values
	alphas = new fvalue[problemSize];
	kernelValues = new fvalue[problemSize];
	for (sample_id i = 0; i < problemSize; i++) {
		alphas[i] = 0.0;
		kernelValues[i] = 0.0;
	}
	alphas[0] = 1.0;
	kernelValues[0] = evaluator->getKernelTau();
	svnumber = 1;
	alphasView = fvectorv_array(alphas, svnumber);
	kernelValuesView = fvectorv_array(kernelValues, svnumber);

	// initialize buffer
	fbuffer = fvector_alloc(problemSize);
	fbufferView = fvector_subv(fbuffer, 0, svnumber);

	// initialize order
	forwardOrder = new sample_id[problemSize];
	backwardOrder = new sample_id[problemSize];
	for (quantity i = 0; i < problemSize; i++) {
		forwardOrder[i] = i;
		backwardOrder[i] = i;
	}

	// initialize vector views
	quantity offset = 0;
	views = new fvectorv[cacheSize];
	for (quantity i = 0; i < cacheSize; i++) {
		views[i] = fvectorv_array(cache + offset, cacheDepth);
		views[i].vector.size = 0;
		offset += cacheDepth;
	}

	// initialize cache mappings
	mappings = new EntryMapping[problemSize];
	fvector *initialVector = &views[INITIAL_ID].vector;
	initialVector->size = 1;
	fvector_set(initialVector, 0, evaluator->getKernelTau());
	mappings[INITIAL_ID].cacheEntry = INITIAL_ID;

	// initialize cache entries
	entries = new CacheEntry[cacheSize];
	for (entry_id i = INITIAL_ID; i < cacheSize; i++) {
		CacheEntry &entry = entries[i];
		entry.prev = i + 1;
		entry.next = i - 1;
		entry.vector = i;
		entry.mapping = i;

		EntryMapping &mapping = mappings[i];
		mapping.cacheEntry = i;
	}
	entries[cacheSize - 1].prev = INITIAL_ID;
	entries[INITIAL_ID].next = cacheSize - 1;

	lruEntry = cacheSize - 1;

	w2 = evaluator->getKernelTau();

	eta = DEFAULT_ETA;

	probeSize = VIOLATOR_PROBE_SIZE;
	minEstimationSize = VIOLATOR_MIN_EST_PROP * probeSize;
	probeBuffer = fvector_alloc(probeSize);
	discardThreshold = gsl_cdf_gaussian_Pinv(VIOLATOR_DISCARD_PROB, 1.0);
}

template<typename Kernel, typename Matrix, typename Strategy>
CachedKernelEvaluator<Kernel, Matrix, Strategy>::~CachedKernelEvaluator() {
	delete evaluator;
	delete listener;
	delete [] cache;
	delete [] views;
	delete [] mappings;
	delete [] entries;
	delete [] alphas;
	delete [] kernelValues;
	fvector_free(fbuffer);
	fvector_free(probeBuffer);
}

template<typename Kernel, typename Matrix, typename Strategy>
fvalue CachedKernelEvaluator<Kernel, Matrix, Strategy>::evalKernelAXV(sample_id v) {
	fvalue result;
	if (v >= svnumber) {
		fvector &vector = evalKernelVector(v);
		fvector_dot(&alphasView.vector, &vector, &result);
		// save current result (valid until alpha values stay unchanged)
		kernelValues[v] = result;
	} else {
		// get precomputed kernel value
		result = kernelValues[v];
		refreshEntry(v);
	}
	return result;
}

template<typename Kernel, typename Matrix, typename Strategy>
bool CachedKernelEvaluator<Kernel, Matrix, Strategy>::checkViolation(sample_id v, fvalue threshold) {
	return evalKernelAXV(v) < threshold;
}

template<typename Kernel, typename Matrix, typename Strategy>
inline fvalue CachedKernelEvaluator<Kernel, Matrix, Strategy>::evalKernelAXVCached(sample_id v) {
	return kernelValues[v];
}

template<typename Kernel, typename Matrix, typename Strategy>
fvector& CachedKernelEvaluator<Kernel, Matrix, Strategy>::evalKernelVector(sample_id v) {
	entry_id mappedEntry = mappings[v].cacheEntry;
	// if not support vector then search the cache
	if (mappedEntry != INVALID_ENTRY_ID) {
		// entry exists in the cache
		CacheEntry &entry = entries[mappedEntry];
		fvector &vector = views[entry.vector].vector;
		fvalue *vdata = vector.data;
		if (vector.size < svnumber) {
			sample_id from = vector.size;
			vector.size = svnumber;

			if (v < svnumber) {
				// TODO remove condition 'final != INVALID_ENTRY_ID'
				sample_id final = svnumber - 1;
				bool updated = true;
				while (final >= from && updated && final != INVALID_ENTRY_ID) {
					updated = false;
					if (final == v) {
						vdata[final] = evaluator->getKernelTau();
						updated = true;
						final--;
					} else {
						entry_id entry = mappings[final].cacheEntry;
						if (entry != INVALID_ENTRY_ID) {
							fvectorv &buf = views[entries[entry].vector];
							if (buf.vector.size > v) {
								vdata[final] = fvector_get(&buf.vector, v);
								updated = true;
								final--;
							}
						}
					}
				}

				if (final >= from && final != INVALID_ENTRY_ID) {
					evalKernel(v, from, final + 1, &vector);
				}
			} else {
				evalKernel(v, from, svnumber, &vector);
			}
		}
		refreshEntry(v);
		return vector;
	} else {
		CacheEntry entry = initializeEntry(v);
		fvector &vector = views[entry.vector].vector;
		vector.size = svnumber;

		evalKernel(v, 0, svnumber, &vector);
		return vector;
	}
}

template<typename Kernel, typename Matrix, typename Strategy>
CacheEntry& CachedKernelEvaluator<Kernel, Matrix, Strategy>::initializeEntry(sample_id v) {
	// entry to be initialized
	CacheEntry &released = entries[lruEntry];

	// clear previous mapping
	EntryMapping &prevMapp = mappings[released.mapping];
	prevMapp.cacheEntry = INVALID_ENTRY_ID;
	released.mapping = v;

	// initialize current mapping
	EntryMapping &currMapp = mappings[v];
	currMapp.cacheEntry = lruEntry;

	// scroll LRU pointer
	lruEntry = released.next;

	return released;
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::refreshEntry(sample_id v) {
	CacheEntry &lru = entries[lruEntry];

	entry_id currentId = mappings[v].cacheEntry;
	if (currentId != INVALID_ENTRY_ID) {
		if (currentId != lruEntry) {
			CacheEntry& current = entries[currentId];

			CacheEntry &prev = entries[current.prev];
			prev.next = current.next;

			CacheEntry &next = entries[current.next];
			next.prev = current.prev;

			CacheEntry &lruPrev = entries[lru.prev];

			current.next = lruEntry;
			current.prev = lru.prev;

			lru.prev = currentId;
			lruPrev.next = currentId;
		} else {
			lruEntry = lru.next;
		}
	}
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::resizeCache() {
	quantity newCacheDepth = 2 * cacheDepth;
	fvalue *newCache = new fvalue[newCacheDepth * cacheSize];
	for (quantity i = 0; i < cacheSize; i++) {
		entry_id id = cacheSize - i - 1;
		fvectorv oldView = views[id];
		size_t oldSize = oldView.vector.size;
		fvectorv newView;
		if (oldSize > 0) {
			newView = fvectorv_array(newCache + id * newCacheDepth, oldView.vector.size);
			fvector_cpy(&newView.vector, &oldView.vector);
		} else {
			newView = fvectorv_array(newCache + id * newCacheDepth, 1);
			newView.vector.size = 0;
		}
		views[id] = newView;
	}
	delete [] cache;
	cache = newCache;
	cacheDepth = newCacheDepth;

	//	shrink();
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::shrink() {
	// removing non-support vectors
	quantity *nonsvnumbers = new quantity[svnumber + 1];
	quantity nonsvnumber = 0;
	sample_id firstViol = 0;
	nonsvnumbers[0] = 0;
	for (quantity i = 1; i <= svnumber; i++) {
		if (alphas[i - 1] <= 0) {
			nonsvnumbers[i] = nonsvnumbers[i - 1] + 1;
			nonsvnumber++;
			if (!firstViol) {
				firstViol = i;
			}
		} else {
			nonsvnumbers[i] = nonsvnumbers[i - 1];
		}
	}
	if (nonsvnumber > 0) {
		// update vector lengths
		for (sample_id i = 0; i < svnumber; i++) {
			if (mappings[i].cacheEntry != INVALID_ENTRY_ID) {
				CacheEntry &entry = entries[mappings[i].cacheEntry];
				fvectorv &vector = views[entry.vector];
				fvalue *data = vector.vector.data;
				for (quantity j = firstViol; j <= vector.vector.size; j++) {
					data[j - nonsvnumbers[j]] = data[j];
				}
				vector.vector.size -= nonsvnumbers[vector.vector.size];
			}
			if (nonsvnumbers[i] > 0) {
				sample_id svid = i - nonsvnumbers[i];
				swapSamples(i, svid);
			}
		}

		for (sample_id i = svnumber; i < problemSize; i++) {
			if (mappings[i].cacheEntry != INVALID_ENTRY_ID) {
				CacheEntry &entry = entries[mappings[i].cacheEntry];
				fvectorv &vector = views[entry.vector];
				vector.vector.size = 0;
			}
		}

		svnumber -= nonsvnumber;
		alphasView.vector.size -= nonsvnumber;
		kernelValuesView.vector.size -= nonsvnumber;
	}
	delete [] nonsvnumbers;
}

template<typename Kernel, typename Matrix, typename Strategy>
fvalue CachedKernelEvaluator<Kernel, Matrix, Strategy>::evalKernelUV(sample_id u, sample_id v) {
	fvalue result;
	if (mappings[v].cacheEntry != INVALID_ENTRY_ID
			&& views[mappings[v].cacheEntry].vector.size > u) {
		result = fvector_get(&views[mappings[v].cacheEntry].vector, u);
	} else if (mappings[u].cacheEntry != INVALID_ENTRY_ID
			&& views[mappings[u].cacheEntry].vector.size > v) {
		result = fvector_get(&views[mappings[u].cacheEntry].vector, v);
	} else {
		// calculate the kernel, do not cache the value
		result = evalKernel(u, v);
	}
	return result;
}

template<typename Kernel, typename Matrix, typename Strategy>
inline fvalue CachedKernelEvaluator<Kernel, Matrix, Strategy>::getTau() {
	return evaluator->getKernelTau();
}

template<typename Kernel, typename Matrix, typename Strategy>
inline fvalue CachedKernelEvaluator<Kernel, Matrix, Strategy>::getVectorWeight(sample_id v) {
	return alphas[v];
}

template<typename Kernel, typename Matrix, typename Strategy>
inline quantity CachedKernelEvaluator<Kernel, Matrix, Strategy>::getSVNumber() {
	return svnumber;
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::performUpdate(sample_id u, sample_id v) {
	fvalue uw = evalKernelAXVCached(u);
	fvalue vw = evalKernelAXVCached(v);
	fvalue uv = evalKernelUV(u, v);

	fvalue tau = evaluator->getKernelTau();

	fvalue beta = eta * 0.5 * (uw - vw) / (tau - uv);

	if (v >= svnumber) {
		// swap rows
		swapSamples(v, svnumber);
		v = svnumber;

		// update cache entries
		fvectorv &vview = views[entries[mappings[v].cacheEntry].vector];
		vview.vector.size++;
		fvector_set(&vview.vector, svnumber, tau);

		// adjust sv number
		svnumber++;
		alphasView.vector.size++;
		kernelValuesView.vector.size++;
	}

	fvalue au = alphas[u];
	if (beta > au) {
		beta = au;
	}

	updateKernelValues(u, v, beta);
	w2 += 2.0 * beta * (vw - uw + beta * (tau - uv));

	alphas[v] += beta;
	alphas[u] -= beta;

	if (svnumber >= cacheDepth) {
//		cout << "resizing the cache (SV#: "<< svnumber << ")" << endl;
		resizeCache();
	}
	//	structureCheck();
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::releaseSupportVectors(fold_id *membership, fold_id fold) {
	quantity valid = 0;
	for (sample_id v = 0; v < svnumber; v++) {
		fvalue beta = alphas[v];
		if (membership[v] != fold && beta > 0) {
			valid++;
		}
	}
	if (valid > 0) {
		for (sample_id v = 0; v < svnumber; v++) {
			fvalue beta = alphas[v];
			if (membership[v] == fold && beta > 0) {
				fvalue rbeta = 1 / (1 - beta);

				// update w2
				fvalue rbeta2 = pow2(rbeta);
				fvalue vw = kernelValues[v];
				fvalue v2 = evaluator->getKernelTau();
				w2 = rbeta2 * w2 - 2 * beta * rbeta2 * vw + pow2(beta) * rbeta2 * v2;

				// update kernel values
				fvector &vvector = evalKernelVector(v);
				fvector_mul_const(&kernelValuesView.vector, 1 / beta);
				fvector_sub(&kernelValuesView.vector, &vvector);
				fvector_mul_const(&kernelValuesView.vector, beta * rbeta);

				// update alphas
				fvector_mul_const(&alphasView.vector, rbeta);
				alphas[v] = 0.0;
			}
		}
	} else {
		sample_id proper = svnumber;
		while (membership[proper] == fold) {
			proper++;
		}
		swapSamples(0, proper);
		reset();
	}
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::setSwapListener(SwapListener *listener) {
	this->listener = listener;
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::swapSamples(sample_id u, sample_id v) {
	evaluator->swapSamples(u, v);
	swap(alphas[u], alphas[v]);
	swap(kernelValues[u], kernelValues[v]);

	strategy->notifyExchange(u, v);
	if (listener) {
		listener->notify(u, v);
	}

	EntryMapping &vmap = mappings[v];
	EntryMapping &umap = mappings[u];
	entry_id temp = vmap.cacheEntry;
	vmap.cacheEntry = umap.cacheEntry;
	umap.cacheEntry = temp;

	if (umap.cacheEntry != INVALID_ENTRY_ID) {
		entries[umap.cacheEntry].mapping = u;
	}
	if (vmap.cacheEntry != INVALID_ENTRY_ID) {
		entries[vmap.cacheEntry].mapping = v;
	}

	forwardOrder[backwardOrder[u]] = v;
	forwardOrder[backwardOrder[v]] = u;
	swap(backwardOrder[u], backwardOrder[v]);
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::reset() {
	// initialize alphas
	for (sample_id i = 1; i < problemSize; i++) {
		alphas[i] = 0.0;
	}
	alphas[INITIAL_ID] = 1.0;
	svnumber = 1;
	alphasView = fvectorv_array(alphas, svnumber);

	// initialize kernelvalues cache
	kernelValuesView = fvectorv_array(kernelValues, svnumber);
	kernelValues[0] = evaluator->getKernelTau();

	for (entry_id i = 0; i < cacheSize; i++) {
		views[i].vector.size = 0;
	}

	w2 = evaluator->getKernelTau();
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::setKernelParams(fvalue c, Kernel gparams) {
	evaluator->setKernelParams(c, gparams);
	reset();
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::updateKernelValues(sample_id u, sample_id v, fvalue beta) {
	fvector &uvector = evalKernelVector(u);
	fvector &vvector = evalKernelVector(v);

	fvalue *uptr = uvector.data;
	fvalue *vptr = vvector.data;
	fvalue *kptr = kernelValues;
	for (sample_id i = 0; i < svnumber; i++) {
		*kptr++ += beta * (*vptr++ - *uptr++);
	}

	// TODO remove blas implementation
//	fbufferView.vector.size = svnumber;
//	fvector_cpy(&fbufferView.vector, &vvector);
//	fvector_sub(&fbufferView.vector, &uvector);
//	fvector_mul_const(&fbufferView.vector, beta);
//	fvector_add(&kernelValuesView.vector, &fbufferView.vector);
}

template<typename Kernel, typename Matrix, typename Strategy>
inline fvalue CachedKernelEvaluator<Kernel, Matrix, Strategy>::getWNorm() {
	return w2;
}

template<typename Kernel, typename Matrix, typename Strategy>
inline bool CachedKernelEvaluator<Kernel, Matrix, Strategy>::performSvUpdate(fvalue threshold, quantity minViol) {
	fvalue minKernel = MAX_FVALUE;
	sample_id minIdx = INVALID_SAMPLE_ID;
	fvalue maxKernel = -MAX_FVALUE;
	sample_id maxIdx = INVALID_SAMPLE_ID;

	quantity violator = 0;
	fvalue *kptr = kernelValues;
	for (sample_id i = 0; i < svnumber; i++) {
		fvalue kernel = *kptr++;
		if (kernel > maxKernel) {
			if (alphas[i] > 0) {
				maxKernel = kernel;
				maxIdx = i;
			}
		}
		if (kernel < minKernel) {
			minKernel = kernel;
			minIdx = i;
		}
		if (kernel < threshold) {
			violator++;
		}
	}

	bool update = false;
	if (violator > minViol && minIdx != maxIdx) {
		performUpdate(maxIdx, minIdx);
		update = true;
	}
	return update;
}

template<typename Kernel, typename Matrix, typename Strategy>
sample_id CachedKernelEvaluator<Kernel, Matrix, Strategy>::findMaxSVKernelVal(sample_id v) {
	sample_id violator;
	fvalue maxGoodness = -MAX_FVALUE;
	fvalue *aptr = alphas;

	fvalue vw = kernelValues[v];
	fvalue av = alphas[v];
	fvalue *kv = evalKernelVector(v).data;
	fvalue tau = evaluator->getKernelTau();

	for (sample_id i = 0; i < svnumber; i++) {
		fvalue uw = kernelValues[i];
		if (uw > vw) {
			fvalue au = alphas[i];
			fvalue uv = kv[i];

			fvalue goodness = strategy->evaluateGoodness(au, av, uv, uw, vw, tau);
			if (goodness > maxGoodness && aptr[i] > 0) {
				maxGoodness = goodness;
				violator = i;
			}
		}
	}
	return violator;
}

template<typename Kernel, typename Matrix, typename Strategy>
sample_id CachedKernelEvaluator<Kernel, Matrix, Strategy>::findMinSVKernelVal() {
	sample_id violator;
	fvalue minKernel = MAX_FVALUE;
	fvalue *kptr = kernelValues;
	for (sample_id i = 0; i < svnumber; i++) {
		fvalue kernel = *kptr++;
		if (kernel < minKernel) {
			minKernel = kernel;
			violator = i;
		}
	}
	return violator;
//	return fvector_min(&kernelValuesView.vector);
}

template<typename Kernel, typename Matrix, typename Strategy>
inline fvalue CachedKernelEvaluator<Kernel, Matrix, Strategy>::evalKernel(sample_id uid, sample_id vid) {
	STATS({
		stats.kernel.kernelEvaluations++;
	});
	return evaluator->evalKernel(uid, vid);
}

template<typename Kernel, typename Matrix, typename Strategy>
inline void CachedKernelEvaluator<Kernel, Matrix, Strategy>::evalKernel(sample_id id,
		sample_id rangeFrom, sample_id rangeTo, fvector *result) {
	STATS({
		stats.kernel.kernelEvaluations += rangeTo - rangeFrom;
	});
	evaluator->evalKernel(id, rangeFrom, rangeTo, result);
}

template<typename Kernel, typename Matrix, typename Strategy>
inline Kernel CachedKernelEvaluator<Kernel, Matrix, Strategy>::getParams() {
	return evaluator->getParams();
}

template<typename Kernel, typename Matrix, typename Strategy>
inline fvalue CachedKernelEvaluator<Kernel, Matrix, Strategy>::getC() {
	return evaluator->getC();
}

template<typename Kernel, typename Matrix, typename Strategy>
inline RbfKernelEvaluator<Kernel, Matrix>* CachedKernelEvaluator<Kernel, Matrix, Strategy>::getEvaluator() {
	return evaluator;
}

template<typename Kernel, typename Matrix, typename Strategy>
inline fvector* CachedKernelEvaluator<Kernel, Matrix, Strategy>::getAlphas() {
	return &alphasView.vector;
}

template<typename Kernel, typename Matrix, typename Strategy>
inline fvector* CachedKernelEvaluator<Kernel, Matrix, Strategy>::getBuffer() {
	return &fbufferView.vector;
}

template<typename Kernel, typename Matrix, typename Strategy>
sample_id* CachedKernelEvaluator<Kernel, Matrix, Strategy>::getBackwardOrder() {
	return backwardOrder;
}

template<typename Kernel, typename Matrix, typename Strategy>
sample_id* CachedKernelEvaluator<Kernel, Matrix, Strategy>::getForwardOrder() {
	return forwardOrder;
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::reportStatistics() {
	logger << format("stats: kernel evaluations: %d\n") % stats.kernel.kernelEvaluations;
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::structureCheck() {
	try {
		entry_id ptr;

		ptr = lruEntry;
		for (quantity i = 0; i < cacheSize; i++) {
			if (ptr > cacheSize || ptr < 0) {
				throw "invalid ptr (next)";
			}
			ptr = entries[ptr].next;
		}
		if (ptr != lruEntry) {
			throw "invalid cache structure (next)";
		}

		ptr = lruEntry;
		for (quantity i = 0; i < cacheSize; i++) {
			if (ptr > cacheSize || ptr < 0) {
				throw "invalid ptr (prev)";
			}
			ptr = entries[ptr].prev;
		}
		if (ptr != lruEntry) {
			throw "invalid cache structure (prev)";
		}

		if (fabs(fvector_sum(&alphasView.vector) - 1.0) > 0.0001) {
			throw "invalid alphas";
		}

		for (entry_id e = 0; e < cacheSize; e++) {
			CacheEntry &entry = entries[e];
			fvectorv vect = views[entry.vector];
			for (size_t i = 0; i < vect.vector.size; i++) {
				fvalue kern = evalKernel(entry.mapping, i);
				fvalue buff = fvector_get(&vect.vector, i);
				if (fabs(kern - buff) > 0.0001) {
					throw "invalid kernel value";
				}
			}
		}

		for (entry_id e = 0; e < cacheSize; e++) {
			CacheEntry &entry = entries[e];
			fvectorv vect = views[entry.vector];
			if (vect.vector.size == svnumber && entry.mapping < svnumber) {
				fvalue kern;
				fvector_dot(&vect.vector, &alphasView.vector, &kern);
				fvalue buff = kernelValues[entry.mapping];
				if (fabs(kern - buff) > 0.0001) {
					throw "invalid buffered kernel value";
				}
			}
		}

		fvalue w2ex;
		fvector_dot(&kernelValuesView.vector, &alphasView.vector, &w2ex);
		if (fabs(w2ex - w2) > 0.0001) {
			throw "invalid w2";
		}
	} catch (char const *ex) {
		cerr << ex << endl;
		throw ex;
	}
}

#endif
