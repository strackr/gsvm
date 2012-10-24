#include "data.h"

template<>
dfmatrix* FeatureMatrixBuilder<dfmatrix>::getFeatureMatrix(
		list<map<feature_id, fvalue>*> &features, map<feature_id, feature_id> mappings) {
	fmatrix *samples = fmatrix_alloc(features.size(), mappings.size());
	sample_id row = 0;

	list<map<feature_id, fvalue>*>::iterator lit;
	for (lit = features.begin(); lit != features.end(); lit++) {
		map<feature_id, fvalue>::iterator mit;
		for (mit = (*lit)->begin(); mit != (*lit)->end(); mit++) {
			fmatrix_set(samples, row, mappings[mit->first], mit->second);
		}
		row++;
	}
	return new DenseMatrix(samples);
}

template<>
sfmatrix* FeatureMatrixBuilder<sfmatrix>::getFeatureMatrix(
		list<map<feature_id, fvalue>*> &features, map<feature_id, feature_id> mappings) {
	// count total number of features
	quantity total = 0;
	list<map<feature_id, fvalue>*>::iterator lit;
	for (lit = features.begin(); lit != features.end(); lit++) {
		total += (*lit)->size();
	}

	// initialize storage
	fvalue *vals = new fvalue[total + features.size()];
	feature_id *feats = new feature_id[total + features.size()];

	id *offsets = new id[features.size()];

	sample_id row = 0;
	id offset = 0;

	for (lit = features.begin(); lit != features.end(); lit++) {
		offsets[row] = offset;
		map<feature_id, fvalue>::iterator mit;
		for (mit = (*lit)->begin(); mit != (*lit)->end(); mit++) {
			feats[offset] = mappings[mit->first];
			vals[offset] = mit->second;
			offset++;
		}
		feats[offset] = INVALID_FEATURE_ID;
		vals[offset] = 0.0;
		offset++;

		row++;
	}

	return new sfmatrix(vals, feats, offsets, features.size(), mappings.size());
}

