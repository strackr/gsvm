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

#include "solver_factory.h"

template<>
dfmatrix* FeatureMatrixBuilder<dfmatrix>::getFeatureMatrix(
		list<map<feature_id, fvalue> >& features,
		map<feature_id, feature_id>& mappings) {
	fmatrix *samples = fmatrix_alloc(features.size(), mappings.size());
	sample_id row = 0;

	list<map<feature_id, fvalue> >::iterator lit;
	for (lit = features.begin(); lit != features.end(); lit++) {
		map<feature_id, fvalue>::iterator mit;
		for (mit = lit->begin(); mit != lit->end(); mit++) {
			fmatrix_set(samples, row, mappings[mit->first], mit->second);
		}
		row++;
	}
	return new DenseMatrix(samples);
}

template<>
sfmatrix* FeatureMatrixBuilder<sfmatrix>::getFeatureMatrix(
		list<map<feature_id, fvalue> >& features,
		map<feature_id, feature_id>& mappings) {
	// count total number of features
	quantity total = 0;
	list<map<feature_id, fvalue> >::iterator lit;
	for (lit = features.begin(); lit != features.end(); lit++) {
		total += lit->size();
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
		for (mit = lit->begin(); mit != lit->end(); mit++) {
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

