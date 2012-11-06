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

#include "dataset.h"

DataSet SparseFormatDataSetFactory::createDataSet() {
	map<string, label_id> labelIds;
	list<label_id> sampleLabels;
	list<map<feature_id, fvalue> > sampleFeatures;

	label_id labelCounter = 0;

	while (!input.eof()) {
		string line;
		getline(input, line);
		stringstream lineStream(line);
		if (!line.empty()) {
			// read label
			string label;
			lineStream >> label;
			if (!labelIds.count(label)) {
				labelIds[label] = labelCounter++;
			}
			sampleLabels.push_back(labelIds[label]);

			// read features
			sampleFeatures.push_back(readFeatures(lineStream));
		}
	}

	map<label_id, string> labels = getLabelMap(labelIds);

	return DataSet(labels, sampleLabels, sampleFeatures);
}

map<feature_id, fvalue> SparseFormatDataSetFactory::readFeatures(
		istream& lineStream) {
	map<feature_id, fvalue> features;
	while (!lineStream.eof()) {
		feature_id featureId;
		char delim;
		fvalue value;
		lineStream >> featureId >> delim >> value;
		if (value != 0.0) {
			features[featureId] = value;
		}
	}
	return features;
}

map<label_id, string> SparseFormatDataSetFactory::getLabelMap(
		map<string, label_id> &labelIds) {
	map<label_id, string> labels;
	map<string, label_id>::iterator it;
	for (it = labelIds.begin(); it != labelIds.end(); it++) {
		labels[it->second] = it->first;
	}
	return labels;
}
