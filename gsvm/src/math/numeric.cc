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

#include "numeric.h"

ostream& operator <<(ostream& os, const fvector& vector) {
	os << "[";
	for (size_t i = 0; i < vector.size - 1; ++i) {
		os << fvector_get(&vector, i) << ", ";
	}
	if (vector.size > 0) {
		os << fvector_get(&vector, vector.size - 1);
	}
	os << "]";
	return os;
}

ostream& operator <<(ostream& os, const fvectorv& view) {
	return os << view.vector;
}

fvalue fvector_mean(fvector *vect) {
	return fvalue_mean(vect->data, vect->stride, vect->size);
}

fvalue fvector_sd(fvector *vect) {
	return fvalue_sd(vect->data, vect->stride, vect->size);
}

fvalue fvector_sum(fvector *vect) {
	fvalue *ptr = vect->data;
	size_t size = vect->size;
	size_t stride = vect->stride;
	fvalue sum = 0.0;
	for (size_t i = 0; i < size; i++) {
		sum += *ptr;
		ptr += stride;
	}
	return sum;
}

void fvector_dot(fvector *u, fvector *v, fvalue *res) {
	size_t size = u->size;

	fvalue *uptr = u->data;
	size_t ustride = u->stride;
	fvalue *vptr = v->data;
	size_t vstride = v->stride;

	fvalue sum = 0.0;
	for (size_t i = 0; i < size; i++) {
		sum += *uptr * *vptr;
		uptr += ustride;
		vptr += vstride;
	}
	*res = sum;
}

void _fmatrix_swap_rows(fmatrix *m, size_t i, size_t j) {
	fvalue *iptr = m->data + i * m->tda;
	fvalue *jptr = m->data + j * m->tda;
	size_t msize = m->size2;
	fvalue buff;
	for (size_t i = 0; i < msize; i++) {
		buff = *jptr;
		*jptr++ = *iptr;
		*iptr++ = buff;
	}
}
