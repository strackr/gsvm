SOURCES = \
		src/configuration.cc \
		src/configuration.h \
		src/launcher.cc \
		src/launcher.h \
		src/math/numeric.cc \
		src/math/numeric.h \
		src/math/matrix.cc \
		src/math/matrix.h \
		src/math/matrix_dense.cc \
		src/math/matrix_dense.h \
		src/math/matrix_sparse.cc \
		src/math/matrix_sparse.h \
		src/math/random.cc \
		src/math/random.h \
		src/math/average.h \
		src/svm/cache.h \
		src/svm/classify.h \
		src/svm/kernel.cc \
		src/svm/kernel.h \
		src/svm/solver.h \
		src/svm/pairwise_solver.h \
		src/svm/universal_solver.h \
		src/svm/stop.cc \
		src/svm/stop.h \
		src/svm/strategy.h \
		src/svm/validation.h \
		src/svm/violation.h \
		src/svm/generator.cc \
		src/svm/generator.h \
		src/svm/params.cc \
		src/svm/params.h \
		src/model/selection.cc \
		src/model/selection.h \
		src/feature/feature.cc \
		src/feature/feature.h \
		src/data/dataset.cc \
		src/data/dataset.h \
		src/data/solver_factory.cc \
		src/data/solver_factory.h \
		src/time/timer.cc \
		src/time/timer.h
SOURCES_LOGGING = \
		src/logging/log.cc \
		src/logging/log.h

bin_PROGRAMS = gsvm
gsvm_SOURCES = ${SOURCES} ${SOURCES_LOGGING} \
		src/gsvm.cc \
		src/gsvm.h


SOURCES_CHECK = \
		tests/data/dataset_factory_test.cc \
		tests/math/matrix_operations_test.cc \
		tests/math/matrix_utils.h \
		tests/svm/classification_test.cc
SORUCES_LOGGING_CHECK = \
		tests/logging/fake_log.cc

check_PROGRAMS = gsvmcheck
gsvmcheck_SOURCES = ${SOURCES} ${SOURCES_CHECK} ${SORUCES_LOGGING_CHECK} \
		tests/test.cc

TESTS = gsvmcheck

EXTRA_DIST = gsvm.spec		
