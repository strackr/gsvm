SOURCES = \
		src/configuration.cc \
		src/launcher.cc \
		src/math/numeric.cc \
		src/math/matrix.cc \
		src/math/matrix_dense.cc \
		src/math/matrix_sparse.cc \
		src/math/random.cc \
		src/svm/kernel.cc \
		src/svm/stop.cc \
		src/svm/generator.cc \
		src/svm/params.cc \
		src/model/selection.cc \
		src/feature/feature.cc \
		src/data/dataset.cc \
		src/data/solver_factory.cc \
		src/time/timer.cc
SOURCES_LOGGING = \
		src/logging/log.cc

bin_PROGRAMS = gsvm
gsvm_SOURCES = ${SOURCES} ${SOURCES_LOGGING} \
		src/gsvm.cc


SOURCES_CHECK = \
		tests/data/dataset_factory_test.cc \
		tests/math/matrix_operations_test.cc \
		tests/svm/classification_test.cc
SORUCES_LOGGING_CHECK = \
		tests/logging/fake_log.cc

check_PROGRAMS = gsvmcheck
gsvmcheck_SOURCES = ${SOURCES} ${SOURCES_CHECK} ${SORUCES_LOGGING_CHECK} \
		tests/test.cc

TESTS = gsvmcheck
		