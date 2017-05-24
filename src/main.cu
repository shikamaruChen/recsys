#include <stdio.h>
#include <unistd.h>
#include <string>
#include "sslim.h"
#include "mapping.h"
#include "feature.h"
#include "ufsm.h"
#include "lce.h"
#include "content.h"
#include "fsm.h"

cusparseHandle_t Sparse::handle;
cublasHandle_t Dense::handle;
cusolverDnHandle_t Dense::solver_handle;
cusparseMatDescr_t Sparse::descr;
cusparseMatDescr_t Sparse::descr_L;
cusparseMatDescr_t Sparse::descr_U;

#define PATH_SIZE 100

float alpha = 0.7f;
float beta = 0.1f;
float lambda = 0.1f;
float mu = 0.1f;
float alpha1 = 0.000001;
float alpha2 = 0.000001;
float mu1 = 0.1f;
float mu2 = 0.1f;
int k = 100;
int l = 1;
int maxiter = 20;
int fold = 1;
int method_type = 1;
int train_type = 0;
bool out_model = false;
std::string dir = "dataset";
std::string resdir = "";
std::string featurefile = "feature";
std::string methodname;

void usage() {
	printf("Recsys implement by Yifan Chen\n");
	printf("---------------I am a splitter----------------\n");
	printf("options:\n");
	printf(" -h, show this usage help\n");
	printf(
			" -H, if defined, then evaluate with HR and ARHR; otherwise, with precision and recall\n");
	printf("-T, algorithms\n");
	printf(" -f, specify the fold (default 1)\n");
	printf("parameters:\n");
	printf(" -a, alpha (default 0.1)\n");
	printf(" -b, beta (default 0.1)\n");
	printf(" -l, lambda (default 2)\n");
	printf(" -M, max number of iterations (default 10)\n");
	printf(" -d, path to working directory\n");
}

void print() {
	printf("Algorithm: %s\n", methodname);
	printf("parameter setting:\n");
	printf("maxiter=%d\n", maxiter);
	printf("alpha=%f\n", alpha);
	printf("beta=%f\n", beta);
	printf("lambda=%f\n", lambda);
	printf("mu=%f\n", mu);
	printf("alpha1=%f\n", alpha1);
	printf("alpha2=%f\n", alpha2);
	printf("mu1=%f\n", mu1);
	printf("mu2=%f\n", mu2);
	printf("iteration for %d times\n", maxiter);
	printf(
			"-----------------------------dataset-----------------------------\n");
	printf("dataset directory=%s\n", dir.c_str());
	printf("evaluated on fold%d\n", fold);
}

void initial() {
	checkCudaErrors(cublasCreate(&Dense::handle));
	checkCudaErrors(cusparseCreate(&Sparse::handle));
	checkCudaErrors(cusolverDnCreate(&Dense::solver_handle));

	checkCudaErrors(cusparseCreateMatDescr(&Sparse::descr));
	checkCudaErrors(
			cusparseSetMatType(Sparse::descr, CUSPARSE_MATRIX_TYPE_GENERAL));
	checkCudaErrors(
			cusparseSetMatIndexBase(Sparse::descr, CUSPARSE_INDEX_BASE_ZERO));

	checkCudaErrors(cusparseCreateMatDescr(&Sparse::descr_L));
	checkCudaErrors(
			cusparseSetMatIndexBase(Sparse::descr_L, CUSPARSE_INDEX_BASE_ZERO));
	checkCudaErrors(
			cusparseSetMatType(Sparse::descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
	checkCudaErrors(
			cusparseSetMatFillMode(Sparse::descr_L, CUSPARSE_FILL_MODE_LOWER));
	checkCudaErrors(
			cusparseSetMatDiagType(Sparse::descr_L, CUSPARSE_DIAG_TYPE_UNIT));

	checkCudaErrors(cusparseCreateMatDescr(&Sparse::descr_U));
	checkCudaErrors(
			cusparseSetMatIndexBase(Sparse::descr_U, CUSPARSE_INDEX_BASE_ZERO));
	checkCudaErrors(
			cusparseSetMatType(Sparse::descr_U, CUSPARSE_MATRIX_TYPE_GENERAL));
	checkCudaErrors(
			cusparseSetMatFillMode(Sparse::descr_U, CUSPARSE_FILL_MODE_UPPER));
	checkCudaErrors(
			cusparseSetMatDiagType(Sparse::descr_U,
					CUSPARSE_DIAG_TYPE_NON_UNIT));

}

void clean() {
	if (Dense::handle)
		checkCudaErrors(cublasDestroy(Dense::handle));
	if (Sparse::handle)
		checkCudaErrors(cusparseDestroy(Sparse::handle));
	if (Dense::solver_handle)
		checkCudaErrors(cusolverDnDestroy(Dense::solver_handle));
	if (Sparse::descr)
		checkCudaErrors(cusparseDestroyMatDescr(Sparse::descr));
	if (Sparse::descr_L)
		checkCudaErrors(cusparseDestroyMatDescr(Sparse::descr_L));
	if (Sparse::descr_U)
		checkCudaErrors(cusparseDestroyMatDescr(Sparse::descr_U));
	cudaDeviceReset();
}

void run(int argc, char* argv[]) {
	int opt;
	if (argc < 2) {
		usage();
		return;
	}
	while ((opt = getopt(argc, argv, "a:b:E:k:l:L:f:F:M:m:d:o:T:1:2:O")) != -1) {
		switch (opt) {
		case 'a':
			alpha = atof(optarg);
			break;
		case 'b':
			beta = atof(optarg);
			break;
		case 'E':
			train_type = atoi(optarg);
			break;
		case 'd':
			dir = optarg;
			break;
		case 'f':
			fold = atoi(optarg);
			break;
		case 'F':
			featurefile = optarg;
			break;
		case 'k':
			k = atoi(optarg);
			break;
		case 'M':
			maxiter = atoi(optarg);
			break;
		case 'L':
			l = atoi(optarg);
			break;
		case 'm':
			mu = atof(optarg);
			break;
		case 'o':
			resdir = optarg;
			break;
		case 'O':
			out_model = true;
			break;
		case 'l':
			lambda = atof(optarg);
			break;
		case 'T':
			method_type = atoi(optarg);
			break;
		case '1':
			mu1 = atof(optarg);
			break;
		case '2':
			mu2 = atof(optarg);
			break;
		case 'h':
		default:
			usage();
			return;
		}
	}
	if (resdir.length() == 0)
		resdir = dir;

	initial();
	Model* recsys;

	switch (method_type) {
	case 1:
		recsys = new Mapping(alpha, beta, lambda, k, maxiter, fold);
		methodname = "mapping";
		break;
	case 2:
		recsys = new Feature(alpha, beta, lambda, mu, k, maxiter, fold);
		methodname = "feature";
		break;
	case 3: //sslim1
		recsys = new SSLIM1(alpha, beta, lambda, maxiter, fold);
		methodname = "sslim1";
		break;
	case 4: //sslim2
		recsys = new SSLIM2(alpha, beta, lambda, maxiter, fold);
		methodname = "sslim2";
		break;
	case 5: //ufsmr
		recsys = new UFSMrmse(alpha1, alpha2, mu1, mu2, lambda, l, maxiter,
				fold);
		methodname = "ufsmr";
		break;
	case 6: //ufsmb
		recsys = new UFSMbpr(alpha1, alpha2, mu1, mu2, lambda, l, maxiter,
				fold);
		methodname = "ufsmb";
		break;
	case 7: //lce
		recsys = new LCE(alpha, beta, lambda, k, maxiter, fold);
		methodname = "lce";
		break;
	case 8: //CoSim
		recsys = new Content(maxiter, fold);
		methodname = "cosim";
		break;
	case 9: //FSMrec
		recsys = new FSMrec(alpha, lambda, maxiter, fold);
		methodname = "fsm";
		break;
	}
	print();
	std::string::size_type i;
	std::string::size_type idx = dir.find("/");
	while ((i = dir.find("/", idx + 1)) != std::string::npos)
		idx = i;
	std::string dataname = dir.substr(idx + 1);
	std::string path;
	path = dir + "/" + featurefile;
	recsys->readF(path.c_str());

	if (train_type == 0) { //validation
		path = dir + "/split/train" + std::to_string(fold);
		recsys->readR(path.c_str());
		recsys->learn();
		path = dir + "/split/valid" + std::to_string(fold);
		recsys->readTest(path.c_str());
		recsys->crossValidation();
	} else if (train_type == 1) { //test with split fold
		path = dir + "/split/train" + std::to_string(fold);
		recsys->readR(path.c_str());
		recsys->learn();
		path = dir + "/split/test" + std::to_string(fold);
		recsys->readTest(path.c_str());
		recsys->crossValidation();
	} else { //test with leave-one-out
		path = dir + "/loo/train" + std::to_string(fold);
		recsys->readR(path.c_str());
		recsys->learn();
		path = dir + "/loo/test" + std::to_string(fold);
		recsys->readTest(path.c_str());
		recsys->leaveOneOut();
	}

	path = resdir + "/result_" + dataname + "_" + methodname;
	recsys->record(path.c_str());
	path = resdir + "/model_" + dataname + "_" + methodname;
	if (out_model)
		recsys->model(path.c_str());
	delete recsys;
	clean();
}

void test() {
	initial();
	Sparse* s = new Sparse;
	s->readCSR("dataset/s");
	s->printFull();
	Dense* d = new Dense;
	d->input("dataset/v2");
	d->print();
	Sparse* r = new Sparse;
	s->diagTimes(r, d, true);
	r->printFull();
	clean();
}
//-H -d /home/yifan/dataset/nips -a 0.5 -k 100 -M 100
int main(int argc, char* argv[]) {
	run(argc, argv);
//	test();
}

