#include "pcf.h"

void PCF::learn() {
	Dense*H = new Dense;
	Dense*T1 = new Dense; //temp matrix
	Dense*T2 = new Dense; //temp matrix
	W->initial(Ni, Ni);
	W->setValue(1.0f);
	H->initial(Ni, Ni);
	H->setIdentity();
	H->plus(1.0f, -1.0f / Ni);
	F->outerTimes(T1, H);
	T1->plusDiag(1.0f, beta);
	T1->inv(T2);
//	T2->print();
	F->times(T1, H, true, false);
//	T1->print();
	T1->rtimes(W, 1.0f, false, false);
	T1->print();
	R->times(Q, T1, false, true);
	Q->transpose();
//	Q->print();
	Q->ltimes(T2, 1.0f, false, false);
//	Q->print();
	delete H;
	delete T1;
	delete T2;
}

void PCF::record(const char*filename) {
	FILE*file = fopen(filename, "a");
	if (LOO) {
		printf("%d\t%f\t%f\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold, alpha,
				beta, k, HR[0] / test->nnz, ARHR[0] / test->nnz,
				HR[1] / test->nnz, ARHR[1] / test->nnz, HR[2] / test->nnz,
				ARHR[2] / test->nnz, HR[3] / test->nnz, ARHR[3] / test->nnz);
		fprintf(file, "%d\t%f\t%f\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold,
				alpha, beta, k, HR[0] / test->nnz, ARHR[0] / test->nnz,
				HR[1] / test->nnz, ARHR[1] / test->nnz, HR[2] / test->nnz,
				ARHR[2] / test->nnz, HR[3] / test->nnz, ARHR[3] / test->nnz);
	} else {
		printf("%d\t%f\t%f\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold, alpha,
				beta, k, REC[0] / valid, REC[1] / valid, REC[2] / valid,
				REC[3] / valid, DCG[0] / valid, DCG[1] / valid, DCG[2] / valid,
				DCG[3] / valid);
		fprintf(file, "%d\t%f\t%f\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold,
				alpha, beta, k, REC[0] / valid, REC[1] / valid, REC[2] / valid,
				REC[3] / valid, DCG[0] / valid, DCG[1] / valid, DCG[2] / valid,
				DCG[3] / valid);
	}
	fclose(file);
}

PCF::~PCF() {
	if (Q)
		delete Q;
	if (W)
		delete W;
}
