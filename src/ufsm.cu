/*
 * ufsm.cpp
 *
 *  Created on: 2017年2月20日
 *      Author: yifan
 */
#include "ufsm.h"
#include <algorithm>
#include <set>
#define THREADS 512
#define PER_THREADS 16

__global__ void lossfuncKernel(double*v, int*row, int*col, int m, int n,
		double*d);
__global__ void predictKernel(int*row1, int*col1, double*v1, int*row2, int*col2,
		double*v2, int*row3, int*col3, double*d, double*tR, int m, int n);

void UFSM::record(const char* filename) {
	FILE*file = fopen(filename, "a");
	if (LOO) {
		printf("%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
				fold, l, lambda, alpha1, alpha2, mu1, mu2, HR[0] / test->nnz,
				ARHR[0] / test->nnz, HR[1] / test->nnz, ARHR[1] / test->nnz,
				HR[2] / test->nnz, ARHR[2] / test->nnz, HR[3] / test->nnz,
				ARHR[3] / test->nnz);
		fprintf(file,
				"%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
				fold, l, lambda, alpha1, alpha2, mu1, mu2, HR[0] / test->nnz,
				ARHR[0] / test->nnz, HR[1] / test->nnz, ARHR[1] / test->nnz,
				HR[2] / test->nnz, ARHR[2] / test->nnz, HR[3] / test->nnz,
				ARHR[3] / test->nnz);
	} else {
		printf("%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
				fold, l, lambda, alpha1, alpha2, mu1, mu2, REC[0] / test->nnz,
				REC[1] / test->nnz, REC[2] / test->nnz, REC[3] / test->nnz,
				DCG[0] / test->nnz, DCG[1] / test->nnz, DCG[2] / test->nnz,
				DCG[3] / test->nnz);
		fprintf(file,
				"%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
				fold, l, lambda, alpha1, alpha2, mu1, mu2, REC[0] / test->nnz,
				REC[1] / test->nnz, REC[2] / test->nnz, REC[3] / test->nnz,
				DCG[0] / test->nnz, DCG[1] / test->nnz, DCG[2] / test->nnz,
				DCG[3] / test->nnz);
	}
	fclose(file);
}

void UFSM::sample(int u, int&i, int&j) {
	std::set<int> pos;
	std::vector<int> neg;
	pos.insert(&R->col[R->row[u]], &R->col[R->row[u + 1]]);
	int nnz = R->row[u + 1] - R->row[u];
	for (int i = 0; i < Ni; ++i)
		if (pos.find(i) == pos.end())
			neg.push_back(i);
	int rn = rand() % nnz;
	i = R->col[R->row[u] + rn];
	rn = rand() % (Ni - nnz);
	j = neg[rn];
}

void UFSM::predict() {
	Dense*MW = new Dense;
	M->rtimes(MW, W, 1.0f, false, true);
	dim3 threadsPerBlock(PER_THREADS, PER_THREADS);
	dim3 numBlocks((Nu + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(Ni + threadsPerBlock.y - 1) / threadsPerBlock.y);
	predictKernel<<<numBlocks, threadsPerBlock>>>(RF->cu_row_index, RF->cu_col,
			RF->cu_val, F->cu_row_index, F->cu_col, F->cu_val, R->cu_row_index,
			R->cu_col, MW->cu_val, pR->cu_val, Nu, Ni);
	checkCudaErrors(cudaDeviceSynchronize());
	delete MW;
}

float UFSM::predict(int u, int i) {
	Dense*f = new Dense;
	Dense*r = new Dense;
	Dense*m = new Dense;
	Dense*v1 = new Dense;
	Dense*v2 = new Dense;
	F->rowVec(f, i);
	R->rowVec(r, u);
	M->getRow(m, u);
	r->setElem(1, i, 0.0f);
	F->times(v1, r, true, true); //F^T r^T
	v1->eTimes(f);
	W->timesVec(v2, m, false);
	float val = v1->dot(v2);
	delete f;
	delete r;
	delete m;
	delete v1;
	delete v2;
	return val;
}

UFSM::~UFSM() {
	if (RF)
		delete RF;
	if (M)
		delete M;
	if (W)
		delete W;
	if (WW)
		delete WW;
}

void UFSMrmse::learn() {
	R->times(RF, F, false, false);
	W->initial(Nf, l);
	M->initial(Nu, l);
	W->setRandom();
	M->setRandom();
	pR->initial(Nu, Ni);
	predict();
	int i1, i2;
	printf("obj=%f\n", object());
	for (int iter = 1; iter <= maxiter; ++iter) {
		printf("---------- iter:%d ------------\n", iter);
		for (int u = 0; u < Nu; ++u) {
			if (R->row[u + 1] - R->row[u] == 0)
				continue;
			sample(u, i1, i2);
//			printf("(%d,%d,%d)\n", u, i1, i2);
			update(u, i1); //positive sample
//			printf("success positive update\n");
			update(u, i2); //negative sample
//			printf("success negative update\n");
		}
		printf("obj=%f\n", object());
	}
	predict();
}

void UFSMrmse::update(int u, int i) {
	Dense*d1 = new Dense;
	Dense*d2 = new Dense;
	Dense*d3 = new Dense;
	Dense*d4 = new Dense;
	Dense*frF = new Dense;
	Dense*f = new Dense;
	Dense*r = new Dense;
	Dense*m = new Dense;
	Dense* mW = new Dense;
	Dense*dWW = new Dense;

	F->rowVec(f, i);
	R->rowVec(r, u);
	M->getRow(m, u);
	W->rtimes(WW, W, 1.0f, true, false);
	r->setElem(i, 0, 0.0f);
	F->times(frF, r, true, true); //F^T r^T Nf\times 1
	frF->eTimes(f);
	float pr = pR->getElem(u, i);
	float tr = R->getElem(u, i);

	frF->rtimes(d1, m, 2 * (pr - tr), false, false); // derivation 1
	frF->rtimes(d2, W, 2 * (pr - tr), true, false); // derivation 2
	d2->plus(m, 1.0f, lambda, false);

	WW->diag(dWW);
	dWW->plus(1.0f, -1.0f);
	W->timesDiag(d3, dWW, 4 * mu1, false); // derivation 3
	WW->setDiagValue(0.0f);
	W->rtimes(d4, WW, 2 * mu2, false, false); // derivation 4

	m->plus(d2, 1.0f, -alpha2, false); // update M
	M->setRow(m, u);
	W->plus(d1, 1.0f, -alpha1, false); // update W
	W->plus(d3, 1.0f, -alpha1, false);
	W->plus(d4, 1.0f, -alpha1, false);

	W->timesVec(mW, m, false); // update rui
	pr = mW->dot(frF);
	pR->setElem(u, i, pr);

	delete d1;
	delete d2;
	delete d3;
	delete d4;
	delete frF;
	delete f;
	delete r;
	delete m;
	delete mW;
	delete dWW;
}

double UFSMrmse::object() {
	Dense*term = new Dense;
	Dense*one = new Dense;
	W->rtimes(WW, W, 1.0f, true, false);
	one->initial(l, 1);
	pR->clone(term);
	R->plus(term, 1.0f, -1.0f, false);
	float nf = term->norm2();
	double obj = nf * nf;
	nf = M->norm2();
	obj += lambda * nf * nf;
	WW->diag(term);
	term->plus(one, 1.0f, -1.0f, false);
	nf = term->norm2();
	obj += mu1 * nf * nf;
	WW->setDiagValue(0.0f);
	nf = WW->norm2();
	obj += mu2 * nf * nf;
	delete term;
	delete one;
	return obj;
}

void UFSMbpr::learn() {
	R->times(RF, F, false, false);
	W->initial(Nf, l);
	M->initial(Nu, l);
	W->setRandom();
	M->setRandom();
	pR->initial(Nu, Ni);
	predict();
	int i, j;
	printf("obj=%f\n", object());
	for (int iter = 1; iter <= maxiter; ++iter) {
		printf("---------- iter:%d ------------\n", iter);
		for (int u = 0; u < Nu; ++u) {
			sample(u, i, j);
			update(u, i, j);
		}
		printf("obj=%f\n", object());
	}
	predict();
}

void UFSMbpr::update(int u, int i, int j) {
	Dense*d1 = new Dense;
	Dense*d2 = new Dense;
	Dense*d3 = new Dense;
	Dense*d4 = new Dense;
	Dense*frF = new Dense;
	Dense*firF = new Dense;
	Dense*fjrF = new Dense;
	Dense*fi = new Dense;
	Dense*fj = new Dense;
	Dense*r = new Dense;
	Dense*m = new Dense;
	Dense* mW = new Dense;
	Dense*dWW = new Dense;

	M->getRow(m, u);
	W->rtimes(WW, W, 1.0f, true, false);

	R->rowVec(r, u);
	F->times(fjrF, r, true, true); //F^T r^T Nf\times 1
	F->rowVec(fj, j);
	fjrF->eTimes(fj);

	r->setElem(i, 0, 0.0f);
	F->times(firF, r, true, true); //F^T r^T Nf\times 1
	F->rowVec(fi, i);
	firF->eTimes(fi);

//	r->setElem(j, 0, 0.0f);

	firF->plus(frF, fjrF, 1.0f, -1.0f, false, false);

	float ri = pR->getElem(u, i);
	float rj = pR->getElem(u, j);
//	printf("ri=%f,rj=%f\n", ri, rj);
	float dr = ri - rj;
	float sigma = exp(-dr);
	sigma = sigma / (1 + sigma);

	frF->rtimes(d1, m, sigma, false, false); // derivation 1
//	d1->print();
	frF->rtimes(d2, W, sigma, true, false);  // derivation 2
	d2->plus(m, 1.0f, -lambda, false);
//	d2->print();
	WW->diag(dWW);
	dWW->plus(1.0f, -1.0f);
	W->timesDiag(d3, dWW, 4 * mu1, false); // derivation 3
	WW->setDiagValue(0.0f);
	W->rtimes(d4, WW, 2 * mu2, false, false); // derivation 4

	m->plus(d2, 1.0f, alpha2, false); // update M
	M->setRow(m, u);
	W->plus(d1, 1.0f, alpha1, false); // update W
	W->plus(d3, 1.0f, -alpha1, false);
	W->plus(d4, 1.0f, -alpha1, false);

	W->timesVec(mW, m, false); // update rui
	ri = mW->dot(firF);
	rj = mW->dot(fjrF);
	pR->setElem(u, i, ri);
	pR->setElem(u, j, rj);

	delete d1;
	delete d2;
	delete d3;
	delete d4;
	delete frF;
	delete firF;
	delete fjrF;
	delete fi;
	delete fj;
	delete r;
	delete m;
	delete mW;
	delete dWW;
}

double UFSMbpr::object() {
	Dense*d = new Dense;
	Dense*term = new Dense;
	Dense*one = new Dense;

	d->initial(1, R->nnz);
	dim3 threadsPerBlock(PER_THREADS, PER_THREADS);
	dim3 numBlocks((Nu + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(Ni + threadsPerBlock.y - 1) / threadsPerBlock.y);
	lossfuncKernel<<<numBlocks, threadsPerBlock>>>(pR->cu_val, R->cu_row_index,
			R->cu_col, Nu, Ni, d->cu_val);
	checkCudaErrors(cudaDeviceSynchronize());

	double obj = -d->sum();
	float nf = M->norm2();
	obj += lambda * nf * nf;
	W->rtimes(WW, W, 1.0f, true, false);

	WW->diag(term);
	one->initial(l, 1);
	one->setValue(1.0f);
	term->plus(one, 1.0f, -1.0f, false);
	nf = term->norm2();
	obj += mu1 * nf * nf;
	WW->setDiagValue(0.0f);
	nf = WW->norm2();
	obj += mu2 * nf * nf;

	delete d;
	delete term;
	delete one;

	return obj;
}

