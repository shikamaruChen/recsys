/*
 * ubsm.cpp
 *
 *  Created on: 2017年5月30日
 *      Author: yifan
 */

#include "ubsm.h"
#include <set>

__global__ void lossfuncKernel(double*v, int*row, int*col, int m, int n,
		double*d);

void UBSM::sample(int u, int&i, int&j) {
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

void UBSM::update(int u, int i, int j) {
	Dense*fi = new Dense;
	Dense*fj = new Dense;
	Dense*fu = new Dense;
	Dense*delta = new Dense;
	Dense*t1 = new Dense;
	Dense*t2 = new Dense;
	double r, sigma;
	F->rowVec(fi, i, true);
	F->rowVec(fj, j, true);
	Fu->rowVec(fu, u, true);
	fi->plus(delta, fj, 1.0, -1.0, false, false);
	delta->rtimes(t1, W, 1.0, true, false);
	r = t1->dot(fu);
	fi->rtimes(t1, W, 1.0, true, false);
	r += t1->dot(fi);
	sigma = exp(-r);
	sigma = sigma / (1 + sigma);
	//update d
	delta->eTimes(t1, fu, 1.0);
	fi->eTimes(t2, fi, 1.0);
	t1->plus(t2, 1.0, -1.0, false);
	t2->plus(t1, d, sigma, -2 * beta, false, false);
	d->plus(t1, 1.0, alpha1, false);
	//update V
	delta->rtimes(t1, fu, 1.0, false, true);
	t1->times(fu, delta, 1.0, 1.0, false, true);
	V->rtimes(t2, t1, 1.0, false, false);
	fi->rtimes(t1, fi, 1.0, false, true);
	t2->times(V, t1, 1.0, -2.0, false, false);
	t2->plus(t1, V, sigma, -2 * lambda, false, false);
	V->plus(t2, 1.0, alpha2, false);
	delete fi;
	delete fj;
	delete fu;
	delete delta;
	delete t1;
	delete t2;
}

void UBSM::learn() {
	int i, j;
	R->times(Fu, F, false, false);
	d->initial(Nf, 1);
	d->setRandom();
	V->initial(k, Nf);
	V->setRandom();
	V->rtimes(W, V, 1.0, true, false);
	W->plusDiag(d, 1.0, 1.0);
	predict();
	update(0, 0, 1);
	d->print();
	V->print();
//	printf("obj=%f\n", object());
//	for (int iter = 1; iter <= maxiter; ++iter) {
//		printf("---------- iter:%d ------------\n", iter);
//		for (int u = 0; u < Nu; ++u) {
//			sample(u, i, j);
//			update(u, i, j);
//		}
//		predict();
//	}

}

double UBSM::object() {
	Dense*d = new Dense;
	d->initial(1, R->nnz);
	dim3 threadsPerBlock(PER_THREADS, PER_THREADS);
	dim3 numBlocks((Nu + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(Ni + threadsPerBlock.y - 1) / threadsPerBlock.y);
	lossfuncKernel<<<numBlocks, threadsPerBlock>>>(pR->cu_val, R->cu_row_index,
			R->cu_col, Nu, Ni, d->cu_val);
	checkCudaErrors(cudaDeviceSynchronize());
	double obj = -d->sum();
	V->norm2();
	obj += lambda * V->frobenius();
	obj += beta * d->frobenius();
	delete d;
	return obj;
}

void UBSM::predict() {
	Dense*S = new Dense;
	F->innerTimes(S, W);
	S->setDiagValue(0);
	R->times(pR, S, false, false);
	delete S;
}

void UBSM::record(const char*filename) {
	FILE*file = fopen(filename, "a");
	if (LOO) {
		printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold, lambda,
				alpha1, alpha2, HR[0] / test->nnz, ARHR[0] / test->nnz,
				HR[1] / test->nnz, ARHR[1] / test->nnz, HR[2] / test->nnz,
				ARHR[2] / test->nnz, HR[3] / test->nnz, ARHR[3] / test->nnz);
		fprintf(file, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold,
				lambda, alpha1, alpha2, HR[0] / test->nnz, ARHR[0] / test->nnz,
				HR[1] / test->nnz, ARHR[1] / test->nnz, HR[2] / test->nnz,
				ARHR[2] / test->nnz, HR[3] / test->nnz, ARHR[3] / test->nnz);
	} else {
		printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold, lambda,
				alpha1, alpha2, REC[0] / test->nnz, REC[1] / test->nnz,
				REC[2] / test->nnz, REC[3] / test->nnz, DCG[0] / test->nnz,
				DCG[1] / test->nnz, DCG[2] / test->nnz, DCG[3] / test->nnz);
		fprintf(file, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold,
				lambda, alpha1, alpha2, REC[0] / test->nnz, REC[1] / test->nnz,
				REC[2] / test->nnz, REC[3] / test->nnz, DCG[0] / test->nnz,
				DCG[1] / test->nnz, DCG[2] / test->nnz, DCG[3] / test->nnz);
	}
	fclose(file);
}

UBSM::~UBSM() {
	if (d)
		delete d;
	if (V)
		delete V;
	if (W)
		delete W;
	if (Fu)
		delete Fu;
}

