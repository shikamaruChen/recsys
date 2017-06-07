/*
 * ubsm.cpp
 *
 *  Created on: 2017年5月30日
 *      Author: yifan
 */

#include "fbsm.h"
#include <set>

__global__ void lossfuncKernel(double*v, int*row, int*col, int m, int n,
		double*d);

void FBSM::sample(int u, int&i, int&j) {
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

void FBSM::update(int u, int i, int j) {
	Dense*fi = new Dense;
	Dense*fj = new Dense;
	Dense*fu = new Dense;
	Dense*delta = new Dense;
	Dense*vi = new Dense;
	Dense*vd = new Dense;
	Dense*vu = new Dense;
	Dense*t = new Dense;
	Dense*n = new Dense;
	double r, sigma;
	F->rowVec(fi, i, true);
	F->rowVec(fj, j, true);
	Fu->rowVec(fu, u, true);
	fi->plus(delta, fj, 1.0, -1.0, false, false);
	delta->eTimes(t, d, 1.0);
	r = t->dot(fu);
	fi->eTimes(t, d, 1.0);
	r -= t->dot(fi);

	V->rtimes(vd, delta, 1.0, false, false);
	V->rtimes(vu, fu, 1.0, false, false);
	r += vd->dot(vu);

	V->rtimes(vi, fi, 1.0, false, false);
	r -= vi->dot(vi);
//	printf("r=%f\n", r);
	sigma = exp(-r);
	sigma = sigma / (1 + sigma);
//	delta->rtimes(t1, W, 1.0, true, false);
//	r = t1->dot(fu);
//	fi->rtimes(t1, W, 1.0, true, false);
//	r -= t1->dot(fi);
//update d
	delta->eTimes(n, fu, 1.0);
	fi->eTimes(t, fi, 1.0);
	n->plus(t, 1.0, -1.0, false);
	n->plus(d, sigma, -2 * beta, false);
	d->plus(n, 1.0, alpha1, false);
	//update V
	vd->rtimes(n, fu, 1.0, false, true);
	n->times(vu, delta, 1.0, 1.0, false, true);
	n->times(vi, fi, 1.0, -2.0, false, true);
	n->plus(V, sigma, -2 * lambda, false);
	V->plus(n, 1.0, alpha2, false);
//	delta->rtimes(t1, fu, 1.0, false, true);
//	t1->times(fu, delta, 1.0, 1.0, false, true);
//	t1->times(fi, fi, 1.0, -2.0, false, true);
//	V->rtimes(t2, t1, 1.0, false, false);
//	t2->plus(t1, V, sigma, -2 * lambda, false, false);
//	V->plus(t1, 1.0, alpha2, false);
//	Dense*fi = new Dense;
//		Dense*fj = new Dense;
//		Dense*fu = new Dense;
//		Dense*delta = new Dense;
//		Dense*vi = new Dense;
//		Dense*vd = new Dense;
//		Dense*vu = new Dense;
//		Dense*t = new Dense;
//		Dense*n = new Dense;
	delete fi;
	delete fj;
	delete fu;
	delete delta;
	delete vi;
	delete vd;
	delete vu;
	delete t;
	delete n;
}

void FBSM::learn() {
	int i, j, nnz;
	R->times(Fu, F, false, false);
	d->initial(Nf, 1);
	d->setValue(0.0001);
	V->initial(k, Nf);
	V->setValue(0.0001);
//	predict();
//	pR->print("dataset/pR");
//	printf("obj=%f\n", object());
//	update(0, 0, 1);
//	d->print();
	for (int iter = 1; iter <= maxiter; ++iter) {
		printf("---------- iter:%d ------------\n", iter);
		for (int u = 0; u < Nu; ++u) {
//			printf("u=%d\n", u);
			nnz = R->row[u + 1] - R->row[u];
			if (nnz == 0)
				continue;
			sample(u, i, j);
			update(u, i, j);
		}
//		predict();
//		printf("obj=%f\n", object());
	}
	predict();
//	pR->print("dataset/pR");
//	d->print("dataset/d");
//	V->print("dataset/V");
}

double FBSM::object() {
	Dense*d = new Dense;
	d->initial(1, R->nnz);
	dim3 threadsPerBlock(PER_THREADS, PER_THREADS);
	dim3 numBlocks((Nu + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(Ni + threadsPerBlock.y - 1) / threadsPerBlock.y);
	lossfuncKernel<<<numBlocks, threadsPerBlock>>>(pR->cu_val, R->cu_row_index,
			R->cu_col, Nu, Ni, d->cu_val);
	checkCudaErrors(cudaDeviceSynchronize());
	double obj = -d->sum();
//	printf("dsum=%f\n", obj);
	V->norm2();
	obj += lambda * V->frobenius();
	obj += beta * d->frobenius();
	delete d;
	return obj;
}

void FBSM::predict() {
	Dense*S = new Dense;
	Dense*FV = new Dense;
	Sparse*T1 = new Sparse;
	Sparse*T2 = new Sparse;
	F->times(FV, V, false, true);
//	FV->print();
	FV->rtimes(S, FV, 1.0, false, true);
	F->timesDiag(T1, d, 1.0, false);
	T1->times(T2, F, false, true);
	T2->plus(S, 1.0, 1.0, false);
//	F->timesDiag()
//	F->diagTimes()
//	F->innerTimes(S, W);
	S->setDiagValue(0);
	R->times(pR, S, false, false);
//	pR->print("dataset/pR");
	delete S;
	delete FV;
	delete T1;
	delete T2;
}

void FBSM::record(const char*filename) {
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

void FBSM::print() {
	printf("lambda=%f\n", lambda);
	printf("alpha1=%f\n", alpha1);
	printf("alpha2=%f\n", alpha2);
}

FBSM::~FBSM() {
	if (d)
		delete d;
	if (V)
		delete V;
	if (Fu)
		delete Fu;
}

