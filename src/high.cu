/*
 * high.cpp
 *
 *  Created on: 2017年6月6日
 *      Author: yifan
 */
#include "high.h"

__global__ void setkKernel(int*order, double*v, int m, int n, int k);

void High::sample() {
	std::set<int> pos;
	std::vector<int> neg;
	pos.insert(&R->col[R->row[u]], &R->col[R->row[u + 1]]);
	int nnz = R->row[u + 1] - R->row[u];
	for (int t = 0; t < Ni; ++t)
		if (pos.find(t) == pos.end())
			neg.push_back(t);
	int rn = rand() % nnz;
	i = R->col[R->row[u] + rn];
	rn = rand() % (Ni - nnz);
	j = neg[rn];
}

void High::initial() {
	F->rowVec(fi, i, true);
	F->rowVec(fj, j, true);
	RF->rowVec(fu, u, true);
	fi->plus(fd, fj, 1.0, -1.0, false, false);
	Q->getRow(qi, i, true);
	Q->getRow(qj, j, true);
	qi->plus(qd, qj, 1.0, -1.0, false, false);
	Dense*r = new Dense;
	R->rowVec(r, u, true);
//	r->print();
//	Q->print();
	Q->rtimes(qu, r, 1.0, true, false);
	delete r;
}

void High::graph(Dense*A) {
	A->clean();
	Sparse*RR = new Sparse;
	Dense*n = new Dense;
	Dense*t = new Dense;
	R->times(RR, R, true, false);
	R->colNorm(n, false);
	n->pow(2);
	n->repmat(A, Ni, 1);
	A->transpose(t);
	RR->plus(A, -2.0, 1.0, false);
	A->plus(t, 1.0, 1.0, false);
	thrust::device_ptr<int> order = A->sortKeyRow(false);
	A->setValue(0);
	dim3 threadsPerBlock(PER_THREADS, PER_THREADS);
	dim3 numBlocks((Ni + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(Ni + threadsPerBlock.y - 1) / threadsPerBlock.y);
	setkKernel<<<numBlocks, threadsPerBlock>>>(order.get(), A->cu_val, Ni, Ni,
			3);
	checkCudaErrors(cudaDeviceSynchronize());
	thrust::device_free(order);
	delete RR;
	delete n;
	delete t;
}

void High::laplace() {
	Dense*row = new Dense;
	Dense*col = new Dense;
	Dense*v = new Dense;
	Dense*D = new Dense;
	Dense*A = new Dense;
	graph(A);

	A->rowSum(row);
	A->colSum(col);
	row->plus(v, col, 0.5f, 0.5f, true, false);
	v->diag(D);
	D->plus(L, A, 1.0f, -0.5f, false, false);
	L->plus(A, 1.0f, -0.5f, true);	//L -= 0.5 S
	delete A;
	delete D;
	delete row;
	delete col;
	delete v;
}

double High::bayesian() {
	Dense*xu = new Dense;
	Dense*xd = new Dense;
	Dense*xi = new Dense;
	double gamma = 1 - alpha;
	qu->clone(xu);

	xu->times(P, fu, alpha, gamma, true, false);
	qd->clone(xd);
	xd->times(P, fd, alpha, gamma, true, false);
	qi->clone(xi);
	xi->times(P, fi, alpha, gamma, true, false);
	double r = xd->dot(xu) - xi->dot(xi);
//	printf("r=%f\n", r);
	delete xu;
	delete xd;
	delete xi;
	return r;
}

void High::computeP(Dense*dP) {
	Dense*fP = new Dense;
	Dense*PP = new Dense;
//	printf("step 0\n");
	fd->rtimes(dP, qu, 1.0, false, true);
//	printf("step 1\n");
	dP->times(fu, qd, 1.0, 1.0, false, true);
//	printf("step 2\n");
	fu->rtimes(fP, P, 1.0, true, false);
//	printf("step 3\n");
	double a = alpha * (1 - alpha);
	double b = (1 - alpha) * (1 - alpha);
//	printf("step 4\n");
//	dP->print();
//	fd->print();
//	fP->print();
	dP->times(fd, fP, a, b, false, false);
//	printf("step 5\n");
	fd->rtimes(fP, P, 1.0, true, false);
//	printf("step 6\n");
	dP->times(fu, fP, 1.0, b, false, false);
//	printf("step 7\n");
	dP->times(fi, qi, 1.0, -2 * a, false, true);
//	printf("step 8\n");
	fi->rtimes(fP, P, 1.0, true, false);
//	printf("step 9\n");
	dP->times(fi, fP, 1.0, -2 * b, false, false);
//	printf("step 10\n");
	P->rtimes(PP, P, 1.0, true, false);
//	printf("step 11\n");
	dP->times(P, PP, sigma, -mu, false, false);
//	printf("step 12\n");
	dP->plus(P, 1.0, mu - lambda, false);
//	printf("step 13\n");
	delete fP;
	delete PP;
}

void High::computeQ(Dense*dQ) {
	Dense*delta = new Dense;
	Dense*r = new Dense;
	double a = alpha * (1 - alpha);
	double b = alpha * alpha;
	//compute gradient for qt
	P->rtimes(delta, fd, 1.0, true, false);
	delta->plus(qd, a, b, false);
	R->rowVec(r, u, true);
	r->rtimes(dQ, delta, 1.0, false, true);
	//compute gradient for qi
	fu->plus(delta, fi, 1.0, -1.0, false, false);
	delta->plus(fj, 1.0, -1.0, false);
	delta->ltimes(P, 1.0, false, true);
	delta->plus(qu, a, b, false);
	delta->plus(qi, 1.0, -b, false);
	delta->plus(qj, 1.0, -b, false);
	dQ->setRow(delta, i);
	//compute gradient for qj
	P->rtimes(delta, fu, 1.0, true, false);
	delta->plus(qu, -a, -b, false);
	dQ->setRow(delta, j);
//	dQ->print("dataset/Q");
//	printf("sigma=%f\n", sigma);
	dQ->times(L, Q, sigma, -beta, false, false);
	dQ->plus(Q, 1.0, -lambda, false);
	delete delta;
	delete r;
}

void High::predict() {
	Dense*factor = new Dense;
	Dense*S = new Dense;
	F->times(factor, P, false, false);
	double gamma = 1 - alpha;
	factor->plus(Q, gamma, alpha, false);
	factor->rtimes(S, factor, 1.0, false, true);
//	S->print("dataset/S");
	S->setDiagValue(0.0);
	R->times(pR, S, false, false);
	delete factor;
	delete S;
}

void High::learn() {
	R->times(RF, F, false, false);
	P->initial(Nf, k);
	Q->initial(Ni, k);
	P->setRandom();
	Q->setRandom();
	laplace();
	Dense*dP = new Dense;
	Dense*dQ = new Dense;
	double r;
//	sample();
//	initial();
//	r = bayesian();
//	sigma = exp(-r);
//	sigma = sigma / (1 + sigma);
//	computeP(dP);
//	computeQ(dQ);
////	printf("r=%f\n",r);
//	dQ->print("dataset/Q");
//	dP->print("dataset/P");
	Stopwatch watch;
	for (int iter = 1; iter <= maxiter; ++iter) {
		printf("---------- iter:%d ------------\n", iter);
		for (u = 0; u < Nu; ++u) {
			if (R->row[u + 1] - R->row[u] == 0)
				continue;
//			printf("step 0\n");
			sample();

//			printf("step 1\n");
			initial();

//			printf("step 2\n");

			r = bayesian();
//			printf("step 3\n");
			sigma = exp(-r) / (1 + exp(-r));
			if (sigma != sigma)
				sigma = 1;
//			printf("step 4\n");
//			watch.resume();
			computeP(dP);
//			watch.pause();
//			printf("step 5\n");
			computeQ(dQ);
//			printf("step 6\n");
			P->plus(dP, 1.0, alpha1, false);
//			printf("step 7\n");
			Q->plus(dQ, 1.0, alpha2, false);
		}
	}
//	dP->print("dataset/P");
//	dQ->print("dataset/Q");
//	watch.stop();
//	printf("time=%f\n",watch.get_time());
	delete dP;
	delete dQ;
	predict();
}

void High::record(const char*filename) {
	result();
	FILE*file = fopen(filename, "a");
	if (LOO) {
		fprintf(file,
				"%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
				fold, k, alpha, beta, lambda, mu, alpha1, alpha2,
				HR[0] / test->nnz, ARHR[0] / test->nnz, HR[1] / test->nnz,
				ARHR[1] / test->nnz, HR[2] / test->nnz, ARHR[2] / test->nnz,
				HR[3] / test->nnz, ARHR[3] / test->nnz);
	} else {
		fprintf(file,
				"%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
				fold, k, alpha, beta, lambda, mu, alpha1, alpha2,
				REC[0] / valid, REC[1] / valid, REC[2] / valid, REC[3] / valid,
				DCG[0] / valid, DCG[1] / valid, DCG[2] / valid, DCG[3] / valid);
	}
	fclose(file);
}

void High::print() {
	printf("k=%d\n", k);
	printf("alpha=%f\n", alpha);
	printf("beta=%f\n", beta);
	printf("lambda=%f\n", lambda);
	printf("mu=%f\n", mu);
	printf("alpha1=%f\n", alpha1);
	printf("alpha2=%f\n", alpha2);
}
//Sparse* RF = 0;
//Dense*P = 0;
//Dense*Q = 0;
//Dense*fi = 0;
//Dense*fj = 0;
//Dense*fu = 0;
//Dense*fd = 0;
//Dense*qi = 0;
//Dense*qj = 0;
//Dense*qu = 0;
//Dense*qd = 0;
High::~High() {
	if (RF)
		delete RF;
	if (L)
		delete L;
	if (P)
		delete P;
	if (Q)
		delete Q;
	if (fi)
		delete fi;
	if (fj)
		delete fj;
	if (fu)
		delete fu;
	if (fd)
		delete fd;
	if (qi)
		delete qi;
	if (qj)
		delete qj;
	if (qu)
		delete qu;
	if (qd)
		delete qd;
}
