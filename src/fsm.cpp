/*
 * FSMrec.cpp
 *
 *  Created on: 2017年5月17日
 *      Author: yifan
 */

#include "fsm.h"

double FSMrec::object() {
	double obj = tR->frobenius() / 2;
	obj += lambda * w->norm1();
	obj += alpha * S->norm1();
	return obj;
}

void FSMrec::partial() {
	Dense*L = new Dense;
	Dense*r = new Dense;
	Dense*Q = new Dense;
	Dense* temp = new Dense;
	R->times(Q, tR, true, false);
	Q->plus(1.0f, alpha);
	Q->clone(L);
	L->eTimes(S);
	L->rowSum(temp);
	temp->diag(L);
	L->plus(Q, -1.0f, 1.0f, false);
	nF->selfTimes(dw, L);
	dw->plus(1.0, lambda);
//	dw->plus(w, 1.0f, lambda, false);
	delete L;
	delete r;
	delete temp;
	delete Q;
}

void FSMrec::tildeCalc() {
	R->times(tR, S, false, true);
	R->plus(tR, -1.0f, 1.0f, false);
}

void FSMrec::prosimCalc() {
	Dense* t = new Dense;
	Sparse*temp = new Sparse;
	w->clone(t);
	t->square_root();
	F->timesDiag(nF, t, 1.0, false);
	nF->rowNorm(t, true);
	t->recip();
	nF->innerTimes(temp);
	temp->toDense(S);
	S->setDiagValue(0.0f);
	F->timesDiag(nF, t, 1.0, true);
	delete t;
	delete temp;
}

void FSMrec::learn() {
	int iter;
	double a;
	double b = 0.1;
	double obj0, obj1;
	double n2;
	double tol = 0.00001;
	int i;
	Dense* w0 = new Dense;
	Dense* w1 = new Dense;
	w->initial(Nf, 1);
	w->setRandom();
	w->clone(w0);
	for (iter = 1; iter <= maxiter; ++iter) {
		prosimCalc();
		tildeCalc();
		obj0 = object();
		printf("---------- iter:%d ------------\n", iter);
		printf("obj=%f\n", obj0);
		//tildeR->plus(Rb, train, &ONE, &NEG_ONE);
		partial();
		a = dw->norm2();
		if (a > 1)
			a = 1;
		a = -a;
		w0->plus(w, dw, 1.0, a, false, false);
		w->project();
		prosimCalc();
		tildeCalc();
		obj1 = object();
//		printf("obj1=%f\n", obj1);
		w->plus(w1, w0, 1.0f, -1.0f, false, false);
		//w0->plus(w2, w, 1.0f, -1.0f);
		n2 = w->dot(dw);
		if (obj1 - obj0 > 0.01 * n2) {
			for (i = 0; i < 20; ++i) {
				a *= b;
				w0->plus(w, dw, 1.0, a, false, false);
				w->project();
				prosimCalc();
				tildeCalc();
				obj1 = object();
//				printf("obj1=%f\n", obj1);
				w->plus(w1, w0, 1.0f, -1.0f, false, false);
				n2 = w1->dot(dw);
				if (obj1 - obj0 <= 0.01 * n2)
					break;
			}
		}
		w->plus(w1, w0, 1.0f, -1.0f, false, false);
		n2 = w1->norm2();
		printf("norm dw=%f\n", n2);
		if (n2 < tol)
			break;
		//if (obj1>=obj0) break;
		w->copyto(w0);
		//printf("norm w=%f\n", w->norm2());
	}
	R->times(pR, S, false, false);
	delete w0;
	delete w1;
}

void FSMrec::record(const char*filename) {
	result();
	FILE*file = fopen(filename, "a");
	if (LOO) {
		fprintf(file, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold,
				alpha, lambda, HR[0] / test->nnz, ARHR[0] / test->nnz,
				HR[1] / test->nnz, ARHR[1] / test->nnz, HR[2] / test->nnz,
				ARHR[2] / test->nnz, HR[3] / test->nnz, ARHR[3] / test->nnz);
	} else {
		fprintf(file, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold,
				alpha, lambda, REC[0] / valid, REC[1] / valid, REC[2] / valid,
				REC[3] / valid, DCG[0] / valid, DCG[1] / valid, DCG[2] / valid,
				DCG[3] / valid);
	}
	fclose(file);
}

void FSMrec::print() {
	printf("alpha=%f\n", alpha);
	printf("lambda=%f\n", lambda);
}

void FSMrec::model(const char*filename) {
	FILE*file = fopen(filename, "w");
	int len = w->length();
	double v;
	for (int i = 0; i < len - 1; i++) {
		v = w->val[i];
		fprintf(file, "%f ", v);
	}
	v = w->val[len - 1];
	fprintf(file, "%f\n", v);
	fclose(file);
}
//nF = new Sparse();
//tR = new Dense();
//tS = new Dense();
//w = new Dense();
//dw = new Dense();
FSMrec::~FSMrec() {
	// TODO Auto-generated destructor stub
	if (nF)
		delete nF;
	if (tR)
		delete tR;
	if (S)
		delete S;
	if (w)
		delete w;
	if (dw)
		delete dw;
}

