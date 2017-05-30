#include "pcf.h"

void PCF::learn() {
	Dense*iR = new Dense;
	Dense*iF = new Dense;
	Dense*H = new Dense;
	Dense*A = new Dense;
	Dense*B = new Dense;
	Dense*T = new Dense;
	Dense*O = new Dense;
	Dense*U = new Dense;
	Dense*V = new Dense;
	Dense*S = new Dense;
	Dense*Pl = new Dense;
	Dense*Pr = new Dense;
	Dense*M = new Dense;
	Dense*W = new Dense;
	double tol = 0.0001;

	R->pinv(iR, tol);//compute inv of R
	//compute A
	H->initial(Ni, Ni);
	H->setIdentity();
	H->plus(1.0f, -1.0f / Ni);
	F->outerTimes(T, H);
	T->plusDiag(1.0f, beta);
	T->pinv(A, tol);
	F->times(T, H, true, false);
	A->rtimes(T, 1.0f, false, false);
	//compute B
	F->times(B, A, false, false);
	B->plusDiag(1.0, -1.0);
	B->ltimes(H, 1.0f, false, false);
	//compute F (replace by T)
	B->rbind(T, A, 1.0, sqrt(beta));
	O->initial(Ni, Ni);
	O->setIdentity();
	T->rbind(O, 1.0, sqrt(k));
	T->pinv(iF, tol);//compute inv of F
//	iF->print();
	//compute Pl and Pr
	project(T, Pl, true);
	R->toDense(A);
	A->transpose(B);
	project(B, Pr, false);
	//compute S (replace by T)
	O->clean();
	O->initial(Ni + Nf, Nu);
	O->setValue(0);
	O->rbind(T, B, 1.0, sqrt(k));
	//compute M
	Pl->rtimes(M, T, 1.0, false, false);
	M->rtimes(Pr, 1.0, false, false);
	M->truncation(k);
	//compute W
	iF->rtimes(W, M, 1.0, false, false);
	W->rtimes(iR, 1.0, false, true);
	R->times(pR, W, false, false);
	delete iF;
	delete iR;
	delete H;
	delete A;
	delete B;
	delete T;
	delete O;
	delete U;
	delete V;
	delete S;
	delete Pl;
	delete Pr;
	delete M;
	delete W;
}

struct proj: public thrust::unary_function<double, double> {
	__host__ __device__
	double operator()(const double &x) const {
		if (x > 0)
			return 1;
		else
			return 0;
	}
};

void PCF::project(Dense*A, Dense*P, bool left) {
	Dense*U = new Dense;
	Dense*V = new Dense;
	Dense*S = new Dense;
	A->svd(U, V, S);
	thrust::transform(S->val, S->val + S->length(), S->val, proj());
	if (left) {
		U->timesDiag(P, S, 1.0, false);
		P->rtimes(U, 1.0, false, true);
	} else {
		V->timesDiag(P, S, 1.0, false);
		P->rtimes(V, 1.0, false, true);
	}
	delete U;
	delete V;
	delete S;
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
}
