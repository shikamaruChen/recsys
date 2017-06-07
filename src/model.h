/*
 * model.h
 *
 *  Created on: 2017年2月16日
 *      Author: yifan
 */

#ifndef MODEL_H_
#define MODEL_H_

#include "cumatrix.h"
#include "stopwatch.h"
class Model {
public:
	Sparse* R = 0;
	Sparse* test = 0;
	Sparse* F = 0;
	Dense*pR = 0;

	int maxiter;
	int fold;
	int Nf = 0;
	int Nu = 0;
	int Ni = 0;
	int valid = 0;
	double HR[4];
	double ARHR[4];
	double REC[4];
	double DCG[4];
	bool LOO = false;
public:
	Model(int m, int f) :
			maxiter(m), fold(f) {
		R = new Sparse;
		F = new Sparse;
		test = new Sparse;
		pR = new Dense;
		memset(HR, 0, sizeof(HR));
		memset(ARHR, 0, sizeof(ARHR));
		memset(REC, 0, sizeof(REC));
		memset(DCG, 0, sizeof(DCG));
	}
	virtual void record(const char*) = 0;
	virtual void learn() = 0;
	virtual void print() = 0;
	void result();
	virtual void model(const char*) {
	}
	void readR(const char*);
	void readF(const char*);
	void readTest(const char*);
	void leaveOneOut();
	void crossValidation();
	virtual ~Model();
};

#endif /* MODEL_H_ */
