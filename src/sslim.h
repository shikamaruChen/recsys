/*
 * sslim.h
 *
 *  Created on: 2017年2月16日
 *      Author: yifan
 */

#ifndef SSLIM_H_
#define SSLIM_H_

#include "model.h"

class SSLIM: public Model {
public:
	Dense* S = 0;
	Dense* Q = 0;
	float alpha, beta, lambda;

public:
	SSLIM(float a, float b, float l, int maxiter, int fold) :
			alpha(a), beta(b), lambda(l), Model(maxiter, fold) {
		S = new Dense;
		Q = new Dense;
	}
	virtual void learn() = 0;
	void print();
	void record(const char*);
	virtual ~SSLIM(); //virtual
};

class SSLIM1: public SSLIM {
public:
	SSLIM1(float a, float b, float l, int maxiter, int fold) :
			SSLIM(a, b, l, maxiter, fold) {
	}
	double object();
	void learn();
};

class SSLIM2: public SSLIM {
public:
	SSLIM2(float a, float b, float l, int maxiter, int fold) :
			SSLIM(a, b, l, maxiter, fold) {
	}
	double object();
	void learn();
};

#endif /* SSLIM_H_ */
