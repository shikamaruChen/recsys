/*
 * ufsm.h
 *
 *  Created on: 2017年2月20日
 *      Author: yifan
 */

#ifndef UFSM_H_
#define UFSM_H_
#include "model.h"

class UFSM: public Model {
public:
	Sparse* RF = 0;
	Dense* M = 0;
	Dense* W = 0;
	Dense* WW = 0;

	int l;
	double alpha1, alpha2;
	double alpha, beta;
	double lambda;
public:
	UFSM(double a1, double a2, double m1, double m2, double _lambda, int _l,
			int maxiter, int fold) :
			alpha1(a1), alpha2(a2), alpha(m1), beta(m2), lambda(_lambda), l(_l), Model(
					maxiter, fold) {
		RF = new Sparse;
		M = new Dense;
		W = new Dense;
		WW = new Dense;
	}

	void record(const char*);
	virtual void learn() = 0;
	void sample(int u, int&i, int&j);
	void predict();
	double predict(int u, int i);
	virtual ~UFSM(); //virtual
};

class UFSMrmse: public UFSM {
public:
	UFSMrmse(double a1, double a2, double m1, double m2, double lambda, int l,
			int maxiter, int fold) :
			UFSM(a1, a2, m1, m2, lambda, l, maxiter, fold) {
	}
	double object();
	void learn();
	void update(int u, int i);
};

class UFSMbpr: public UFSM {
public:
	UFSMbpr(double a1, double a2, double m1, double m2, double lambda, int l,
			int maxiter, int fold) :
			UFSM(a1, a2, m1, m2, lambda, l, maxiter, fold) {
	}
	double object();
	void learn();
	void update(int u, int i, int j);
};

#endif /* UFSM_H_ */
