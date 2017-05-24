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
	float alpha1, alpha2;
	float mu1, mu2;
	float lambda;
public:
	UFSM(float a1, float a2, float m1, float m2, float _lambda, int _l,
			int maxiter, int fold) :
			alpha1(a1), alpha2(a2), mu1(m1), mu2(m2), lambda(_lambda), l(_l), Model(
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
	float predict(int u, int i);
	virtual ~UFSM(); //virtual
};

class UFSMrmse: public UFSM {
public:
	UFSMrmse(float a1, float a2, float m1, float m2, float lambda, int l,
			int maxiter, int fold) :
			UFSM(a1, a2, m1, m2, lambda, l, maxiter, fold) {
	}
	double object();
	void learn();
	void update(int u, int i);
};

class UFSMbpr: public UFSM {
public:
	UFSMbpr(float a1, float a2, float m1, float m2, float lambda, int l,
			int maxiter, int fold) :
			UFSM(a1, a2, m1, m2, lambda, l, maxiter, fold) {
	}
	double object();
	void learn();
	void update(int u, int i, int j);
};

#endif /* UFSM_H_ */
