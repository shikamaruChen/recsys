/*
 * prism.h
 *
 *  Created on: 2017年8月25日
 *      Author: yifan
 */

#ifndef PRISM_H_
#define PRISM_H_

#include "model.h"

class Prism: public Model {
public:
	Dense*S = 0;
	Dense*Sy = 0;
	Dense*FV = 0;
	Dense*RR = 0;
	Dense*Q = 0;
	Dense*V = 0;
	Dense*D = 0;
	Dense*L = 0;
	double alpha, beta, lambda1, lambda2;
	int k;
public:
	Prism(double a, double b, double l1, double l2, int _k, int maxiter,
			int fold) :
			alpha(a), beta(b), lambda1(l1), lambda2(l2), k(_k), Model(maxiter,
					fold) {
		S = new Dense;
		Sy = new Dense;
		FV = new Dense;
		RR = new Dense;
		Q = new Dense;
		V = new Dense;
		D = new Dense;
		L = new Dense;
	}
	virtual double object() = 0;
	void laplace();
	void learn();
	void calcQ();
	virtual void updateS() = 0;
	virtual void udpateV() = 0;
	void record(const char*);
	~Prism();
};

class Prism1: public Prism {
public:
	Prism1(double a, double b, double l1, double l2, int k, int maxiter,
			int fold) :
			Prism(a, b, l1, l2, k, maxiter, fold) {
	}
	double object();
	void updateS();
	void updateV();
};
#endif /* PRISM_H_ */
