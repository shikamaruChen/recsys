/*
 * high.h
 *
 *  Created on: 2017年6月6日
 *      Author: yifan
 */

#ifndef HIGH_H_
#define HIGH_H_
#include "model.h"

class High: public Model {
public:
	Sparse* RF = 0;
	Dense*P = 0;
	Dense*Q = 0;
	Dense*L = 0;
	Dense*fi = 0;
	Dense*fj = 0;
	Dense*fu = 0;
	Dense*fd = 0;
	Dense*qi = 0;
	Dense*qj = 0;
	Dense*qu = 0;
	Dense*qd = 0;
	double alpha, beta, lambda, mu;
	double alpha1, alpha2;
	double sigma = 0;
	int u, i, j;
	int k;
public:
	High(double a, double b, double l, double m, double a1, double a2, int _k,
			int maxiter, int fold) :
			alpha(a), beta(b), lambda(l), mu(m), alpha1(a1), alpha2(a2), k(_k), Model(
					maxiter, fold) {
		u = i = j = 0;
		RF = new Sparse;
		P = new Dense;
		Q = new Dense;
		L = new Dense;
		fi = new Dense;
		fj = new Dense;
		fu = new Dense;
		fd = new Dense;
		qi = new Dense;
		qj = new Dense;
		qu = new Dense;
		qd = new Dense;
	}
	void sample();
	void initial();
	void graph(Dense*A);
	void laplace();
	void computeP(Dense*dP);
	void computeQ(Dense*dQ);
	double bayesian();
	void predict();
	void learn();
	void record(const char*);
	void print();
	~High();
};

#endif /* HIGH_H_ */
