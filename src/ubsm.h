/*
 * ubsm.h
 *
 *  Created on: 2017年5月30日
 *      Author: yifan
 */

#ifndef UBSM_H_
#define UBSM_H_

#include "model.h"
class UBSM: public Model {
	Dense*d = 0;
	Dense*V = 0;
	Dense*W = 0;
	Sparse*Fu = 0;
	double alpha1, alpha2, beta, lambda;
	int k;
public:
	UBSM(double b, double l, double a1, double a2, int _k, int maxiter,
			int fold) :
			beta(b), lambda(l), alpha1(a1), alpha2(a2), k(_k), Model(maxiter,
					fold) {
		d = new Dense;
		V = new Dense;
		W = new Dense;
		Fu = new Sparse;
	}
	void learn();
	double object();
	void predict();
	void sample(int u, int&i, int&j);
	void update(int u, int i, int j);
	void record(const char*);
	~UBSM();
};

#endif /* UBSM_H_ */
