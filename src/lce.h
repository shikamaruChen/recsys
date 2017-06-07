/*
 * lce.h
 *
 *  Created on: 2017年2月20日
 *      Author: yifan
 */

#ifndef LCE_H_
#define LCE_H_
#include "model.h"
class LCE: public Model {
public:
	Dense*D = 0;
	Dense*S = 0;
	Dense*L = 0;
	Dense*U = 0;
	Dense*V = 0;
	Dense*W = 0;
	Dense*VV = 0;
	float alpha, beta, lambda;
	int k;
public:
	LCE(float _alpha, float _beta, float _lambda, int _k, int maxiter, int fold) :
			alpha(_alpha), beta(_beta), lambda(_lambda), k(_k), Model(maxiter,
					fold) {
		D = new Dense;
		S = new Dense;
		L = new Dense;
		U = new Dense;
		V = new Dense;
		W = new Dense;
		VV = new Dense;
	}
	void print();
	void record(const char*);
	void learn();
	void laplace(int);
	double object();
	void updateW();
	void updateU();
	void updateV();
	void predict();
	~LCE();
};

#endif /* LCE_H_ */
