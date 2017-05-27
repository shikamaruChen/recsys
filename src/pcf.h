/*
 * pcf.h
 *
 *  Created on: 2017年5月27日
 *      Author: yifan
 */

#ifndef PCF_H_
#define PCF_H_

#include "model.h"

class PCF: public Model {
public:
	Dense*Q;
	Dense*W;
	float alpha, beta;
	int k;

public:
	PCF(float _alpha, float _beta, int _k, int maxiter, int folder) :
			alpha(_alpha), beta(_beta), k(_k), Model(maxiter, folder) {
		Q = new Dense;
		W = new Dense;
	}
	void learn();
	void record(const char*);
	~PCF();
};

#endif /* PCF_H_ */
