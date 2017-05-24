/*
 * mapping.h
 *
 *  Created on: 2017年2月17日
 *      Author: yifan
 */

#ifndef MAPPING_H_
#define MAPPING_H_

#include "model.h"

class Mapping: public Model {
public:
	Dense* FW = 0;
	Dense* RR = 0;
	Dense* S = 0;
	Dense* L = 0;
	Dense* D = 0;
	Dense* W = 0;
	int k;
	float alpha, beta, lambda;
public:
	Mapping(float _alpha, float _beta, float _lambda, int _k, int maxiter,
			int fold) :
			alpha(_alpha), beta(_beta), lambda(_lambda), k(_k), Model(maxiter,
					fold) {
		FW = new Dense;
		RR = new Dense;
		S = new Dense;
		L = new Dense;
		D = new Dense;
		W = new Dense;
	}

	double object();
	void learn();
	void laplace();
	void updateW();
	void updateS();
	void record(const char*);
	~Mapping();
};

#endif /* MAPPING_H_ */
