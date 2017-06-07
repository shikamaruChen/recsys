/*
 * feature.h
 *
 *  Created on: 2017年2月16日
 *      Author: yifan
 */

#ifndef FEATURE_H_
#define FEATURE_H_

#include "model.h"

class Feature: public Model {
public:
	Dense* FW = 0;
	Dense* S = 0;
	Dense* L = 0;
	Dense* W = 0;
	Dense*RF = 0;
	int k;
	float alpha, beta, lambda, mu;
public:
	Feature(float a, float b, float l, float _mu, int _k, int m, int f) :
			alpha(a), beta(b), lambda(l), mu(_mu), k(_k), Model(m, f) {
		FW = new Dense;
		S = new Dense;
		L = new Dense;
		W = new Dense;
		RF = new Dense;
	}
	void print();
	double object();
	void laplace();
	void learn();
	void updateW();
	void updateS();
	void record(const char*);
	~Feature();

};

#endif /* FEATURE_H_ */
