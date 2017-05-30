/*
 * FSMrec.h
 *
 *  Created on: 2017年5月17日
 *      Author: yifan
 */

#ifndef FSM_H_
#define FSM_H_
#include "model.h"
class FSMrec: public Model {
public:
	Sparse*nF = 0;
	Dense* tR = 0;
	Dense* S = 0;
	Dense* w = 0;
	Dense* dw = 0;
	Dense* L = 0;
	float lambda = 0;
	float alpha = 0;
public:
	FSMrec(float a, float l, int maxiter, int fold) :
			alpha(a), lambda(l), Model(maxiter, fold) {
		nF = new Sparse();
		tR = new Dense();
		S = new Dense();
		w = new Dense();
		dw = new Dense();
	}
	;
	void prosimCalc();
	void tildeCalc();
	double object();
	void partial();
	void learn();
	void record(const char*);
	void model(const char*);
	~FSMrec();
};

#endif /* FSMREC_H_ */
