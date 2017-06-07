/*
 * content.h
 *
 *  Created on: 2017年2月21日
 *      Author: yifan
 */

#ifndef CONTENT_H_
#define CONTENT_H_
#include "model.h"

class Content: public Model {
public:
	Content(int maxiter, int fold) :
			Model(maxiter, fold) {
	}
	void print() {
	}
	void learn();
	void record(const char*);
};

void Content::learn() {
	Dense*S = new Dense;
	Sparse*FF = new Sparse;
	F->innerTimes(FF);
	FF->toDense(S);
	S->setDiagValue(0.0f);
	R->times(pR, S, false, false);
	delete S;
}

void Content::record(const char*filename) {
	printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold, HR[0] / test->nnz,
			ARHR[0] / test->nnz, HR[1] / test->nnz, ARHR[1] / test->nnz,
			HR[2] / test->nnz, ARHR[2] / test->nnz, HR[3] / test->nnz,
			ARHR[3] / test->nnz);
	FILE*file = fopen(filename, "a");
	fprintf(file, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", fold,
			HR[0] / test->nnz, ARHR[0] / test->nnz, HR[1] / test->nnz,
			ARHR[1] / test->nnz, HR[2] / test->nnz, ARHR[2] / test->nnz,
			HR[3] / test->nnz, ARHR[3] / test->nnz);
	fclose(file);
}

#endif /* CONTENT_H_ */
