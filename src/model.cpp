/*
 * model.cpp
 *
 *  Created on: 2017年2月16日
 *      Author: yifan
 */
#include "model.h"

void Model::result() {
	if (LOO) {
		printf("HR5\tHR10\tHR15\tHR20\tARHR5\tARHR10\tARHR15\tARHR20\n");
		printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", HR[0] / test->nnz,
				HR[1] / test->nnz, HR[2] / test->nnz, HR[3] / test->nnz,
				ARHR[0] / test->nnz, ARHR[1] / test->nnz, ARHR[2] / test->nnz,
				ARHR[3] / test->nnz);

	} else {
		printf("REC5\tREC10\tREC15\tREC20\tDCG5\tDCG10\tDCG15\tDCG20\n");
		printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", REC[0] / valid,
				REC[1] / valid, REC[2] / valid, REC[3] / valid, DCG[0] / valid,
				DCG[1] / valid, DCG[2] / valid, DCG[3] / valid);
	}
}

void Model::readR(const char*filename) {
	R->readCSR(filename);
	Nu = R->m;
}

void Model::readF(const char* filename) {
	F->readCSR(filename);
	Ni = F->m;
	Nf = F->n;
	F->rowNorm();
}

void Model::readTest(const char* filename) {
	test->readCSR(filename);
}

void Model::leaveOneOut() {
//	pR->print();
	LOO = true;
	thrust::device_ptr<int> order = pR->sortKeyRow(true);
	int Ns[] = { 5, 10, 15, 20 };
	int u = 0;
	for (int user = 0; user < Nu; ++user) {
		int nnz = test->row[user + 1] - test->row[user];
		if (nnz == 0)
			continue;
		int rank;
		for (rank = 0; rank < 20; ++rank)
			if (test->col[u] == order[rank + user * Ni])
				break;
		u++;
		for (int n = 0; n < 4; ++n)
			if (rank < Ns[n]) {
				HR[n] += 1;
				ARHR[n] += 1.0 / (rank + 1);
			}
	}
	thrust::device_free(order);
}

void Model::crossValidation() {
	thrust::device_ptr<int> order = pR->sortKeyRow(true);
	int Ns[] = { 0, 5, 10, 15, 20 };
	int u = 0;
	for (int user = 0; user < Nu; ++user) {
		if (R->row[user + 1] - R->row[user] == 0)
			continue;
		int start = test->row[user];
		int end = test->row[user + 1];
		int nnz = end - start;
		if (nnz == 0)
			continue;
		valid++;
		std::set<int> test_user;
		for (int t = 0; t < nnz; t++)
			test_user.insert(test->col[start + t]);
//		printf("test items:");
//		for (int item : test_user)
//			printf("%d ", item);

//		int rank;
//		for (rank = 0; rank < 20; ++rank)
//			if (test->col[u] == order[rank + user * Ni])
//				break;
//		u++;
		float recall = 0;
		float dcg = 0;
//		printf("recommend items:");
		for (int n = 0; n < 4; ++n) {
			for (int s = Ns[n]; s < Ns[n + 1]; s++) {
//				std::cout<<order[s + user * Ni]<<" ";
				if (test_user.find(order[s + user * Ni]) != test_user.end()) {
					recall++;
					dcg += log(s + 1) / log(2);
				}
			}
			REC[n] += recall / nnz;
			DCG[n] += dcg / Ns[n + 1];
		}
//		printf("\n");
//			if (rank < Ns[n]) {
//				HR[n] += 1;
//				ARHR[n] += 1.0 / (rank + 1);
//			}
	}
	printf("valid=%d\n", valid);
	thrust::device_free(order);
}

Model::~Model() {
	if (R)
		delete R;
	if (test)
		delete test;
	if (F)
		delete F;
	if (pR)
		delete pR;
}
