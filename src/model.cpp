/*
 * model.cpp
 *
 *  Created on: 2017年2月16日
 *      Author: yifan
 */
#include "model.h"
#include <fstream>

void Model::result() {
	if (LOO) {
		printf("HR5\tHR10\tHR15\tHR20\tARHR5\tARHR10\tARHR15\tARHR20\n");
		printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", HR[0], HR[1], HR[2], HR[3],
				ARHR[0], ARHR[1], ARHR[2], ARHR[3]);

	} else {
		printf("REC5\tREC10\tREC15\tREC20\tDCG5\tDCG10\tDCG15\tDCG20\n");
		printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", REC[0], REC[1], REC[2],
				REC[3], DCG[0], DCG[1], DCG[2], DCG[3]);
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
//	R->plus(pR, -10.0, 1.0, false);
	std::set<int> colset;
	std::ofstream file("check", std::ios::out);
	int n;
	for (n = 0; n < test->nnz; ++n)
		colset.insert(test->col[n]);

	int*col = new int[colset.size()];
	n = 0;
	file << "test items:";
	for (std::set<int>::iterator iter = colset.begin(); iter != colset.end();
			++iter) {
		file << (*iter) << " ";
		col[n++] = (*iter);
	}
	file << std::endl;
	Dense*tR = new Dense;
	pR->keepCol(tR, col, n);
//	tR->print("tR");
	delete[] col;
	thrust::device_ptr<int> order = tR->sortKeyRow(true);

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
		file << "user:" << user << std::endl;
		file << "testset:";
		for (int t = 0; t < nnz; t++) {
			test_user.insert(test->col[start + t]);
			file << test->col[start + t] << " ";
		}
		file << std::endl;
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
		file << "recommend:";
		for (int i = 0; i < 4; ++i) {
			for (int s = Ns[i]; s < Ns[i + 1]; s++) {
				file << order[s + user * Ni];
				if (test_user.find(order[s + user * Ni]) != test_user.end()) {
					recall++;
					if (s == 0)
						dcg += 1;
					else
						dcg += log(2) / log(s + 1);
					file << ":" << s;
				}
				file << " ";
			}
			REC[i] += recall / Ns[i + 1];
			DCG[i] += dcg / Ns[i + 1];
		}
		file << std::endl;
		file << std::endl;
//		printf("\n");
//			if (rank < Ns[n]) {
//				HR[n] += 1;
//				ARHR[n] += 1.0 / (rank + 1);
//			}
	}
	file.close();
	printf("valid=%d\n", valid);
	thrust::device_free(order);
	delete tR;
	for (int i = 0; i < 4; ++i) {
		REC[i] /= valid;
		DCG[i] /= valid;
		HR[i] /= test->nnz;
		ARHR[i] /= test->nnz;
	}
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
