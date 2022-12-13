#include "header.cuh"
#include <bits/stdc++.h>

using namespace std;

// 그래프 개수(일단 예시로)
#define V 4

void multiply(int A[][V], int B[][V], int C[][V])
{
	for (int i = 0; i < V; i++)
	{
		for (int j = 0; j < V; j++)
		{
			C[i][j] = 0;
			for (int k = 0; k < V; k++)
				C[i][j] += A[i][k]*B[k][j];
		}
	}
}

int naive_counter(const Edge *edges) {
	// TODO
}