#include <stdio.h>
#include <vector>
#include "def.cuh"
#include "host_functions.cuh"

int test_triangle() {
	int ret = 0;
	size_t count;

	std::vector<Edge> host_test_input {
        {0, 1}, {0, 2}, {2, 1}, {0, 3}, {3, 1}, {0, 4}, {4, 1}
    };

	count = naive_counter(host_test_input);
	if (count != 3) {
		printf("error on naive_counter\n");
		printf("count: %d\n", count);
		ret = -1;
	}

	count = cuda_counter(host_test_input);
	if (count != 3) {
		printf("error on cuda_counter\n");
		printf("count: %d\n", count);
		ret = -1;
	}

	return ret;
}
