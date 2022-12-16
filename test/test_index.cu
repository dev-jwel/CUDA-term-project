#include <stdio.h>
#include "def.cuh"
#include "device_functions.cuh"
#include "host_functions.cuh"

__device__
int compare_int(const void *int1, const void *int2) {
	if (*((int *)int1) < *((int *)int2)) return -1;
	if (*((int *)int1) > *((int *)int2)) return 1;
	return 0;
}

__global__
void test_binary_search(
    int *array, size_t size, int _target, size_t left, size_t right, int *result
) {
    int target = _target;
    *result = 0;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int _left = binary_search(
            array, compare_int, sizeof(int), size, &target, true
        );

        if (left != _left) {
            *result += 1;
        }

        int _right = binary_search(
            array, compare_int, sizeof(int), size, &target, false
        );

        if (right != _right) {
            *result += 2;
        }
    }
}

__global__
void test_start_src(
    const Edge *array, size_t size, size_t src, size_t expected, int *result
) {
    *result = 0;

    size_t res = start_src_node_index_of_edge_list(array, size, src);
    if (res != expected) {
        *result = 1;
    }
}

__global__
void test_start_dst(
    const Edge *array, size_t size, size_t dst, size_t expected, int *result
) {
    *result = 0;

    size_t res = start_dst_node_index_of_edge_list(array, size, dst);
    if (res != expected) {
        *result = 1;
    }
}

__global__
void test_start_node(
    const size_t *array, size_t size, size_t tid, size_t num_threads, size_t expected, int *result
) {
    *result = 0;

    size_t res = start_node_of_candidates(array, size, tid, num_threads);
    if (res != expected) {
        *result = 1;
    }
}

__global__
void test_end_node(
    const size_t *array, size_t size, size_t tid, size_t num_threads, size_t expected, int *result
) {
    *result = 0;

    size_t res = end_node_of_candidates(array, size, tid, num_threads);
    if (res != expected) {
        *result = 1;
    }
}

__global__
void test_has_pair(
    const Edge *array, size_t size, Edge target, bool expected, int *result
) {
    *result = 0;
    bool res = has_pair(array, target, size);
    if (res != expected) {
        *result = 1;
    }
}

#define CHECK_TEST_RESULT(message) { \
    cudaMemcpy(host_result_int, dev_result_int, sizeof(host_result_int), cudaMemcpyDeviceToHost); \
    if ((*host_result_int) != 0) { \
        printf("%s\n", (message)); \
        ret = -1; \
    } \
}

int test_index() {
	int i, ret = 0;

	int host_test_int[10] = {1,1,4,4,4,6,6,6,9,9};
	Edge host_test_src[10] = {
        {0, 1}, {0, 3}, {0, 4}, {3, 2}, {3, 3}, {6, 9}, {6, 1}, {6, 4}, {9, 1}, {9, 2}
    };
    Edge host_test_dst[10] = {
        {0, 1}, {3, 1}, {0, 2}, {9, 2}, {3, 6}, {1, 6}, {4, 6}, {8, 6}, {9, 8}, {1, 8}
    };
    size_t host_test_cand[10] = {0,6,6,7,7,8,23,45,66,66};
	int host_result_int[1];

	int *dev_test_int;
    Edge *dev_test_src;
    Edge *dev_test_dst;
    size_t *dev_test_cand;
    int *dev_result_int;

	cudaMalloc((void **) &dev_test_int, sizeof(host_test_int));
	cudaMalloc((void **) &dev_test_src, sizeof(host_test_src));
	cudaMalloc((void **) &dev_test_dst, sizeof(host_test_dst));
	cudaMalloc((void **) &dev_test_cand, sizeof(host_test_cand));
	cudaMalloc((void **) &dev_result_int, sizeof(host_result_int));

	cudaMemcpy(dev_test_int, host_test_int, sizeof(host_test_int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_test_src, host_test_src, sizeof(host_test_src), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_test_dst, host_test_dst, sizeof(host_test_dst), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_test_cand, host_test_cand, sizeof(host_test_cand), cudaMemcpyHostToDevice);

	test_binary_search <<<1, 1>>> (dev_test_int, 10, 6, 5, 8, dev_result_int);
	cudaMemcpy(host_result_int, dev_result_int, sizeof(host_result_int), cudaMemcpyDeviceToHost);
	if ((*host_result_int) & 1) {
        printf("error on left element selection\n");
        ret = -1;
    }
    if ((*host_result_int) & 2) {
        printf("error on right element selection\n");
        ret = -1;
    }

    test_start_src <<<1, 1>>> (dev_test_src, 10, 3, 3, dev_result_int);
    CHECK_TEST_RESULT("error on start src")

    test_start_dst <<<1, 1>>> (dev_test_dst, 10, 8, 8, dev_result_int);
    CHECK_TEST_RESULT("error on start dst");

    test_start_node <<<1, 1>>> (dev_test_cand, 10, 1, 11, 3, dev_result_int);
    CHECK_TEST_RESULT("error on start node with duplication");

    test_start_node <<<1, 1>>> (dev_test_cand, 10, 1, 2, 8, dev_result_int);
    CHECK_TEST_RESULT("error on start node without duplication");

    test_end_node <<<1, 1>>> (dev_test_cand, 10, 0, 11, 3, dev_result_int);
    CHECK_TEST_RESULT("error on end node with duplication");

    test_end_node <<<1, 1>>> (dev_test_cand, 10, 0, 2, 8, dev_result_int);
    CHECK_TEST_RESULT("error on end node without duplication");

    test_has_pair <<<1, 1>>> (dev_test_src, 10, {9,1}, true, dev_result_int);
    CHECK_TEST_RESULT("error on end node with duplication");

    test_has_pair <<<1, 1>>> (dev_test_src, 10, {9,1}, true, dev_result_int);
    CHECK_TEST_RESULT("error on has pair with existing pair");

    test_has_pair <<<1, 1>>> (dev_test_src, 10, {5,1}, false, dev_result_int);
    CHECK_TEST_RESULT("error on has pair with not existing pair");

	cudaFree(dev_test_int);
	cudaFree(dev_test_src);
	cudaFree(dev_test_dst);
	cudaFree(dev_test_cand);
    cudaFree(dev_result_int);

	return ret;
}
