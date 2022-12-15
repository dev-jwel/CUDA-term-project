#include <iostream>
#include <vector>
#include "def.cuh"
#include "host_functions.cuh"

#define edge_size 512 // Edge 개수

using namespace std;

int main(int argc, char *argv[]) {
	srand(time(NULL));

	size_t max_node_idx = 0;
	Edge *dev_dst, *dev_dst_final, *buffer;
	Edge *dev_src, *dev_src_final;
	Edge *host_src_result;
	vector<Edge> edge_list = load_bitcoin_otc();

	for (auto &edge : edge_list) {
		if (edge.src > max_node_idx) {
			max_node_idx = edge.src;
		}
		if (edge.dst > max_node_idx) {
			max_node_idx = edge.dst;
		}
	}

	size_t host_result_in_degree[max_node_idx], host_result_out_degree[max_node_idx];
	size_t *dev_result_degree;

	cudaMalloc((void **)&dev_dst, sizeof(edge_list));
	cudaMalloc((void **)&dev_src, sizeof(edge_list));
	cudaMalloc((void **)&dev_dst_final, sizeof(edge_list));
	cudaMalloc((void **)&buffer, sizeof(edge_list));
	cudaMalloc((void **)&dev_src_final, sizeof(edge_list));

	cudaMemcpy(dev_dst, edge_list, sizeof(edge_list), cudaMemcpyHostToDevice);
	sort_by_dst(dev_dst, dev_dst_final, buffer, edge_list.size());

	cudaMemcpy(dev_src, dev_dst_final, sizeof(edge_list), cudaMemcpyDeviceToDevice);
	stable_sort_by_src(dev_src, dev_src_final, buffer, edge_list.size());

	host_src_result = (edge_list*)malloc(sizeof(edge_list));
	cudaMemcpy(host_src_result, dev_result, sizeof(edge_list), cudaMemcpyDeviceToHost);
	// 위에까지 sort 함수

	cudaMalloc((void **) &dev_result_degree, sizeof(host_result_in_degree));

	count_in_degree(dev_dst_final, dev_result_degree, edge_list, max_node_idx);
	cudaMemcpy(host_result_in_degree, dev_result_degree, sizeof(host_result_in_degree), cudaMemcpyDeviceToHost);
	
	count_out_degree(dev_src_final, dev_result_degree, edge_list, max_node_idx);
	cudaMemcpy(host_result_out_degree, dev_result_degree, sizeof(host_result_out_degree), cudaMemcpyDeviceToHost);
	//위에까지 count_degree 함수

	cout << "max_node_idx: " << max_node_idx << ", num_edge: " << edge_list.size() << endl;
    cout << "count: " << naive_counter(edge_list) << endl;

	return 0;
}
