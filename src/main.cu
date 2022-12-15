#include <iostream>
#include <vector>
#include "def.cuh"
#include "host_functions.cuh"

#define edge_size 512 // Edge 개수

using namespace std;

int main(int argc, char *argv[]) {
	srand(time(NULL));

	size_t max_node_idx = 0;
	size_t *counter[edge_size]; //edge_size만큼 ..?????
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

	//size_t host_result_in_degree[max_node_idx], host_result_out_degree[max_node_idx];
	size_t *dev_result_in_degree, *dev_result_out_degree;
	size_t *dev_result_mul, *dev_result_sum;
	size_t *dev_counter;

	cudaMalloc((void **)&dev_dst, sizeof(Edge)*edge_size);
	cudaMalloc((void **)&dev_src, sizeof(Edge)*edge_size);
	cudaMalloc((void **)&dev_dst_final, sizeof(Edge)*edge_size);
	cudaMalloc((void **)&buffer, sizeof(Edge)*edge_size);
	cudaMalloc((void **)&dev_src_final, sizeof(Edge)*edge_size);
	cudaMalloc((void **)&dev_result_in_degree, sizeof(int)*max_node_idx);
	cudaMalloc((void **)&dev_result_out_degree, sizeof(int)*max_node_idx);

	cudaMemcpy(dev_dst, edge_list, sizeof(Edge)*edge_size, cudaMemcpyHostToDevice);
	sort_by_dst(dev_dst, dev_dst_final, buffer, edge_list.size());

	cudaMemcpy(dev_src, dev_dst_final, sizeof(Edge)*edge_size, cudaMemcpyDeviceToDevice);
	stable_sort_by_src(dev_src, dev_src_final, buffer, edge_list.size());

	//host_src_result = (Edge*)malloc(sizeof(Edge)*edge_size);
	//cudaMemcpy(host_src_result, dev_result, sizeof(Edge)*edge_size, cudaMemcpyDeviceToHost);
	// 위에까지 sort 함수

	count_in_degree(dev_dst_final, dev_result_in_degree, edge_list.size(), max_node_idx);
	//cudaMemcpy(host_result_in_degree, dev_result_degree, sizeof(host_result_in_degree), cudaMemcpyDeviceToHost);
	
	count_out_degree(dev_src_final, dev_result_out_degree, edge_list.size(), max_node_idx);
	//cudaMemcpy(host_result_out_degree, dev_result_degree, sizeof(host_result_out_degree), cudaMemcpyDeviceToHost);
	//위에까지 count_degree 함수

	cudaMalloc((void **) &dev_result_mul, sizeof(int)*max_node_idx);
	element_mul(dev_result_in_degree, dev_result_out_degree, dev_result_mul, max_node_idx);
	// 각 노드마다 확인해야하는 개수 구하는 부분까지

	
	cudaMalloc((void **) &dev_result_sum, sizeof(int)*max_node_idx);
	acc_sum(dev_result_mul, dev_result_sum, max_node_idx); // 누적합 구하기

	cudaMalloc((void **) &dev_counter, sizeof(counter));
	count_triangles(dev_dst_final, dev_src_final, dev_result_in_degree, dev_result_out_degree, dev_result_sum, max_node_idx, edge_list.size(), dev_counter);
	cudaMemcpy(counter, dev_counter, sizeof(counter), cudaMemcpyDeviceToHost);

	cudaFree(dev_dst);
	cudaFree(dev_src);
	cudaFree(dev_dst_final);
	cudaFree(buffer);
	cudaFree(dev_src_final);
	cudaFree(dev_result_in_degree);
	cudaFree(dev_result_out_degree);
	cudaFree(dev_result_mul);
	cudaFree(dev_result_sum);
	cudaFree(dev_counter);
	
	
	cout << "max_node_idx: " << max_node_idx << ", num_edge: " << edge_list.size() << endl;
    cout << "count: " << naive_counter(edge_list) << endl;

	return 0;
}
