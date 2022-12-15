#include <iostream>
#include <vector>
#include "header.cuh"

#define edge_size 512 // Edge 개수

int main(int argc, char *argv[]) {
	srand(time(NULL));

	size_t max_node_idx = 0;
	Edge* edge_list = load_bitcoin_otc();

	for (auto &edge : edge_list) {
		if (edge.from > max_node_idx) {
			max_node_idx = edge.from;
		}
		if (edge.to > max_node_idx) {
			max_node_idx = edge.to;
		}
	}

	cout << "max_node_idx: " << max_node_idx << ", num_edge: " << edge_list.size() << endl;


	return 0;
}
