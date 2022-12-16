#include <bits/stdc++.h>
#include <vector>
#include <iostream>
#include "def.cuh"
#include "host_functions.cuh"

using namespace std;

__host__
size_t naive_counter(const vector<Edge> edges) {
    vector<vector<size_t>> adj_list;

    // assumes maximum index of node is number of nodes
    size_t num_node = 0;
    for (auto &edge : edges) {
        if (edge.src > num_node) {
            num_node = edge.src;
        }
        if (edge.dst > num_node) {
            num_node = edge.dst;
        }
    }
    num_node += 1;

    for (size_t i=0; i<num_node; i++) {
        adj_list.push_back(vector<size_t>());
    }

    for (auto &edge : edges) {
        adj_list[edge.src].push_back(edge.dst);
    }

    size_t count = 0;
    for (size_t i=0; i<num_node; i++) {
        for (size_t _j=0; _j<adj_list[i].size(); _j++) {
            size_t j = adj_list[i][_j];
            for (size_t _k=0; _k<adj_list[j].size(); _k++) {
                size_t k = adj_list[j][_k];
                if (i == k) {
                    continue;
                }

                for (size_t _l=0; _l < adj_list[i].size(); _l++) {
                    size_t l = adj_list[i][_l];
                    if (l == k) {
                        count++;
                        break;
                    }
                }
            }
        }
    }

    return count;
}

__host__
size_t cuda_counter(const vector<Edge> edges) {
	size_t edge_size = edges.size();
	size_t node_size = 0;
	size_t ret;

	Edge *dev_dst_sorted, *dev_src_sorted, *dev_sort_buffer;
	size_t *dev_in_degree, *dev_out_degree;
	size_t *dev_num_candidates, *dev_accumulated_candidates;
	size_t *dev_counter;
	size_t *dev_counter_buffer;

	for (auto &edge : edges) {
		if (edge.src > node_size) {
			node_size = edge.src;
		}
		if (edge.dst > node_size) {
			node_size = edge.dst;
		}
	}
	node_size += 1;

	Edge *host_edges = (Edge *) malloc(sizeof(Edge) * edge_size);
	size_t *host_nodes  = (size_t *) malloc(sizeof(size_t) * node_size);
	size_t *host_counter = (size_t *) malloc(sizeof(size_t) * COUNTER_SIZE);

	cudaMalloc((void **) &dev_dst_sorted,             sizeof(Edge)   * edge_size);
	cudaMalloc((void **) &dev_src_sorted,             sizeof(Edge)   * edge_size);
	cudaMalloc((void **) &dev_sort_buffer,            sizeof(Edge)   * edge_size);
	cudaMalloc((void **) &dev_in_degree,              sizeof(size_t) * node_size);
	cudaMalloc((void **) &dev_out_degree,             sizeof(size_t) * node_size);
	cudaMalloc((void **) &dev_num_candidates,         sizeof(size_t) * node_size);
	cudaMalloc((void **) &dev_accumulated_candidates, sizeof(size_t) * node_size);
	cudaMalloc((void **) &dev_counter,                sizeof(size_t) * COUNTER_SIZE);
	cudaMalloc((void **) &dev_counter_buffer,         sizeof(size_t) * COUNTER_SIZE);

	// load and sort edges
	cudaMemcpy(dev_src_sorted, edges.data(), sizeof(Edge)*edge_size, cudaMemcpyHostToDevice);
	sort_by_dst(dev_src_sorted, dev_dst_sorted, dev_sort_buffer, edge_size);
	stable_sort_by_src(dev_dst_sorted, dev_src_sorted, dev_sort_buffer, edge_size);

	cout << "dst sort" << endl;
	cudaMemcpy(host_edges, dev_dst_sorted, sizeof(Edge)*edge_size, cudaMemcpyDeviceToHost);
	for (size_t i=0; i<edge_size; i++) {
		cout << host_edges[i].src << " " << host_edges[i].dst << " , ";
	}
	cout << endl;

	cout << "src sort" << endl;
	cudaMemcpy(host_edges, dev_src_sorted, sizeof(Edge)*edge_size, cudaMemcpyDeviceToHost);
	for (size_t i=0; i<edge_size; i++) {
		cout << host_edges[i].src << " " << host_edges[i].dst << " , ";
	}
	cout << endl;

	// count node degrees
	count_in_degree(dev_dst_sorted, dev_in_degree, edge_size, node_size);
	count_out_degree(dev_src_sorted, dev_out_degree, edge_size, node_size);

	// calculate accumulated number of candidates
	element_mul(dev_in_degree, dev_out_degree, dev_num_candidates, node_size);
	acc_sum(dev_num_candidates, dev_accumulated_candidates, node_size);

	cout << "indeg" << endl;
	cudaMemcpy(host_nodes, dev_in_degree, sizeof(size_t)*node_size, cudaMemcpyDeviceToHost);
	for (size_t i=0; i<node_size; i++) {
		cout << host_nodes[i] << " ";
	}
	cout << endl;

	cout << "outdeg" << endl;
	cudaMemcpy(host_nodes, dev_out_degree, sizeof(size_t)*node_size, cudaMemcpyDeviceToHost);
	for (size_t i=0; i<node_size; i++) {
		cout << host_nodes[i] << " ";
	}
	cout << endl;

	cout << "can" << endl;
	cudaMemcpy(host_nodes, dev_num_candidates, sizeof(size_t)*node_size, cudaMemcpyDeviceToHost);
	for (size_t i=0; i<node_size; i++) {
		cout << host_nodes[i] << " ";
	}
	cout << endl;

	cout << "acc" << endl;
	cudaMemcpy(host_nodes, dev_accumulated_candidates, sizeof(size_t)*node_size, cudaMemcpyDeviceToHost);
	for (size_t i=0; i<node_size; i++) {
		cout << host_nodes[i] << " ";
	}
	cout << endl;

	// count triangles
	count_triangles(
		dev_dst_sorted, dev_src_sorted,
		dev_in_degree, dev_out_degree,
		dev_accumulated_candidates,
		node_size, edge_size,
		dev_counter
	);

	cout << "cnt" << endl;
	cudaMemcpy(host_counter, dev_counter, sizeof(size_t) * COUNTER_SIZE, cudaMemcpyDeviceToHost);
	for(size_t i=0; i<COUNTER_SIZE; i++) {
		cout << host_counter[i] << " ";
	}
	cout << endl;

	reduce_sum(dev_counter, dev_counter_buffer, COUNTER_SIZE);

	// get count
	cudaMemcpy(&ret, dev_counter_buffer, sizeof(size_t), cudaMemcpyDeviceToHost);

	cout << ret << endl;

	// free
	cudaFree(dev_dst_sorted);
	cudaFree(dev_src_sorted);
	cudaFree(dev_sort_buffer);
	cudaFree(dev_in_degree);
	cudaFree(dev_out_degree);
	cudaFree(dev_num_candidates);
	cudaFree(dev_accumulated_candidates);
	cudaFree(dev_counter);
	cudaFree(dev_counter_buffer);

	free(host_edges);
	free(host_nodes);
	free(host_counter);

	// return triangle count
	return ret;
}
