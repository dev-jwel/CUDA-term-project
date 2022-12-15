#include <bits/stdc++.h>
#include <vector>
#include <iostream>
#include "def.cuh"

using namespace std;

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