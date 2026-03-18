/**
 * @file scc.cuh
 * @brief Unit test for Strongly Connected Components (multi-round BFS + trim).
 */

#include <gunrock/algorithms/scc.hxx>
#include <gunrock/formats/formats.hxx>
#include <gunrock/graph/graph.hxx>

#include <gtest/gtest.h>

using namespace gunrock;
using namespace memory;

TEST(algorithm, scc_two_cycles) {
  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  // Two disjoint cycles: 0->1->2->0 and 3->4->3
  vertex_t n = 5;
  edge_t nnz = 5;
  thrust::device_vector<edge_t> row_offsets = std::vector{0, 1, 2, 3, 4, 5};
  thrust::device_vector<vertex_t> col_indices = std::vector{1, 2, 0, 4, 3};
  thrust::device_vector<weight_t> values(nnz, 1.0f);

  format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t> csr(n, n, nnz);
  csr.row_offsets = row_offsets;
  csr.column_indices = col_indices;
  csr.nonzero_values = values;

  format::csc_t<memory_space_t::device, vertex_t, edge_t, weight_t> csc;
  csc.from_csr(csr);

  format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t> reverse_csr;
  reverse_csr.number_of_rows = csc.number_of_columns;
  reverse_csr.number_of_columns = csc.number_of_rows;
  reverse_csr.number_of_nonzeros = csc.number_of_nonzeros;
  reverse_csr.row_offsets = csc.column_offsets;
  reverse_csr.column_indices = csc.row_indices;
  reverse_csr.nonzero_values = csc.nonzero_values;

  graph::graph_properties_t properties;
  properties.directed = true;
  auto G_forward = graph::build<memory_space_t::device>(properties, csr);
  auto G_reverse = graph::build<memory_space_t::device>(properties, reverse_csr);

  auto context = std::make_shared<gcuda::multi_context_t>(0);
  thrust::device_vector<vertex_t> scc_id(n);

  scc::param_t<vertex_t> param;
  scc::result_t<vertex_t> result(scc_id.data().get());
  scc::run(G_forward, G_reverse, param, result, context);

  thrust::host_vector<vertex_t> h_scc_id(scc_id);
  // Same SCC => same representative: {0,1,2} get one id, {3,4} get another
  EXPECT_EQ(h_scc_id[0], h_scc_id[1]);
  EXPECT_EQ(h_scc_id[1], h_scc_id[2]);
  EXPECT_EQ(h_scc_id[3], h_scc_id[4]);
  EXPECT_NE(h_scc_id[0], h_scc_id[3]);

  thrust::sort(h_scc_id.begin(), h_scc_id.end());
  auto last = thrust::unique(h_scc_id.begin(), h_scc_id.end());
  EXPECT_EQ(thrust::distance(h_scc_id.begin(), last), 2);
}
