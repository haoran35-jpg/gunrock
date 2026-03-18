#include <gunrock/algorithms/scc.hxx>
#include <gunrock/io/parameters.hxx>
#include <gunrock/util/print.hxx>

#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>

using namespace gunrock;
using namespace memory;

int test_scc(int argc, char** argv) {
  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;
  using csr_t = format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
  using csc_t = format::csc_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
  using graph_t = graph::graph_t<memory_space_t::device, vertex_t, edge_t, weight_t, graph::graph_csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>>;

  std::string benchmark_out;
  int num_runs = 3;
  std::vector<std::string> args;
  for (int i = 0; i < argc; ++i) args.push_back(argv[i]);
  for (size_t i = 0; i + 1 < args.size(); ) {
    if (args[i] == "--benchmark") { benchmark_out = args[i + 1]; args.erase(args.begin() + i, args.begin() + i + 2); continue; }
    if (args[i] == "--num_runs") { num_runs = std::stoi(args[i + 1]); args.erase(args.begin() + i, args.begin() + i + 2); continue; }
    ++i;
  }
  std::vector<char*> argv_clean(args.size());
  for (size_t i = 0; i < args.size(); ++i) argv_clean[i] = const_cast<char*>(args[i].c_str());

  gunrock::io::cli::parameters_t arguments((int)argv_clean.size(), argv_clean.data(), "Strongly Connected Components");

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto [properties, coo] = mm.load(arguments.filename);

  csr_t csr;
  if (arguments.binary) {
    csr.read_binary(arguments.filename);
  } else {
    csr.from_coo(coo);
  }

  csc_t csc;
  csc.from_csr(csr);

  format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t> reverse_csr;
  reverse_csr.number_of_rows = csc.number_of_columns;
  reverse_csr.number_of_columns = csc.number_of_rows;
  reverse_csr.number_of_nonzeros = csc.number_of_nonzeros;
  reverse_csr.row_offsets = csc.column_offsets;
  reverse_csr.column_indices = csc.row_indices;
  reverse_csr.nonzero_values = csc.nonzero_values;

  graph_t G_forward = graph::build<memory_space_t::device>(properties, csr);
  graph_t G_reverse = graph::build<memory_space_t::device>(properties, reverse_csr);

  size_t n_vertices = G_forward.get_number_of_vertices();
  size_t n_edges = G_forward.get_number_of_edges();

  auto context = std::make_shared<gcuda::multi_context_t>(0);
  thrust::device_vector<vertex_t> scc_id(n_vertices);
  gunrock::options_t options = arguments.get_options();
  gunrock::scc::param_t<vertex_t> param(options);
  gunrock::scc::result_t<vertex_t> result(scc_id.data().get());

  float avg_ms, min_ms = 0, max_ms = 0;
  double wall_s = 0;
  if (!benchmark_out.empty()) {
    std::vector<float> run_times;
    run_times.reserve(num_runs);
    auto wall_start = std::chrono::steady_clock::now();
    for (int r = 0; r < num_runs; ++r) {
      float elapsed = gunrock::scc::run(G_forward, G_reverse, param, result, context);
      run_times.push_back(elapsed);
    }
    auto wall_end = std::chrono::steady_clock::now();
    wall_s = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end - wall_start).count();
    avg_ms = 0;
    for (float t : run_times) avg_ms += t;
    avg_ms /= run_times.size();
    min_ms = *std::min_element(run_times.begin(), run_times.end());
    max_ms = *std::max_element(run_times.begin(), run_times.end());
  } else {
    avg_ms = gunrock::scc::run(G_forward, G_reverse, param, result, context);
  }

  thrust::host_vector<vertex_t> h_scc_id(n_vertices);
  thrust::copy(scc_id.begin(), scc_id.end(), h_scc_id.begin());
  thrust::host_vector<vertex_t> unique_ids = h_scc_id;
  thrust::sort(unique_ids.begin(), unique_ids.end());
  auto last = thrust::unique(unique_ids.begin(), unique_ids.end());
  size_t num_scc = thrust::distance(unique_ids.begin(), last);

  if (!benchmark_out.empty()) {
    std::ofstream out(benchmark_out, std::ios::app);
    if (out) {
      if (out.tellp() == 0)
        out << "dataset\t|V|\t|E|\ttime_avg_ms\ttime_min_ms\ttime_max_ms\twall_s\tnum_runs\tnum_scc\n";
      out << arguments.filename << "\t" << n_vertices << "\t" << n_edges << "\t" << avg_ms << "\t" << min_ms << "\t" << max_ms << "\t" << wall_s << "\t" << num_runs << "\t" << num_scc << "\n";
    }
    std::cout << arguments.filename << " |V|=" << n_vertices << " |E|=" << n_edges << " time_avg_ms=" << avg_ms << " min=" << min_ms << " max=" << max_ms << " wall_s=" << wall_s << " num_scc=" << num_scc << std::endl;
    return 0;
  }

  std::cout << "=== Input data ===" << std::endl;
  std::cout << "|V| = " << n_vertices << ", |E| = " << n_edges << std::endl;
  std::cout << "=== Algorithm result ===" << std::endl;
  std::cout << "GPU elapsed time (ms): " << avg_ms << std::endl;
  std::cout << "scc_id[] (vertex -> representative):" << std::endl;
  print::head(scc_id, n_vertices > 80 ? 80 : (int)n_vertices, "scc_id");
  if (n_vertices > 80)
    std::cout << "... (" << n_vertices << " vertices total)" << std::endl;
  std::cout << "Number of SCCs: " << num_scc << std::endl;
  std::cout << "SCCs (representative -> vertices):" << std::endl;
  for (size_t i = 0; i < num_scc; ++i) {
    vertex_t rep = unique_ids[i];
    std::cout << "  SCC " << rep << ":";
    for (vertex_t v = 0; v < (vertex_t)n_vertices; ++v)
      if (h_scc_id[v] == rep)
        std::cout << " " << v;
    std::cout << std::endl;
  }
  return 0;
}

int main(int argc, char** argv) {
  return test_scc(argc, argv);
}
