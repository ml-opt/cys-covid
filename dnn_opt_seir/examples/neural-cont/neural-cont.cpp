#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include <common.h>
#include <dnn_opt.h>

using namespace std;
using namespace std::chrono;

#ifdef ENABLE_CUDA
using namespace dnn_opt::cuda;
#elif ENABLE_COPT
using namespace dnn_opt::copt;
#elif ENABLE_CORE
using namespace dnn_opt::core;
#endif

int main(int argc, char** argv)
{
  /* command line argument collection */

  std::string db_train = input_s("-db", "", argc, argv);
  std::string db_test = input_s("-dbt", "", argc, argv);
  int model = input("-model", 0, argc, argv);

  int p = input("-p", 40, argc, argv);
  int eta = input("-eta", 4000, argc, argv);
  int algorithm_type = input("-a", 0, argc, argv);

  float k = input_f("-k", 2.0f, argc, argv);
  float beta = input_f("-beta", 0.8f, argc, argv);

  int in_dim = input("-in-dim", 99, argc, argv);
  int out_dim = input("-out-dim", 1, argc, argv);

  /* generator that defines the search space */
  auto* generator = generators::uniform::make(-1.0f, 1.0f);
  auto* train = readers::csv_reader::make(db_train, in_dim, out_dim, ' ', true);
  auto* test = readers::csv_reader::make(db_test, in_dim, out_dim, ' ', true);
  auto* mse = errors::mse::make();

  /* set that contains the individuals of the population */
  auto* solutions = set<solutions::net::seq>::make(p);

  for (int i = 0; i < p; i++)
  {
    auto* nn = create_model(model, generator, train, mse);

    solutions->add(nn);
  }

  /* random generation of initial population according the generator */
  solutions->generate();

  /* creating algorithm */
  auto* algorithm = create_algorithm(algorithm_type, solutions);
  auto* cont = algorithms::cont::make(algorithm, solutions, train);

  /* hyper-parameters, see @ref dnn_opt::core::algorithm::set_params() */
  set_hyper(algorithm_type, algorithm, argc, argv);

  std::vector<float> cont_params = {k, beta};
  cont->set_params(cont_params);

  /* optimize for eta iterations */

  float terror = 0;
  float gerror = 0;
  float time = 0;

  auto start = high_resolution_clock::now();
  cont->optimize_eval(eta, []()
  {
    return true;
  });
  auto end = high_resolution_clock::now();

  /* collect statics */

  auto* best = dynamic_cast<solutions::net::seq*>(algorithm->get_best());

  time = duration_cast<milliseconds>(end - start).count();
  terror = best->test(train);
  gerror = best->test(test);

  cout << time << " " << terror << " " << gerror << endl;

  /* delete allocated memory */

  delete solutions->clean();
  delete test, train;
  delete algorithm, cont;
  delete generator;
  delete mse;

  return 0;
}
