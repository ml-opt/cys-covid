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
  int output_size = input("-output-size", 1, argc, argv);
  int model = input("-model", 0, argc, argv);

  int p = input("-p", 40, argc, argv);
  int big_eta = input("-big-eta", 10, argc, argv);
  int small_eta = input("-small-eta", 100, argc, argv);
  int fix_layers = input("-fix-layers", 1, argc, argv);
  int algorithm_type = input("-a", 0, argc, argv);

  int in_dim = input("-in-dim", 99, argc, argv);
  int out_dim = input("-out-dim", 1, argc, argv);

  auto* generator = generators::uniform::make(-1.0f, 1.0f);
  auto* error = errors::mse::make();

  std::vector<reader*> train_sets;
  std::vector<reader*> test_sets;
  std::vector<set<solutions::net::seq>*> solutions_sets;
  std::vector<algorithm*> algorithms;

  for(int i = 0; i < output_size; i++)
  {
    auto* train = readers::csv_reader::make(db_train + "/" + std::to_string(i) + ".csv",
                                            in_dim, out_dim, ' ', true);
    auto* test = readers::csv_reader::make(db_test + "/" + std::to_string(i) + ".csv",
                                           in_dim, out_dim, ' ', true);
    auto* solutions = set<solutions::net::seq>::make(p);

    for (int i = 0; i < p; i++)
    {

      solutions->add(create_model(model, generator, train, error));
    }

    solutions->generate();

    auto* algorithm = create_algorithm(algorithm_type, solutions);

    set_hyper(algorithm_type, algorithm, argc, argv);

    train_sets.push_back(train);
    test_sets.push_back(test);
    solutions_sets.push_back(solutions);
    algorithms.push_back(algorithm);
  }

  int size = 0;
  solutions::net::seq* net = solutions_sets.at(0)->get(0);

  for(int i = 0; i < fix_layers; i++)
  {
    size += net->get_layers().at(i)->size();
  }

cout << "Model size is: " << net->size() << endl;
cout << "Model shared parameters are: " << size << endl;
cout <<"==============================================================" << endl;

cout << "Current best results are: " <<endl;
for(int i = 0; i < output_size; i++)
{
  auto* best = dynamic_cast<solutions::net::seq*>(algorithms[i]->get_best());
  float terror = best->test(train_sets[i]);
  float gerror = best->test(test_sets[i]);

  cout << "\t" << terror << " " << gerror << " " << algorithms[i]->get_solutions()->fitness() << endl;
}
cout <<"==============================================================" << endl;
cout << "Starting iterations..." << endl;

  for(int i = 0; i < big_eta; i++)
  {
    cout << "Epoch: " << std::to_string(i) << endl;

    for(int j = 0; j < output_size; j++)
    {
      cout << "\t Solution: " << std::to_string(j) << endl;

      int next = (j + 1) % output_size;

      algorithms.at(j)->optimize_eval(small_eta, []()
      {
        return true;
      });

      for(int k = 0; k < p; k++)
      {
        solution* src = algorithms.at(j)->get_solutions()->get(k);
        solution* dst = algorithms.at(next)->get_solutions()->get(k);

        std::copy_n(src->get_params(), size, dst->get_params());

        dst->set_modified(true);
      }
    }
  }

  for(int j = 0; j < output_size; j++)
  {
    algorithms.at(j)->optimize_eval(small_eta, []()
    {
      return true;
    });
  }

  cout <<"==============================================================" << endl;
  cout << "Current best results are:" << endl;
  cout <<"==============================================================" << endl;

  for(int i = 0; i < output_size; i++)
  {
    auto* best = dynamic_cast<solutions::net::seq*>(algorithms[i]->get_best());
    float terror = best->test(train_sets[i]);
    float gerror = best->test(test_sets[i]);

    cout << "\t" << terror << " " << gerror << " " << algorithms[i]->get_solutions()->fitness() << endl;
  }

   cout <<"==============================================================" << endl;

  /* delete allocated memory */

  for(int i = 0; i < output_size; i++)
  {
    delete solutions_sets[i]->clean();
    delete train_sets[i];
    delete test_sets[i];
    delete algorithms[i];
  }
  
  delete generator;
  delete error;

  return 0;
}
