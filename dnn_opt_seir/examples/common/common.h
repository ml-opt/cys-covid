/*
Copyright (c) 2018, Jairo Rojas-Delgado <jrdelgado@uci.cu>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the <organization> nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
 * This file contain common functions that are used in examples.
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @version 1.0
 * @date October, 2018
 */

#include <iostream>
#include <vector>
#include <dnn_opt.h>

using namespace std;

#ifdef ENABLE_CUDA
using namespace dnn_opt::cuda;
#elif ENABLE_COPT
using namespace dnn_opt::copt;
#elif ENABLE_CORE
using namespace dnn_opt::core;
#endif

namespace
{
static auto* sigmoid = activations::sigmoid::make();
static auto* relu = activations::relu::make();
static auto* identity = activations::identity::make();
}

/**
 * @brief Finds an integer parameter value from command line input.
 *
 * @param param the name of the input param.
 * @param default_value a default value in case that the given parameter is not
 * found in the input.
 * @param argc the amount of input values from command line.
 * @param argv the parameters from command line.
 *
 * @return an integer containing the value of the parameter.
 */
int input(string param, int default_value, int argc, char** argv)
{
  int result = default_value;

  for(int i = 0; i < argc; i++)
  {
    if(std::string(argv[i]) == param)
    {
      result = atoi(argv[i + 1]);
    }
  }

  return result;
}

float input_f(string param, float default_value, int argc, char** argv)
{
  float result = default_value;

  for(int i = 0; i < argc; i++)
  {
    if(std::string(argv[i]) == param)
    {
      result = atof(argv[i + 1]);
    }
  }

  return result;
}

std::string input_s(std::string param, std::string default_value, int argc, char** argv)
{
std::string result = default_value;

  for(int i = 0; i < argc; i++)
  {
    if(std::string(argv[i]) == param)
    {
      result = argv[i + 1];
    }
  }

  return result;
}

/**
 * @brief Create a @ref dnn_opt::core::solution that can be optimized.
 *
 * Solutions that can be created with this function are listed below:
 *
 * 0 - @ref dnn_opt::core::solutions::bench::de_jung
 * 1 - @ref dnn_opt::core::solutions::bench::ackley
 * 2 - @ref dnn_opt::core::solutions::bench::giewangk
 * 3 - @ref dnn_opt::core::solutions::bench::rastrigin
 * 4 - @ref dnn_opt::core::solutions::bench::rosenbrock
 * 5 - @ref dnn_opt::core::solutions::bench::schwefel
 * 6 - @ref dnn_opt::core::solutions::bench::styblinski_tang
 * 7 - @ref dnn_opt::core::solutions::bench::step
 * 8 - @ref dnn_opt::core::solutions::bench::alpine
 * 9 - @ref dnn_opt::core::solutions::bench::brown_function
 * 10 - @ref dnn_opt::core::solutions::bench::chung_reynolds
 * 11 - @ref dnn_opt::core::solutions::bench::cosine_mixture
 * 12 - @ref dnn_opt::core::solutions::bench::csendes
 * 13 - @ref dnn_opt::core::solutions::bench::deb1
 * 14 - @ref dnn_opt::core::solutions::bench::deb3
 * 15 - @ref dnn_opt::core::solutions::bench::dixonp
 * 16 - @ref dnn_opt::core::solutions::bench::eggh
 * 17 - @ref dnn_opt::core::solutions::bench::expo
 * 18 - @ref dnn_opt::core::solutions::bench::giunta
 * 
 * @param type the type of the solution to be created.
 * @param n the amount of dimensions of the solution.
 * @param generator an pointer of dnn_opt::core::generator for the solution.
 *
 * @throw invalid_argument if the provided type can not be created.
 *
 * @return a pointer to the created solution.
 */
solution* create_solution(int type, int n, generator* generator)
{
  switch(type)
  {
  case 0 :
    return solutions::bench::de_jung::make(generator, n);
  case 1:
    return solutions::bench::ackley::make(generator, n);
  case 2:
    return solutions::bench::griewangk::make(generator, n);
  case 3:
    return solutions::bench::rastrigin::make(generator, n);
  case 4:
    return solutions::bench::rosenbrock::make(generator, n);
  case 5:
    return solutions::bench::schwefel::make(generator, n);
  case 6:
    return solutions::bench::styblinski_tang::make(generator, n);
  case 7:
    return solutions::bench::step::make(generator, n);
  case 8:
    return solutions::bench::alpine::make(generator, n);
  case 9:
    return solutions::bench::brown::make(generator, n);
  case 10:
    return solutions::bench::chung_r::make(generator, n);
  case 11:
    return solutions::bench::cosine_m::make(generator, n);
  case 12:
    return solutions::bench::csendes::make(generator, n);
  case 13:
    return solutions::bench::deb1::make(generator, n);
  case 14:
    return solutions::bench::deb3::make(generator, n);
  case 15:
    return solutions::bench::dixonp::make(generator, n);
  case 16:
    return solutions::bench::eggh::make(generator, n);
  case 17:
    return solutions::bench::expo::make(generator, n);
  case 18:
    return solutions::bench::giunta::make(generator, n);
  default:
    throw invalid_argument("solution type not found");
  }
}

solutions::net::seq* create_model(int id,
                                  const generator* gen,
                                  const reader* dataset,
                                  const error* loss)
{
  solutions::net::seq* nn = solutions::net::seq::make(gen, dataset, loss);

  switch(id)
  {
  case 0:
  {
    nn->add_layer(
    {
      layers::fc::make({dataset->get_in_dim()}, 10, sigmoid),
      layers::fc::make({10}, dataset->get_out_dim(), sigmoid)
    });
    break;
  }
  case 1:
  {
    auto* conv1 = layers::conv::make({1, dataset->get_in_dim(), 1}, {1, 6}, {1, 1}, 5, relu);
    auto* max1 = layers::max_pool::make({1, 94, 5}, {1, 3}, {1, 3});
    auto* conv2 = layers::conv::make({1, 31, 5}, {1, 4}, {1, 1}, 5, relu);
    auto* max2 = layers::max_pool::make({1, 28, 5}, {1, 2}, {1, 2});
    auto* lc1 = layers::fc::make({1 * 14 * 5}, 10, relu);
    auto* fc1 = layers::fc::make({10}, 10, sigmoid);
    auto* fc2 = layers::fc::make({10}, dataset->get_out_dim(), sigmoid);

    nn->add_layer({conv1, max1, conv2, max2, lc1,  fc1, fc2});
    break;
  }
  case 2:
  {
    auto* conv1 = layers::conv::make({1, dataset->get_in_dim(), 1}, {1, 6}, {1, 1}, 10, relu);
    auto* max1 = layers::max_pool::make({1, 94, 10}, {1, 2}, {1, 2});
    auto* conv2 = layers::conv::make({1, 47, 10}, {1, 4}, {1, 1}, 15, relu);
    auto* max2 = layers::max_pool::make({1, 44, 15}, {1, 2}, {1, 2});
    auto* conv3 = layers::conv::make({1, 22, 15}, {1, 2}, {1, 1}, 5, relu);
    auto* max3 = layers::max_pool::make({1, 21, 5}, {1, 2}, {1, 2});
    auto* fc1 = layers::fc::make({1 * 10 * 5}, 10, sigmoid);
    auto* fc2 = layers::fc::make({10}, 10, sigmoid);
    auto* fc3 = layers::fc::make({10}, dataset->get_out_dim(), sigmoid);

    nn->add_layer({conv1, max1, conv2, max2, conv3, max3, fc1,  fc2, fc3});
    break;
  }
  default:
    throw invalid_argument("model id not found");
  }

  return nn;
}

/**
 * @brief Create a @ref dnn_opt::core::algorithm.
 *
 * Algorithms that can be created with this function are listed below:
 *
 * 0 - @ref dnn_opt::core::algorithms::pso
 * 1 - @ref dnn_opt::core::algorithms::firefly
 *
 * @param type the algorithm to be created.
 * @param solutions an instance of dnn_opt::core::set to be optimized.
 *
 * @throw invalid_argument if the provided type can not be created.
 *
 * @return a pointer of the created algorithm.
 */
template<class t_solution>
algorithm* create_algorithm(int type, set<t_solution>* solutions)
{
  switch(type)
  {
  case 0 :
    return algorithms::pso::make(solutions);
  case 1 :
    return algorithms::firefly::make(solutions);
  case 2 :
    return algorithms::cuckoo::make(solutions);
  default:
    throw invalid_argument("algorithm type not found");
  }
}

/**
 * @brief Set the hyper-parameters to a given algorithm.
 *
 * Hyper-parameters were previously selected using automatic hyper-parameter
 * optimization. See corresponding example, @ref hpo.cpp.
 *
 * Algorithms that can be used in this function are listed below:
 *
 * 0 - dnn_opt::core::algorithms::pso
 *
 * @param type the type of algorithm to assign its hyper-parameters.
 * @param algorithm a pointer to the algorithm.
 *
 * @throw invalid_argument if the provided type can not be used.
 */
void set_hyper(int type, algorithm* algorithm, int argc, char** argv)
{
  std::vector<float> params;
  float ha, hb, hc, hd, he, hf;

  switch(type)
  {
  case 0 :
    ha = input_f("-ha", 0.8f, argc, argv);
    hb = input_f("-hb", 0.6f, argc, argv);
    hc = input_f("-hc", 2.5f, argc, argv);
    hd = input_f("-hd", 0.01f, argc, argv);

    params = {ha, hb, hc, hd};
    break;
  case 1 :
    ha = input_f("-ha", 0.8f, argc, argv);
    hb = input_f("-hb", 0.6f, argc, argv);
    hc = input_f("-hc", 0.3f, argc, argv);

    params = {ha, hb, hc};
    break;
    
    case 2 :
    ha = input_f("-ha", 0.8f, argc, argv);
    hb = input_f("-hb", 0.6f, argc, argv);
    hc = input_f("-hc", 0.3f, argc, argv);

    params = {ha, hb, hc};
    break;
  default:
    throw invalid_argument("algorithm type not found");
  }

  algorithm->set_params(params);
}

/**
 * @brief Produce a string in the standard output containing information about
 * the optimization process.
 *
 * Posible values for the output type are:
 *
 * 0 - None
 * 1 - Fitness
 * 2 - TimeFitness
 * 3 - HPOLib
 *
 * @param type type of output.
 * @param time amount of time of the optimization.
 * @param fitness fitness value of the best individual.
 */
void example_out(int type, float time, float fitness)
{
  switch(type)
  {
  case 1:
    cout << fitness << endl;
    break;
  case 2:
    cout << time << " " << fitness << endl;
    break;
  case 3:
    cout << "Result for ParamILS: SAT, " << time;
    cout << ", 1, " << fitness << ", -1, dnn_opt" << endl;
    break;
  }	
}
