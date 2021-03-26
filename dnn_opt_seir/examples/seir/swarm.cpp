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

#include <iostream>
#include <fstream>
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
  int p = input("-p", 1000, argc, argv);
  int eta = input("-eta", 40000, argc, argv);
  int algorithm_type = input("-a", 1, argc, argv);
  int model_type = input("-model", 0, argc, argv);

  std::string dbi = input_s("-db-i", "", argc, argv);
  std::string dbr = input_s("-db-r", "", argc, argv);
  std::string odb = input_s("-odb", "", argc, argv);

  int t = input("-time", 100, argc, argv);
  int s_0 = input("-s0", 100, argc, argv);

  std::ifstream finfected(dbi);
  std::ifstream frecovered(dbr);
  std::ofstream ofile(odb);

  float infected[t];
  float recovered[t];

  for(int i = 0; i < t; i++)
  {
    finfected >> infected[i];
  }

  for(int i = 0; i < t; i++)
  {
    frecovered >> recovered[i];
  }

  finfected.close();
  frecovered.close();

  auto* solutions = set<>::make(p);
  solutions::seir::models::base* model;

  if(model_type == 0)
    model = new solutions::seir::models::beta_cos(50, s_0);
  else if(model_type == 1)
    model = new solutions::seir::models::beta_exp(50, s_0);
  else if(model_type == 2)
    model = new solutions::seir::models::beta_exp_cos(50, s_0);
  else if(model_type == 3)
    model = new solutions::seir::models::beta_net(50, s_0);

  for (int i = 0; i < p; ++i)
  {
    solutions->add(solutions::seir::seir::make(50, 2000, model, infected, recovered));
  }
  solutions->generate();

  auto* algorithm = create_algorithm(algorithm_type, solutions);

  set_hyper(algorithm_type, algorithm, argc, argv);
  algorithm->optimize(eta, []()
  {
    return true;
  });

  /* save swarm to file */

  for(int i = 0; i < p; i++)
  {
    for(int j = 0; j < solutions->get_dim() - 1; j++)
    {
      ofile << solutions->get(i)->get(j) << ",";
    }

    ofile << solutions->get(i)->get(solutions->get_dim() - 1) << endl;
  }

  ofile.close();

  delete solutions->clean();
  delete algorithm;

  return 0;
}
