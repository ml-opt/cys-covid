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
  int t = input("-time", 100, argc, argv);
  float s_0 = input_f("-s0", 100, argc, argv);
  int model_type = input("-model", 0, argc, argv);

  solutions::seir::models::base* model;

  if(model_type == 0)
    model = new solutions::seir::models::beta_cos(500, s_0);
  else if(model_type == 1)
    model = new solutions::seir::models::beta_exp(500, s_0);
  else if(model_type == 2)
    model = new solutions::seir::models::beta_exp_cos(500, s_0);
  else if(model_type == 3)
    model = new solutions::seir::models::beta_net(500, s_0);

  std::string odb = input_s("-odb", "", argc, argv);
  std::string dbp = input_s("-db-p", "", argc, argv);

  std::ofstream ofile(odb);
  std::ifstream fparams(dbp);
  float params[model->size()];

  for(int i = 0; i < model->size(); i++)
  {
    fparams >> params[i];
  }

  fparams.close();
  model->run(params);

  const float* hs = model->get_hs();
  const float* he = model->get_he();
  const float* hi = model->get_hi();
  const float* hr = model->get_hr();

  for(int i = 0; i < t; i++)
  {
    ofile << hs[i] << ",";
    ofile << he[i] << ",";
    ofile << hi[i] << ",";
    ofile << hr[i] << ",";
    ofile << model->get_beta(i, params) << endl;
  }

  ofile.close();

  delete model;

  return 0;
}
