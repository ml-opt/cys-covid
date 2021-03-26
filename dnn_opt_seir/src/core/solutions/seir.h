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
SERVICES; LOSS OF USE, infected, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef DNN_OPT_CORE_SOLUTIONS_SEIR
#define DNN_OPT_CORE_SOLUTIONS_SEIR

#include "cmath"
#include "algorithm"

#include <core/generators/uniform.h>
#include <core/generators/group.h>
#include <core/base/solution.h>
#include <core/errors/nmse.h>
#include <core/solutions/net/seq.h>
#include <core/layers/fc.h>
#include <core/activations/sigmoid.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{
namespace seir
{
namespace models
{

class base
{
public:

  base(int time, float s_0)
  : m_s_0(s_0),
    m_time(time),
    m_generator(0)
  {
    m_hs = new float[time];
    m_he = new float[time];
    m_hi = new float[time];
    m_hr = new float[time];

    m_generator = 0;
    m_gamma0_gen = 0;
    m_gamma1_gen = 0;
    m_beta_gen = 0;
  }

  virtual float get_beta(float time, const float* params)
  {
    return params[2];
  }

  virtual void run(const float* params)
  {
    const float& gamma_0 = params[0];
    const float& gamma_1 = params[1];

    float s = m_s_0;
    float e = 0.0f;
    float i = 3.0f;
    float r = 0.0f;
    float n = s + e + i + r;

    for(int time = 0; time < m_time; time++)
    {
      float beta = get_beta(time, params);
      float s2e = std::min(s, beta * i * s / n);

      float ds = -1.0f * s2e;
      float de = s2e - gamma_0 * e;
      float di = gamma_0 * e - gamma_1 * i;
      float dr = gamma_1 * i;

      s += ds;
      e += de;
      i += di;
      r += dr;

      m_hs[time] = s;
      m_he[time] = e;
      m_hi[time] = i;
      m_hr[time] = r;
    }
  }

  virtual int size() const
  {
    return 3;
  }

  virtual const generator* get_generator()
  {
    if(m_generator == 0)
    {
      m_gamma0_gen = generators::uniform::make(1.0f / 14.0f, 1.0f / 2.0f);
      m_gamma1_gen = generators::uniform::make(1.0f / 42.0f, 1.0f / 7.0f);
      m_beta_gen = generators::uniform::make(0.0f, 1.0f);

      m_generator = generators::group::make(
      {
        std::make_tuple(1, m_gamma0_gen),
        std::make_tuple(1, m_gamma1_gen),
        std::make_tuple(1, m_beta_gen),
      });
    }

    return m_generator;
  }

  const float* get_hs() const
  {
    return m_hs;
  }

  const float* get_he() const
  {
    return m_he;
  }

  const float* get_hi() const
  {
    return m_hi;
  }

  const float* get_hr() const
  {
    return m_hr;
  }

  int get_time() const
  {
    return m_time;
  }

  virtual ~base()
  {
    delete[] m_hs;
    delete[] m_he;
    delete[] m_hi;
    delete[] m_hr;

    delete m_generator;
    delete m_gamma0_gen;
    delete m_gamma1_gen;
    delete m_beta_gen;
  }

protected:

  int m_time;

  float m_s_0;
  float m_i_0;

  float* m_hs;
  float* m_he;
  float* m_hi;
  float* m_hr;

  generator* m_generator;
  generator* m_gamma0_gen;
  generator* m_gamma1_gen;
  generator* m_beta_gen;
};

class beta_cos : public base
{
public:

  beta_cos(int time, float s_0)
  : base(time, s_0)
  {
    m_beta_min_gen = 0;
    m_beta_lag_gen = 0;
  }

  virtual float get_beta(float time, const float* params) override
  {
    const float& gamma_1 = params[1];
    const float& beta_min = params[2];
    const float& beta_lag = params[3];

    float max = 6.49 * gamma_1;
    float min = beta_min * max;
    float beta = 0.5f * (max - min) * (std::cos(beta_lag * time) + 1) + min;

    return beta;
  }

  virtual int size() const
  {
    return 4;
  }

  virtual const generator* get_generator()
  {
    if(m_generator == 0)
    {
      m_gamma0_gen = generators::uniform::make(1.0f / 14.0f, 1.0f / 2.0f);
      m_gamma1_gen = generators::uniform::make(1.0f / 42.0f, 1.0f / 7.0f);
      m_beta_min_gen = generators::uniform::make(0.0f, 1.0f);
      m_beta_lag_gen = generators::uniform::make(0.0f, 1.0f);

      m_generator = generators::group::make(
      {
        std::make_tuple(1, m_gamma0_gen),
        std::make_tuple(1, m_gamma1_gen),
        std::make_tuple(1, m_beta_min_gen),
        std::make_tuple(1, m_beta_lag_gen),
      });
    }

    return m_generator;
  }

  virtual ~beta_cos()
  {
    delete m_beta_min_gen;
    delete m_beta_lag_gen;
  }

protected:

  generator* m_beta_min_gen;
  generator* m_beta_lag_gen;
};

class beta_exp : public beta_cos
{
public:

  beta_exp(int time, float s_0)
  : beta_cos(time, s_0)
  {

  }

  virtual float get_beta(float time, const float* params) override
  {
    const float& gamma_1 = params[1];
    const float& beta_min = params[2];
    const float& beta_lag = params[3];

    float max = 6.49 * gamma_1;
    float min = beta_min * max;
    float beta = (max - min) / std::pow(2.71828f, beta_lag * time) + min;

    return beta;
  }
};

class beta_exp_cos : public base
{
public:

  beta_exp_cos(int time, float s_0)
  : base(time, s_0)
  {
    m_beta_min_gen = 0;
    m_beta_lag0_gen = 0;
    m_beta_lag1_gen = 0;
  }

  virtual float get_beta(float time, const float* params) override
  {
    const float& gamma_1 = params[1];
    const float& beta_min = params[2];
    const float& beta_lag_0 = params[3];
    const float& beta_lag_1 = params[4];

    float max = 6.49 * gamma_1;
    float min = beta_min * max;
    float beta_cos = 0.5f * (max - min) * (std::cos(beta_lag_0 * time) + 1) + min;
    float beta_exp = (max - min) / std::pow(2.718281828459045f, beta_lag_1 * time) + min;
    float beta = beta_cos * beta_exp / max;

    return beta;
  }

  virtual int size() const override
  {
    return 5;
  }

  virtual const generator* get_generator() override
  {
    if(m_generator == 0)
    {
      m_gamma0_gen = generators::uniform::make(1.0f / 14.0f, 1.0f / 2.0f);
      m_gamma1_gen = generators::uniform::make(1.0f / 42.0f, 1.0f / 7.0f);
      m_beta_min_gen = generators::uniform::make(0.0f, 1.0f);
      m_beta_lag0_gen = generators::uniform::make(0.0f, 1.0f);
      m_beta_lag1_gen = generators::uniform::make(0.0f, 1.0f);

      m_generator = generators::group::make(
      {
        std::make_tuple(1, m_gamma0_gen),
        std::make_tuple(1, m_gamma1_gen),
        std::make_tuple(1, m_beta_min_gen),
        std::make_tuple(1, m_beta_lag0_gen),
        std::make_tuple(1, m_beta_lag1_gen)
      });
    }

    return m_generator;
  }

  virtual ~beta_exp_cos()
  {
    delete m_beta_lag0_gen;
    delete m_beta_lag1_gen;
  }

protected:

  generator* m_beta_min_gen;
  generator* m_beta_lag0_gen;
  generator* m_beta_lag1_gen;
};

class beta_net : public base
{
public:

  beta_net(int time, float s_0)
  : base(time, s_0)
  {
    m_net_gen = 0;
    m_sigmoid = activations::sigmoid::make();
    m_net = solutions::net::seq::make();
    m_net->add_layer(
    {
        layers::fc::make({1}, 5, m_sigmoid),
        layers::fc::make({5}, 1, m_sigmoid)
    });
  }

  virtual float get_beta(float time, const float* params) override
  {
    const float& gamma_1 = params[1];
    float max_beta = 6.49 * gamma_1;
    float beta = m_net->predict(&time, params + 2)[0];

    return beta * max_beta;
  }

  virtual const generator* get_generator()
  {
    if(m_generator == 0)
    {
      m_gamma0_gen = generators::uniform::make(1.0f / 14.0f, 1.0f / 2.0f);
      m_gamma1_gen = generators::uniform::make(1.0f / 42.0f, 1.0f / 7.0f);
      m_net_gen = generators::uniform::make(-1.0f, 1.0f);
      m_generator = generators::group::make(
      {
        std::make_tuple(1, m_gamma0_gen),
        std::make_tuple(1, m_gamma1_gen),
        std::make_tuple(m_net->size(), m_net_gen)
      });
    }

    return m_generator;
  }

  virtual int size() const override
  {
    return 2 + m_net->size();
  }

  virtual ~beta_net()
  {
    delete m_net;
    delete m_net_gen;
    delete m_sigmoid;
  }

protected:

  solutions::net::seq* m_net;
  generator* m_net_gen;
  activations::sigmoid* m_sigmoid;
};

} // namespace models

class seir : public virtual solution
{
public:

  static seir* make(int time, int max_cap, models::base* model, float* infected, float* recovered)
  {
    auto* result = new seir(time, max_cap, model, infected, recovered);

    result->init();

    return result;
  }

  virtual void init() override
  {
    solution::init();

    m_error = errors::nmse::make();
  }

  virtual ~seir()
  {
    delete m_error;
  }

protected:

  seir(int time, int max_cap, models::base* model, float* infected, float* recovered)
  : solution(model->get_generator(), model->size()),
    m_time(time),
    m_max_cap(max_cap),
    m_model(model),
    m_infected(infected),
    m_recovered(recovered)
  {

  }

  float calculate_fitness() override
  {
    float e_inf = 0.0f;
    float e_rec = 0.0f;
    float e_fut = 0.0f;
    float e = 0.0f;

    int time = m_model->get_time();

    m_model->run(get_params());

    const float* hi = m_model->get_hi();
    const float* hr = m_model->get_hr();

    e_inf = m_error->f(m_time, 1, hi, m_infected);
    e_rec = m_error->f(m_time, 1, hr, m_recovered);

//    for(int i = m_time; i < time; i++)
//    {
//      float current_e = m_max_cap - hi[i];

//      if(current_e > 0.0f)
//      {
//        e_fut += std::pow(current_e, 2.0f);
//      } else
//      {
//        e_fut += std::pow(2.0f * current_e, 2.0f);
//      }
//    }

//    e_fut /= (time - m_time);
//    e = 0.5 * (e_inf + e_fut) + e_rec;

    e = e_inf + e_rec;

    return e;
  }

protected:

  int m_time;
  int m_max_cap;

  const float* m_infected;
  const float* m_recovered;

  models::base* m_model;
  errors::nmse* m_error;

};

} // namespace seir
} // namespace solutions
} // namepsace core
} // namespace dnn_opt

#endif
