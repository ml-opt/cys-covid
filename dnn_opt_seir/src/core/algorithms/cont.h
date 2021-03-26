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

#ifndef DNN_OPT_CORE_ALGORITHMS_CONT
#define DNN_OPT_CORE_ALGORITHMS_CONT

#include "vector"
#include "functional"
#include <core/base/algorithm.h>
#include <core/base/proxy_sampler.h>
#include <core/base/reader.h>
#include <core/base/set.h>
#include <core/solutions/net/seq.h>
#include <core/generators/uniform.h>

namespace dnn_opt
{
namespace core
{
namespace algorithms
{

/**
 * @brief The continuation class implements a continuation method for
 * artificial neural network training.
 *
 * This continuation version reduces execution time of training while no
 * statistical significative reduction of accuracy have been measured.
 *
 * See: Rojas-Delgado J., Trujillo-Rasúa R. y Bello R. A continuation approach
 * for training artificial neural networks with meta-heuristics. Pattern
 * Recognition Letters, 2019, vol. 125, 373 - 380. Elseiver. Available in:
 * http://www.sciencedirect.com/science/article/pii/S0167865519301667
 * DOI: https://doi.org/10.1016/j.patrec.2019.05.017 ISSN: 0167-8655
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date November, 2018
 */
class cont : public virtual algorithm
{
public:

  static cont* make(algorithm* base,
                    const set<solutions::net::seq>* solutions,
                    const reader* dataset);

  virtual void reset() override;

  virtual void optimize() override;

  virtual void optimize(int eta, std::function<bool()> on) override;

  virtual void optimize_idev(int count, float dev, std::function<bool()> on) override;

  virtual void optimize_dev(float dev, std::function<bool()> on) override;

  virtual void optimize_eval(int count, std::function<bool()> on) override;

  virtual solution* get_best() override;

  virtual void init() override;

  virtual void set_params(std::vector<float> &params) override;

  using algorithm::set_params;

  int get_k() const;

  void set_k(int k);

  float get_beta() const;

  void set_beta(float beta);

  const reader* get_dataset() const;

  virtual ~cont() override;

protected:

  virtual void build_seq(int k, float beta, const reader* reader);

  cont(algorithm* base,
       const set<solutions::net::seq>* solutions,
       const reader* dataset);

  int m_k;

  float m_beta;

  /** The base algorithm that performs optimization */
  algorithm* m_base;

  /** A pointer of @ref get_solutions() that do not degrade to core::solution */
  const set<solutions::net::seq>* m_network_solutions;

  /** The dataset reader extracted from the first network solution */
  const reader* m_dataset;

  std::vector<reader*> m_sequence;

  bool m_built;

private:

  void set_reader(int index);
};

} // namespace algorithms
} // namespace core
} // namespace dnn_opt

#endif
