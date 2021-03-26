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

#ifndef DNN_OPT_CORE_SOLUTIONS_NET_SEQ
#define DNN_OPT_CORE_SOLUTIONS_NET_SEQ

#include "vector"
#include "initializer_list"
#include <core/base/error.h>
#include <core/base/reader.h>
#include <core/base/generator.h>
#include <core/base/solution.h>
#include <core/base/layer.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{
namespace net
{

class seq : public virtual solution
{
public:

  static seq* make(const generator* generator = generator::get_numpy(),
                   const reader* reader = reader::get_numpy(),
                   const error* error = error::get_numpy());

  virtual seq* clone() override;

  virtual void assign(solution* s) override;

  virtual bool assignable(const solution* s) const override;

  void add_layer(std::initializer_list<layer*> layers);

  /**
   * Add a new layer at the end of all layers. Make sure to call init()
   * when you finish to alter the network layered structure.
   *
   * @param layer the new layer to be added.
   *
   * @return a pointer to this solution.
   */
  seq* add_layer(layer* layer);

  std::vector<layer*> get_layers() const;

  /**
   * @brief The reader used to obtain the training patterns.
   *
   * @return a constant pointer to the reader.
   */
  virtual const reader* get_reader() const;

  /**
   * @brief Change the reader used to obtain the training patterns in order to
   * calculate this network fitness.
   *
   * This method is expensive because it alters the network internal structure
   * that involves memory allocations and de-allocations. Use with caution.
   *
   * @param reader a reader containing the training patterns.
   */
  virtual void set_reader(const reader* reader);

  /**
   * @brief Propagate a validation set of patterns through the network and
   * calculate generalization error.
   *
   * @param reader a reader containing the validation set of patterns
   *
   * @return the generalization error based on the error of this
   * network.
   */
  virtual float test(const reader* validation_set);

  virtual float* predict(const reader* validation_set);

  virtual const float* predict(const float* example);

  virtual const float* predict(const float* example, const float* params);

  virtual void init() override;

  /**
   * @brief The error function used to calculate the fitness of the network.
   *
   * @return a constant pointer to the error function.
   */
  virtual const error* get_error() const;

  virtual void set_error(const error* error);

  /**
   * @brief The basic destructor of the network class.
   */
  virtual ~seq();

protected:

  /**
   * @brief Propagate reader examples through the layers of this network and
   * calculate the network loss based in the given @ref get_error() function.
   *
   * @return the loss value of the network.
   */
  virtual float calculate_fitness() override;

  /**
   * @brief Propagate the training patterns in through the network
   * and returns the network output for each training pattern.
   *
   * @return a flatten array of dimension [get_reader()->batch_size() x
   * get_reader()->out_get_dim()] in a row by row fashion.
   */
  const float* prop(float* params);

  seq(const generator* generator = generator::get_numpy(),
      const reader* reader = reader::get_numpy(),
      const error* error = error::get_numpy());

  /** List of layers in cascade to propagate the input signal */
  std::vector<layer*> m_layers;

  /** Reader that provides the list of training patterns */
  const reader* m_reader;

  /** Error function used to calculate the fitness of the network */
  const error* m_error;

protected:

  /** 
   * Output of the layer i that is currently propagating the input signal.
   * This is a flatten array of dimension [_r->batch_size() x _max_out] in 
   * a row by row fashion.
   */
  static float* m_current_out;
  
  /** 
   * The output of the layer i - 1.
   * This is a flatten array of dimension [_r->batch_size() x _max_out] in 
   * a row by row fashion.
   */
  static float* m_prior_out;

  /** The amount of outputs of the layer with the higher amount of units */
  static int m_max_out;

  static int m_max_reader;

};


} // namespace net
} // namespace solutions
} // namespace core
} // namespace dnn_opt

#endif
