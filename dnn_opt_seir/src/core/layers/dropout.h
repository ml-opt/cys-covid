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

#ifndef DNN_OPT_CORE_LAYERS_DROPOUT
#define DNN_OPT_CORE_LAYERS_DROPOUT

#include "vector"
#include <core/base/layer.h>
#include <core/base/activation.h>
#include <core/generators/uniform.h>

namespace dnn_opt
{
namespace core
{
namespace layers
{

/**
 * The dropout class implements a dropout layer.
 *
 * The dropout layer set to cero a given proportion of the output values.
 *
 * @author: Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date March, 2020
 * @version 1.0
 */
class dropout : public virtual layer
{
public:

  /**
   * @brief Creates a new instance of the dropout class.
   *
   * @param in_shape the shape of the input of this layer.
   *
   * If one value is provided, it is assumed that the shape represents a tensor
   * with no height nor depth.
   *
   * If two values are provided, it is assumed that the shape represents a
   * tensor with no depth and given height and width in that order.
   *
   * If three values are provided, it is assumed that the shape represents a
   * tensor with given height, width and depth in that order.
   *
   * Other shapes will result in an @ref std::out_of_range exception.
   *
   * @param p the proportion of neurons that are ignored during the forward
   * pass.
   *
   * @return a new instance of the dropout class.
   *
   * @throws @ref std::out_of_range if the given input shape is wrong.
   */
  static dropout* make(std::vector<int> in_shape, float p);

  /**
   * @copydoc core::layer::init()
   *
   * Create an internal generator and a float array of  @ref get_out_dim()
   * size to store randomly identify which neurons will be ignored.
   */
  virtual void init() override;

  /**
   * @copydoc core::layer::prop()
   */
  virtual void prop(int size, const float* in, const float* params, float* out) const override;

  /**
   * @copydoc core::layer::w_size()
   *
   * This layer do not require any weight.
   *
   * @return cero.
   */
  virtual int w_size() const override;

  /**
   * @copydoc core::layer::w_size()
   *
   * This layer do not require any bias.
   *
   * @return cero.
   */
  virtual int b_size() const override;

  /**
   * @copydoc core::layer::clone()
   */
  virtual layer* clone() override;

  /**
   * @brief The destructor of this class.
   *
   * Deleted the internal random number generator and the random number array.
   */
  virtual ~dropout() override;

protected:

  /**
   * @brief Creates a new instance of the dropout class.
   *
   * @param in_shape the shape of the input of this layer.
   *
   * If one value is provided, it is assumed that the shape represents a tensor
   * with no height nor depth.
   *
   * If two values are provided, it is assumed that the shape represents a
   * tensor with no depth and given height and width in that order.
   *
   * If three values are provided, it is assumed that the shape represents a
   * tensor with given height, width and depth in that order.
   *
   * Other shapes will result in an @ref std::out_of_range exception.
   *
   * @param p the proportion of neurons that are ignored during the forward
   * pass.
   *
   * @return a new instance of the dropout class.
   *
   * @throws @ref std::out_of_range if the given input shape is wrong.
   */
  dropout(std::vector<int> in_shape, float p);

  /** the proportion of neurons to be ignored */
  float m_p;

  /** an array of @ref get_out_dim() floats to store random numbers in [0, 1] */
  float* m_r;

  /** a generator of random numbers to populate @ref m_r */
  generators::uniform* m_generator;
};

} // namespace layers
} // namespace core
} // namespace dnn_opt

#endif
