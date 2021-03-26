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

#ifndef DNN_OPT_CORE_LAYERS_FC
#define DNN_OPT_CORE_LAYERS_FC

#include "vector"
#include <core/base/layer.h>
#include <core/base/activation.h>

namespace dnn_opt
{
namespace core
{
namespace layers
{

/**
 * @brief The fc_layer class represents a layer of processing
 * units of an artificial neural network where each unit is fully connected
 * to the output of the previous layer.
 *
 * @author: Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date September, 2016
 * @version 1.0
 */
class fc : public virtual layer
{
public:

  /**
   * @brief Create a new fc class instance.
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
   * @param neurons number of units in this layer. This is also the size of the
   * layer's ouotput.
   *
   * @param activation the @ref activation that will be used to
   * produce this layer ouput. Default: activation::get_numpy().
   *
   * @return a new instance of this class.
   *
   * @throws @throws @ref std::out_of_range if the given input shape
   * is wrong.
   */
  static fc* make(std::vector<int> in_shape,
                  int neurons,
                  const activation* activation = activation::get_numpy());

  /**
   * @copydoc core::layer::init()
   */
  virtual void init() override;

  /**
   * @copydoc dnn_opt::core::layer::prop()
   *
   * The first @ref w_size() parameters are considered weights and the other
   * @ref b_size() parameters are considered biases.
   */
  virtual void prop(int size, const float* in, const float* params, float* out) const override;

  /**
   * @copydoc core::layer::w_size()
   *
   * number of parameters = @ref get_in_dim() * @ref get_out_dim()
   *
   * @return the number of parameters for weights required by this layer.
   */
  virtual int w_size() const override;

  /**
   * @copydoc core::layer::b_size()
   *
   * number of parameters = @ref get_out_size()
   *
   * @return the number of parameters for bias required by this layer.
   */
  virtual int b_size() const override;

  /**
   * @copydoc core::layer::clone()
   */
  virtual layer* clone() override;

protected:

  /**
   * Calculate the weighted sumatory of the inputs with the parameters for
   * each neuron in the layer and stores the results in the out array for each
   * training pattern.
   *
   * @param size the amount of training patterns to propagate.
   *
   * @param in a flatten array containing [size, get_in_dim()] elements
   * representing the in signal to be propagated in a row by row fashion.
   *
   * @param params an array containing at least size() elements
   * representing the parameters to be used by this layer.
   *
   * @param[out] out flatten array of containing @ref size x @ref out_get_dim()
   * elements representing the out signal in a column by column fashion.
   */
  virtual void ws(int size, const float* in, const float* params, float* out) const;

  /**
   * @brief Create a new fc class instance.
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
   * @param neurons number of units in this layer. This is also the size of the
   * layer's ouotput.
   *
   * @param activation the @ref activation that will be used to
   * produce this layer ouput. Default: activation::get_numpy().
   *
   * @return a new instance of this class.
   *
   * @throws @throws @ref std::out_of_range if the given input shape
   * is wrong.
   */
  fc(std::vector<int> in_shape,
     int neurons,
     const activation* activation = activation::get_numpy());

};

} // namespace layers
} // namespace core
} // namespace dnn_opt

#endif
