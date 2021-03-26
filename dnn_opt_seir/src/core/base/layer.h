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

#ifndef DNN_OPT_CORE_LAYER
#define DNN_OPT_CORE_LAYER

#include "initializer_list"
#include "vector"
#include <core/base/activation.h>

namespace dnn_opt
{
namespace core
{

/**
 * @brief The layer class is intended as an interface for custom
 * layers that can be used by an artificial neural network.
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @version 1.0
 * @date June, 2016
 */
class layer
{
public:

  /**
   * @brief Allocate any dynamic memory required by the layer.
   */
  virtual void init() = 0;

  /**
   * @brief Propagate a training signal through the layer.
   *
   * @param size the amount of training patterns to propagate.
   *
   * @param in a flatten array containing [size, get_in_dim()] elements
   * representing the input signal to be propagated in a row by row fashion.
   *
   * @param params an array containing at least size() elements
   * representing the parameters to be used by this layer.
   *
   * @param[out] out a flatten array containing [size, get_out_dim()]
   * elements representing the output signal in a row by row fashion.
   */
  virtual void prop(int size, const float* in, const float* params, float* out) const = 0;

  /**
   * @brief The number of values this layer accepts as input.
   *
   * The input dimension is calculated as the multiplication of input height,
   * width and depth.
   *
   * @return the number of input dimensions.
   */
  int get_in_dim() const;

  /**
   * @brief The number of values this layer produces as output.
   *
   * The output dimension is calculated as the multiplication of output height,
   * width and depth.
   *
   * @return the number of output dimensions.
   */
  int get_out_dim() const;

  /**
   * @brief The input shape this layer accepts as input.
   *
   * @return std::vector<int> with three values: height, width and depth of
   * the expected input.
   */
  std::vector<int> get_in_shape() const;

  /**
   * @brief The output shape this layer produces as output.
   *
   * @return std::vector<int> with three values: height, width and depth of
   * the produced output.
   */
  std::vector<int> get_out_shape() const;

  /**
   * @brief The number of parameters that are required by this layer.
   *
   * @return the number of parameters.
   */
  virtual int size() const;

  /**
   * @brief The number of parameters that are considered weights.
   *
   * @return the number of weights.
   */
  virtual int w_size() const = 0;

  /**
   * @brief The number of parameters that are considered bias.
   *
   * @return the number of biases.
   */
  virtual int b_size() const = 0;

  /**
   * @brief The @ref activation this layer uses to produce its output
   * values.
   *
   * @return a pointer to the @ref activation.
   */
  virtual const activation* get_activation() const;

  /**
   * @brief Creates an exact copy of this layer.
   *
   * @return a pointer to a copy of this layer.
   */
  virtual layer* clone() = 0;

  virtual ~layer();

protected:

  /**
   * @brief The basic contructor for this class.
   *
   * @param in_shape the shape of the input of this layer.
   *
   * The input shape is an integer list of one, two or three values.
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
   * Other shapes will result in an \ref std::out_of_range exception.
   *
   * @param out_shape the shape of the output of this layer.
   *
   * The output shape is an integer list of one, two or three values.
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
   * Other shapes will result in an \ref std::out_of_range exception.
   *
   * @param activation the @ref activation that will be used to
   * produce this layer ouput. Default: activation::get_numpy().
   *
   * @throws @ref std::out_of_range if the given input or output shape are
   * wrong.
   */
  layer(std::vector<int> in_shape,
        std::vector<int> out_shape,
        const activation* activation = activation::get_numpy());

  /** The shape of the input, three values: height, width and depth */
  std::vector<int> m_in_shape;

  /** The shape of the output, three values: height, width and depth */
  std::vector<int> m_out_shape;

  /** The activation used by this layer to produce its output */
  const activation* m_activation;

};

} // namespace core
} // namespace dnn_opt

#endif

