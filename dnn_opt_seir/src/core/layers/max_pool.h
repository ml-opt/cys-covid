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

#ifndef DNN_OPT_CORE_LAYERS_MAX_POOL
#define DNN_OPT_CORE_LAYERS_MAX_POOL

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
 * The max_pool class implements a max pooling layer.
 *
 * @author: Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date March, 2020
 * @version 1.0
 */
class max_pool : public virtual layer
{
public:

  /**
   * @brief Create a new max pooling layer class instance.
   *
   * The input shape is an integer list of one, two or three values.
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
   * @param w_shape the shape of the moving window.
   *
   * The window shape is an integer list of one or two values.
   *
   * If one value is provided, it is assumed that the shape represents a window
   * with the same height as the input and given width.
   *
   * If two values are provided, it is assumed that the shape represents a
   * window with given height and width in that order.
   *
   * @param s_shape the shape of the stride
   *
   * The stride shape is an integer list of one or two values.
   *
   * If one value is provided, it is assumed that the shape represents a stride
   * on the height dimension of 1 and given width dimension.
   *
   * If two values are provided, it is assumed that the shape represents a
   * stride on the given height and width dimensions in that order.
   *
   * @return a new instance of the max_pool class.
   *
   * @throws @ref std::out_of_range if the given input, window or stride shape
   * are wrong.
   */
  static max_pool* make(std::vector<int> in_shape,
                        std::vector<int> w_shape,
                        std::vector<int> s_shape);

  /**
   * @copydoc core::layer::init()
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
   * @copydoc core::layer::b_size()
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

protected:

  /**
   * @brief Create a new max pooling layer class instance.
   *
   * The input shape is an integer list of one, two or three values.
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
   * @param w_shape the shape of the moving window.
   *
   * The window shape is an integer list of one or two values.
   *
   * If one value is provided, it is assumed that the shape represents a window
   * with the same height as the input and given width.
   *
   * If two values are provided, it is assumed that the shape represents a
   * window with given height and width in that order.
   *
   * @param s_shape the shape of the stride
   *
   * The stride shape is an integer list of one or two values.
   *
   * If one value is provided, it is assumed that the shape represents a stride
   * on the height dimension of 1 and given width dimension.
   *
   * If two values are provided, it is assumed that the shape represents a
   * stride on the given height and width dimensions in that order.
   *
   * @return a new instance of the max_pool class.
   *
   * @throws @ref std::out_of_range if the given input, window or stride shape
   * are wrong.
   */
  max_pool(std::vector<int> in_shape,
           std::vector<int> w_shape,
           std::vector<int> s_shape);

  /** the shape of the window, two values: height and width */
  std::vector<int> m_w_shape;

  /** the shape of the stride, two values: height stride and width stride */
  std::vector<int> m_s_shape;
};

} // namespace layers
} // namespace core
} // namespace dnn_opt

#endif
