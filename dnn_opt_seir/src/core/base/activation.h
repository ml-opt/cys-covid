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

#ifndef DNN_OPT_CORE_ACTIVATION
#define DNN_OPT_CORE_ACTIVATION

#include "stdexcept"

namespace dnn_opt
{
namespace core
{

/**
 * @brief The activation class is intended to provide an
 * interface for custom activation functions that can be used by
 * an artificial neural network.
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date September, 2016
 * @version 1.0
 */
class activation
{
public:

  /**
   * @brief Returns an instance of a numpy activation function.
   *
   * The numpy activation function do not do anything. Is a lightweight class
   * that allows you to specify activation functions latter when you need to use
   * them.
   *
   * This numpy activation function will throw an @ref std::logic_error when
   * someone try to use it for actual activation function calculation.
   *
   * @return the instance of the numpy activation function.
   */
  static const activation* get_numpy();

  /**
   * @brief Given the summatory of an artificial neural network layer
   * calculates all the output activation values.
   *
   * @param size the amount of elements to propagate.
   *
   * @param dim the dimension of the layer output.
   *
   * @param sum an array of @size * @dim elements where each element represents
   * the weighted summatory of a neuron.
   *
   * @param[out] out an array where to store the activation values for each
   * unit of the layer.
   */
  virtual void f(int size, int dim, const float* sum, float* out) const = 0;

  /**
   * The basic destructor of this class.
   */
  virtual ~activation();

protected:

  class numpy;

  static numpy* m_numpy;

};

/**
 * @brief The activation::numpy class is an activation function that is
 * intended to be used when you expect to specify a layer's activation
 * function latter.
 *
 * This numpy activation function will throw an @ref std::logic_error when
 * someone try to use it for actual activation function calculation.
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date March, 2020
 * @version 1.0
 */
class activation::numpy : public virtual activation
{
public:

  inline static activation::numpy* make()
  {
    return new activation::numpy();
  }

  /**
   * @copydoc activation::f()
   *
   * @throws std::logic_error always.
   */
  inline virtual void f(int size, int dim, const float* sum, float* out) const override
  {
    throw std::logic_error("You are currently using a numpy activation");
  }

protected:

  inline numpy() : activation()
  {

  }
};

} // namespace core
} // namespace dnn_opt

#endif
