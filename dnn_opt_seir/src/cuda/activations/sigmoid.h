/*
Copyright (c) 2017, Jairo Rojas-Delgado <jrdelgado@uci.cu>
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

#ifndef DNN_OPT_CUDA_ACTIVATIONS_SIGMOID
#define DNN_OPT_CUDA_ACTIVATIONS_SIGMOID

#include <cuda/base/activation.h>
#include <core/activations/sigmoid.h>

namespace dnn_opt
{
namespace cuda
{
namespace activations
{

/**
 * @brief The sigmoid class represents a sigmoid function that can be 
 * used by an artificial neural network as activation function.
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date September, 2016
 * @version 1.0
 */
class sigmoid : public activation,
                public core::activations::sigmoid
{
public:

  /**
   * Create an instance of the sigmoid class.
   *
   * @return a pointer to a new instance of the sigmoid class.
   */
  static sigmoid* make();

  void f(int size, const float* sum, float* out) override;

};
}
}
}

#endif
