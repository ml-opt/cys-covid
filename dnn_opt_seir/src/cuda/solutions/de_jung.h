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

#ifndef DNN_OPT_CUDA_SOLUTIONS_DE_JUNG
#define DNN_OPT_CUDA_SOLUTIONS_DE_JUNG

#include <cuda/base/solution.h>
#include <cuda/base/generator.h>
#include <core/solutions/de_jung.h>

namespace dnn_opt
{
namespace cuda
{
namespace solutions
{

/**
 * @brief The de_jung class represents an optimization solutions which
 * fitness cost is calculated via De'Jung function.
 *
 * De'Jung function have a global minima in {0,..., 0} with a value of 0.
 * A commonly used search domain for testing is [-5.12, 5.12].
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @version 1.0
 * @date November, 2016
 */
class de_jung : public cuda::solution,
                public core::solutions::de_jung
{
public:

  /**
   * @brief Returns an instance of this object. This method
   * is an implementation of the factory pattern.
   *
   * @param generator an instance of a parameter_generator class. The
   * parameter_generator is used to initialize the parameters of this solution.
   *
   * @param size is the number of parameters for this solution. Default is 10.
   *
   * @return an instance of de_jung class.
   */
  static de_jung* make(generator *generator, int size = 10);

  virtual de_jung* clone() override;

  bool assignable(const core::solution* s) const override;

  /**
   * @brief Transfer the data of this @ref cuda::solutions::de_jung into a @ref
   * core::solutions::de_jung for sequential processing.
   *
   * @param solution the solution in the @ref core namespace where to tranfer
   * data.
   */
  void to_core(core::solutions::de_jung* solution) const;

  /**
   * @brief Transfer the data from a @ref core::solutions::de_jung into this
   * @ref cuda::solutions::de_jung for parallel processing in a GPU.
   *
   * @param solution the solution in in the @core namespace from where pull the
   * data.
   */
  void from_core(core::solutions::de_jung* solution);

  virtual ~de_jung() override;

protected:

  float calculate_fitness() override;

  /**
   * @brief This is the basic contructor for this class. Is protected since
   * this this class implements the factory pattern. Derived clasess however
   * can use this constructor.
   *
   * @param generator an instance of a parameter_generator class.
   * The parameter_generator is used to initialize the parameters of this
   * solution.
   *
   * @param size is the number of parameters for this solution. Default is 10.
   */
  de_jung(generator* generator, int size = 10);

};

} // namespace solutions
} // namespace cuda
} // namespace dnn_opt

#endif
