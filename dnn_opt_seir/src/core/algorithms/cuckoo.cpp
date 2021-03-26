#include "vector"
#include "stdexcept"
#include <core/algorithms/cuckoo.h>

namespace dnn_opt
{
namespace core
{
namespace algorithms
{

void cuckoo::init()
{
  std::vector<float> params = {0.8f, 0.8f, 0.3};

  /* set default hyper-params */

  set_params(params);

  /* delete already allocated memory */

  delete _nd_1;
  delete _nd_o;
  delete _selector;
  delete _updated;
  delete [] _r;

  /* mantegna algorithm to calculate levy steep size */

  float dividend = tgamma(1 + _levy) * sin(3.14159265f * _levy / 2);
  float divisor = tgamma((1 + _levy) / 2) * _levy * pow(2, (_levy - 1) / 2);
  float omega = pow(dividend / divisor , 1 / _levy);

  _nd_1 = generators::normal::make(0, 1);
  _nd_o = generators::normal::make(0, omega);
  _selector = generators::uniform::make(0, get_solutions()->size());
  _r = new float[get_solutions()->get_dim()];
  _updated = get_solutions()->get(0)->clone();
}

void cuckoo::reset()
{
  algorithm::reset();
}

void cuckoo::optimize()
{
  int n = get_solutions()->size();

  for(int i = 0; i < n; i++)
  {
    auto source = get_solutions()->get(i);

    generate_new_cuckoo(i);

    if(_updated->is_better_than(source, is_maximization()))
    {
      int evaluations = source->get_evaluations();

      source->assign(_updated);

      source->set_evaluations(evaluations + _updated->get_evaluations());
      _updated->set_evaluations(0);
    }
  }

  get_solutions()->sort(!is_maximization());

  for(int i = 0; i < _replacement * get_solutions()->size(); i++)
  {
    get_solutions()->get(i)->generate();
  }
}

void cuckoo::generate_new_cuckoo(int cuckoo_idx)
{
  float* cuckoo = get_solutions()->get(cuckoo_idx)->get_params();
  float* best = get_solutions()->get_best(is_maximization())->get_params();
  float* params = _updated->get_params();

  float v = _nd_1->generate();
  float u = _nd_o->generate();
  float levy = u / powf(fabs(v), 1 / _levy);

  _nd_1->generate(get_solutions()->get_dim(), _r);

  for(int i = 0; i < get_solutions()->get_dim(); i++)
  {
    params[i] += _scale * levy * (best[i] - cuckoo[i]) * _r[i];
  }

  _updated->set_modified(true);
}

solution* cuckoo::get_best()
{
  return get_solutions()->get_best(is_maximization());
}

void cuckoo::set_params(std::vector<float> &params)
{
  algorithm::set_params(params);

  if(params.size() != 3)
  {
    std::invalid_argument("algorithms::cuckoo set_params expect 3 values");
  }

  set_scale(params.at(0));
  set_levy(params.at(1));
  set_replacement(params.at(2));
}

float cuckoo::get_scale()
{
  return _scale;
}

float cuckoo::get_levy()
{
  return _levy;
}

float cuckoo::get_replacement()
{
  return _replacement;
}

void cuckoo::set_scale(float scale)
{
  _scale = scale;
}

void cuckoo::set_levy(float levy)
{
  _levy = levy;
}

void cuckoo::set_replacement(float replacement)
{
  _replacement = replacement;
}

cuckoo::~cuckoo()
{
  delete _updated;
  delete _nd_o;
  delete _nd_1;
  delete _selector;

  delete[] _r;
}

} // namespace algorithms
} // namespace core
} // namespace dnn_opt
