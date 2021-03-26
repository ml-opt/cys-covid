#include "algorithm"
#include <core/solutions/net/seq.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{
namespace net
{

float* seq::m_current_out = 0;
float* seq::m_prior_out = 0;
int seq::m_max_out = 0;
int seq::m_max_reader = 0;

seq* seq::make(const generator* generator,
               const reader* reader,
               const error* error)
{
  auto* result = new seq(generator, reader, error);

  result->init();

  return result;
}

seq* seq::clone()
{
  seq* nn = make(get_generator(), get_reader(), get_error());

  for(auto &layer : m_layers)
  {
    nn->add_layer(layer->clone());
  }

  nn->init();
  nn->_fitness = fitness();
  nn->_evaluations = get_evaluations();
  nn->set_modified(false);

  std::copy_n(get_params(), size(), nn->get_params());

  return nn;
}

bool seq::assignable(const solution* s) const
{
  /*
   * TODO: Incomplete method implementation.
   * Check also that contains the same layered structure.
   */

  return size() == s->size();
}

void seq::assign(solution* s)
{
  seq* net = dynamic_cast<seq*>(s);

  solution::assign(s);

  set_reader(net->get_reader());
  set_error(net->get_error());
}

void seq::add_layer(std::initializer_list<layer*> layers)
{
  for(layer* layer : layers)
  {
    add_layer(layer);
  }

  init();
}

seq* seq::add_layer(layer* layer)
{
  /* check input and output dimension */

  if (!m_layers.empty())
  {
    int expected = m_layers.back()->get_out_dim();
    int actual = layer->get_in_dim();

    if(expected != actual)
    {
      throw std::logic_error("Layer missmatch exception: " +
                             std::to_string(expected) + "/" +
                             std::to_string(actual));
    }
  }

  _size += layer->size();
  m_max_out = std::max(m_max_out, layer->get_out_dim());
  m_layers.push_back(layer);

  return this;
}

std::vector<layer*> seq::get_layers() const
{
  return m_layers;
}

void seq::set_reader(const reader* reader)
{
  if(m_max_reader < reader->size())
  {
    m_max_reader = reader->size();

    delete[] m_current_out;
    delete[] m_prior_out;

    m_current_out = new float[m_max_reader * m_max_out];
    m_prior_out = new float[m_max_reader * m_max_out];
  }

  m_reader = reader;

  for(auto const & t : _clones)
  {
    seq* net = dynamic_cast<seq*>(t.second);

    net->set_reader(reader);
  }

  set_modified(true);
}

float seq::test(const reader* validation_set)
{
  const reader* current_reader = get_reader();
  float result = 0;

  set_reader(validation_set);
  result = calculate_fitness();
  set_reader(current_reader);

  return result;
}

float* seq::predict(const reader* validation_set)
{
  const reader* current_reader = get_reader();
  int n = validation_set->size() * validation_set->get_out_dim();
  float* result = new float[n];

  set_reader(validation_set);
  std::copy_n(prop(get_params()), n, result);
  set_reader(current_reader);

  return result;
}

const float* seq::predict(const float* example)
{
  predict(example, get_params());
}

const float* seq::predict(const float* example, const float* params)
{
  m_layers.front()->prop(1, example, params, m_prior_out);

  for(size_t i = 1; i < m_layers.size(); i++)
  {
    params += m_layers.at(i - 1)->size();
    m_layers.at(i)->prop(1, m_prior_out, params, m_current_out);
    std::swap(m_current_out, m_prior_out);
  }

  return m_prior_out;
}

void seq::init()
{
  delete[] m_current_out;
  delete[] m_prior_out;

  m_current_out = new float[m_max_reader * m_max_out];
  m_prior_out = new float[m_max_reader * m_max_out];

  solution::init();
}

const reader* seq::get_reader() const
{
  return m_reader;
}

const error* seq::get_error() const
{
  return m_error;
}

void seq::set_error(const error* error)
{
  m_error = error;

  for(auto const & t : _clones)
  {
    seq* net = dynamic_cast<seq*>(t.second);

    net->set_error(error);
  }
}

float seq::calculate_fitness()
{
  float* params = get_params();
  const error* e = get_error();
  const reader* r = get_reader();

  solution::calculate_fitness();

  return e->f(r->size(), r->get_out_dim(), prop(params), r->out_data());
}

const float* seq::prop(float* params)
{
  const reader* r = get_reader();

  /* propagate the signal in the first layer with input patterns */
  m_layers.front()->prop(r->size(), r->in_data(), params, m_prior_out);

  for(size_t i = 1; i < m_layers.size(); i++)
  {
    /* move window to the parameters of this layer */
    params += m_layers.at(i - 1)->size();

    /* propagate the signal through the layer */
    m_layers.at(i)->prop(r->size(), m_prior_out, params, m_current_out);

    /* swap the outs to use m_current_out as m_prior_out in next iteration */
    std::swap(m_current_out, m_prior_out);
  }

  return m_prior_out;
}

seq::seq(const generator* generator,
         const reader* reader,
         const error* error)
: solution(generator, 0),
  m_reader(reader),
  m_error(error)
{
  m_max_reader = std::max(m_max_reader, reader->size());
}

seq::~seq()
{
  for(auto* layer : m_layers)
  {
    delete layer;
  }

  m_layers.clear();
}

} // namespace net
} // namespace solutions
} // namespace core
} // namespace dnn_opt
