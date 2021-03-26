#include "algorithm"
#include "stdexcept"
#include <core/algorithms/cont.h>

namespace dnn_opt
{
namespace core
{
namespace algorithms
{

cont* cont::make(algorithm* base,
                 const set<solutions::net::seq>* solutions,
                 const reader* dataset)
{
  auto* result = new cont(base, solutions, dataset);

  result->init();

  return result;
}

void cont::init()
{
  std::vector<float> params = {2.0f, 0.8f};

  set_params(params);
}

void cont::reset()
{
  m_base->reset();
  m_built = false;
}

void cont::optimize()
{
  throw std::logic_error("cont single step optimization is useless");
}

void cont::optimize(int eta, std::function<bool()> on)
{
  bool on_opt = true;
  int k = get_k();
  int span = eta / k;

  build_seq(k, get_beta(), get_dataset());

  for(int i = 0; i < k && on_opt; i++)
  {
    set_reader(i);

    m_base->optimize(span, [&on_opt, &on]()
    {
      on_opt = on();
      return on_opt;
    });
  }
}

void cont::optimize_idev(int count, float dev,  std::function<bool()> on)
{
  bool on_opt = true;
  int k = get_k();

  build_seq(k, get_beta(), get_dataset());

  for(int i = 0; i < k && on_opt; i++)
  {
    set_reader(i);

    m_base->optimize_idev(count, dev, [&on_opt, &on]()
    {
      on_opt = on();
      return on_opt;
    });
  }
}

void cont::optimize_dev(float dev,  std::function<bool()> on)
{
  bool on_opt = true;
  int k = get_k();

  build_seq(k, get_beta(), get_dataset());

  for(int i = 0; i < k && on_opt; i++)
  {
    set_reader(i);

    m_base->optimize_dev(dev, [&on_opt, &on]()
    {
      on_opt = on();
      return on_opt;
    });
  }
}

void cont::optimize_eval(int count, std::function<bool()> on)
{
  bool on_opt = true;
  int k = get_k();

  build_seq(k, get_beta(), get_dataset());

  for(int i = 0; i < k && on_opt; i++)
  {
    set_reader(i);

    m_base->optimize_eval(count / k,  [&on_opt, &on]()
    {
      on_opt = on();
      return on_opt;
    });
  }
}

solution* cont::get_best()
{
  return m_base->get_best();
}

void cont::set_reader(int index)
{
  for(int i = 0; i < m_network_solutions->size(); i++)
  {
    m_network_solutions->get(i)->set_reader(m_sequence[index]);
  }
}

void cont::build_seq(int k, float beta, const reader* dataset)
{
  if(!m_built)
  {
    int n = (1 - beta) * dataset->size();

    for(reader* sub_dataset : m_sequence)
    {
      delete sub_dataset;
    }

    m_sequence.push_back(proxy_sampler::make(dataset, dataset->size()));

    for(int i = 1; i < k; i++)
    {
      reader* prior = m_sequence.back();
      m_sequence.push_back(proxy_sampler::make(prior, prior->size() - n));
    }

    std::reverse(m_sequence.begin(), m_sequence.end());

    m_built = true;
  }
}

void cont::set_params(std::vector<float> &params)
{
  algorithm::set_params(params);

  if(params.size() < 2)
  {
    throw std::invalid_argument("cont needs at least two hyper-params");
  }

  set_k(params.at(0));
  set_beta(params.at(1));
}

int cont::get_k() const
{
  return m_k;
}

void cont::set_k(int k)
{
  m_k = k;
  m_built = false;
}

float cont::get_beta() const
{
  return m_beta;
}

void cont::set_beta(float beta)
{
  m_beta = beta;
  m_built = false;
}

const reader* cont::get_dataset() const
{
  return m_dataset;
}

cont::cont(algorithm* base,
           const set<solutions::net::seq>* solutions,
           const reader* dataset)
: algorithm(solutions),
  m_base(base),
  m_network_solutions(solutions),
  m_dataset(dataset),
  m_built(false)
{

}

cont::~cont()
{
  for(auto* reader : m_sequence)
  {
    delete reader;
  }
}

} // namespace algorithms
} // namespace core
} // namespace dnn_opt

