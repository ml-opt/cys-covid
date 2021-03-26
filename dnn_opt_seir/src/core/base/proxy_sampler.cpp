#include <stdexcept>
#include <algorithm>
#include <core/base/proxy_sampler.h>

namespace dnn_opt
{
namespace core
{

proxy_sampler* proxy_sampler::make(const reader* reader, int limit, int offset)
{
  return new proxy_sampler(reader, limit, offset);
}

proxy_sampler** proxy_sampler::make_fold(const reader* reader, int folds, int overlap)
{
  proxy_sampler** proxy_samplers = new proxy_sampler*[folds];
  int sample_size = reader->size() / folds;
  int limit = std::min(reader->size(), sample_size + overlap / 2);

  for(int i = 0; i < folds; i++)
  {
    int offset = std::max(0, i * sample_size - overlap / 2);

    proxy_samplers[i] = new proxy_sampler(reader, limit, offset);
  }

  return proxy_samplers;
}

proxy_sampler** proxy_sampler::make_fold_prop(const reader* reader, int folds, float overlap)
{
  int overlap_proportion = reader->size() * overlap;

  if(overlap < 0 || overlap > 1.0f)
  {
    throw std::out_of_range("overlap parameters should be in the range [0, 1]");
  }

  return make_fold(reader, folds, overlap_proportion);
}

float* proxy_sampler::in_data() const
{
  return _reader->in_data() + _offset * get_in_dim();
}

float* proxy_sampler::out_data() const
{
  return _reader->out_data() + _offset * get_out_dim();
}

int proxy_sampler::get_in_dim() const
{
  return _reader->get_in_dim();
}

int proxy_sampler::get_out_dim() const
{
  return _reader->get_out_dim();
}

int proxy_sampler::size() const
{
  return _limit;
}

proxy_sampler::proxy_sampler(const reader* reader, int limit, int offset)
: _reader(reader)
{
  if(offset + limit > reader->size())
  {
    throw std::out_of_range("the sampler limit is beyond the reader size");
  }

  _limit = limit;
  _offset = offset;
}

proxy_sampler::~proxy_sampler()
{

}

} // namespace core
} // namespace dnn_opt
