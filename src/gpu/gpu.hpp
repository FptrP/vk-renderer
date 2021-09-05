#ifndef GPU_HPP_INCLUDED
#define GPU_HPP_INCLUDED

#include "init.hpp"
#include "image.hpp"

namespace gpu {

struct Instance {
  Instance(const DeviceConfig &config) : gpu {config} {}
  ~Instance() {}

  Image create_image() { return Image{gpu}; }
  Device &get_device() { return gpu; }

private:
  Device gpu;
};


}

#endif