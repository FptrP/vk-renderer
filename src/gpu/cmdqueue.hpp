#ifndef CMDQUEUE_HPP_INCLUDED
#define CMDQUEUE_HPP_INCLUDED

#include <mutex>
#include <condition_variable> 
#include <cinttypes>
#include <atomic>

namespace gpu {

enum class QueueStatus {
  Ok,
  Overflow, 
  Closed
};

struct ByteQueue {
  ByteQueue(uint32_t buff_size);
  ~ByteQueue();

  QueueStatus write(uint32_t size, const void *data);
  QueueStatus read(uint32_t max_size, void *dst, uint32_t &readed);
  
  template <typename T>
  QueueStatus write(const T &data) {
    return write(sizeof(data), &data);
  }

  void wait_reset();
  void reset();

  void write_done();
  bool is_closed() const;

private:

  uint8_t *buffer;
  uint32_t buffer_size;
  
  std::atomic<uint8_t*> alloc_pos;
  std::atomic<uint8_t*> write_pos;
  uint8_t *read_pos;

  std::atomic_bool overflow_flag {false};
  std::atomic_bool shutdown_flag {false};
};


}

#endif