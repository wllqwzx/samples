#include <cstdio>

template <typename T> void random_init(T *data, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<T>(10 * rand() / RAND_MAX);
  }
}
