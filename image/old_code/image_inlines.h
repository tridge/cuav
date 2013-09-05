#ifndef _IMAGE_INLINES_H_
#define _IMAGE_INLINES_H_

#include <stdint.h>
#include <arpa/inet.h>

template <size_t N>
uint32_t sum_uint16(const uint16_t* data, size_t stride)
{
  uint32_t s = 0;
  for (size_t i = 0; i < N; ++i)
  {
    const uint16_t* p = data + i*stride;
    for (size_t j = 0; j < N; ++j)
    {
      s += ntohs(*p++);
    }
  }
  return s;
}

template <size_t N>
float pncc_uint16(const uint16_t* a, size_t a_stride, const uint16_t* b, size_t b_stride)
{
  // compute mean
  uint32_t m_a = sum_uint16<N>(a, a_stride)/(N*N);
  uint32_t m_b = sum_uint16<N>(b, b_stride)/(N*N);
  //printf("m_a: %d, m_b: %d\n", m_a, m_b);

  int32_t s_ab = 0;
  int32_t s_aa = 0;
  int32_t s_bb = 0;
  for (size_t i = 0; i < N; ++i)
  {
    const uint16_t* p_a = a + i * a_stride;
    const uint16_t* p_b = b + i * b_stride;
    for (size_t j = 0; j < N; ++j)
    {
      // assume 12 significant bits
      int32_t _a = ((int32_t)ntohs(*p_a++) - (int32_t)m_a) >> 4;
      int32_t _b = ((int32_t)ntohs(*p_b++) - (int32_t)m_b) >> 4;
      s_ab += (_a * _b);
      s_aa += (_a * _a);
      s_bb += (_b * _b);
    }
  }
  printf("s_ab:%d, s_aa:%d, s_bb:%d\n", s_ab, s_aa, s_bb);
  return 2.0*s_ab/(s_aa + s_bb);
}

#endif
