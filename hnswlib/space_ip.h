#pragma once
#include "hnswlib.h"

namespace hnswlib {

static float
Sparse_InnerProduct(const void *pVect1, const void *pVect2) {
    vectorsizeint len1 = *((vectorsizeint *) pVect1);
    vectorsizeint len2 = *((vectorsizeint *) pVect2);

    vectordata_t *data1 = (vectordata_t *) ((vectorsizeint *) pVect1 + 1);
    vectordata_t *data2 = (vectordata_t *) ((vectorsizeint *) pVect2 + 1);
    vectorsizeint *index1 = (vectorsizeint *) (data1 + len1);
    vectorsizeint *index2 = (vectorsizeint *) (data2 + len2);

    float res = 0;
    while ((char *)index1 < (char *) pVect1 + len1 * (sizeof(vectordata_t) + sizeof(vectorsizeint)) + sizeof(vectorsizeint)
        && (char *)index2 < (char *) pVect2 + len2 * (sizeof(vectordata_t) + sizeof(vectorsizeint)) + sizeof(vectorsizeint)) {
        if (*index1 == *index2) {
            res += *data1 * *data2;
            data1++;
            data2++;
            index1++;
            index2++;
        } else if (*index1 < *index2) {
            data1++;
            index1++;
        } else {
            data2++;
            index2++;
        }
    }
    return res;
}

static float
Dense_InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    float res = 0;
    for (unsigned i = 0; i < qty; i++) {
        res += ((vectordata_t *) pVect1)[i] * ((vectordata_t *) pVect2)[i];
    }
    return res;
}

static float
Sparse_InnerProductDistance(const void *pVect1, const void *pVect2) {
    float ip = 1.0f - Sparse_InnerProduct(pVect1, pVect2);
    return ip;
}

static float
Dense_InnerProductDistance(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    float ip = 1.0f - Dense_InnerProduct(pVect1, pVect2, qty_ptr);
    return ip;
}

static float
ComputeDistance(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    float dense_dist;
    float sparse_dist;
    float dist;
    dense_dist = Dense_InnerProductDistance(pVect1, pVect2, qty_ptr);
    sparse_dist = Sparse_InnerProductDistance((char *)pVect1 + qty * sizeof(vectordata_t), (char *)pVect2 + qty * sizeof(vectordata_t));
    dist = dense_dist + sparse_dist;
    return dist;
}


// #if defined(USE_AVX)

// // Favor using AVX if available.
// static float
// InnerProductSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
//     float PORTABLE_ALIGN32 TmpRes[8];
//     float *pVect1 = (float *) pVect1v;
//     float *pVect2 = (float *) pVect2v;
//     size_t qty = *((size_t *) qty_ptr);

//     size_t qty16 = qty / 16;
//     size_t qty4 = qty / 4;

//     const float *pEnd1 = pVect1 + 16 * qty16;
//     const float *pEnd2 = pVect1 + 4 * qty4;

//     __m256 sum256 = _mm256_set1_ps(0);

//     while (pVect1 < pEnd1) {
//         //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

//         __m256 v1 = _mm256_loadu_ps(pVect1);
//         pVect1 += 8;
//         __m256 v2 = _mm256_loadu_ps(pVect2);
//         pVect2 += 8;
//         sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

//         v1 = _mm256_loadu_ps(pVect1);
//         pVect1 += 8;
//         v2 = _mm256_loadu_ps(pVect2);
//         pVect2 += 8;
//         sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
//     }

//     __m128 v1, v2;
//     __m128 sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

//     while (pVect1 < pEnd2) {
//         v1 = _mm_loadu_ps(pVect1);
//         pVect1 += 4;
//         v2 = _mm_loadu_ps(pVect2);
//         pVect2 += 4;
//         sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
//     }

//     _mm_store_ps(TmpRes, sum_prod);
//     float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
//     return sum;
// }

// static float
// InnerProductDistanceSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
//     return 1.0f - InnerProductSIMD4ExtAVX(pVect1v, pVect2v, qty_ptr);
// }

// #endif

// #if defined(USE_SSE)

// static float
// InnerProductSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
//     float PORTABLE_ALIGN32 TmpRes[8];
//     float *pVect1 = (float *) pVect1v;
//     float *pVect2 = (float *) pVect2v;
//     size_t qty = *((size_t *) qty_ptr);

//     size_t qty16 = qty / 16;
//     size_t qty4 = qty / 4;

//     const float *pEnd1 = pVect1 + 16 * qty16;
//     const float *pEnd2 = pVect1 + 4 * qty4;

//     __m128 v1, v2;
//     __m128 sum_prod = _mm_set1_ps(0);

//     while (pVect1 < pEnd1) {
//         v1 = _mm_loadu_ps(pVect1);
//         pVect1 += 4;
//         v2 = _mm_loadu_ps(pVect2);
//         pVect2 += 4;
//         sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

//         v1 = _mm_loadu_ps(pVect1);
//         pVect1 += 4;
//         v2 = _mm_loadu_ps(pVect2);
//         pVect2 += 4;
//         sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

//         v1 = _mm_loadu_ps(pVect1);
//         pVect1 += 4;
//         v2 = _mm_loadu_ps(pVect2);
//         pVect2 += 4;
//         sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

//         v1 = _mm_loadu_ps(pVect1);
//         pVect1 += 4;
//         v2 = _mm_loadu_ps(pVect2);
//         pVect2 += 4;
//         sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
//     }

//     while (pVect1 < pEnd2) {
//         v1 = _mm_loadu_ps(pVect1);
//         pVect1 += 4;
//         v2 = _mm_loadu_ps(pVect2);
//         pVect2 += 4;
//         sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
//     }

//     _mm_store_ps(TmpRes, sum_prod);
//     float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

//     return sum;
// }

// static float
// InnerProductDistanceSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
//     return 1.0f - InnerProductSIMD4ExtSSE(pVect1v, pVect2v, qty_ptr);
// }

// #endif


// #if defined(USE_AVX512)

// static float
// InnerProductSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
//     float PORTABLE_ALIGN64 TmpRes[16];
//     float *pVect1 = (float *) pVect1v;
//     float *pVect2 = (float *) pVect2v;
//     size_t qty = *((size_t *) qty_ptr);

//     size_t qty16 = qty / 16;


//     const float *pEnd1 = pVect1 + 16 * qty16;

//     __m512 sum512 = _mm512_set1_ps(0);

//     while (pVect1 < pEnd1) {
//         //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

//         __m512 v1 = _mm512_loadu_ps(pVect1);
//         pVect1 += 16;
//         __m512 v2 = _mm512_loadu_ps(pVect2);
//         pVect2 += 16;
//         sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v1, v2));
//     }

//     _mm512_store_ps(TmpRes, sum512);
//     float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] + TmpRes[13] + TmpRes[14] + TmpRes[15];

//     return sum;
// }

// static float
// InnerProductDistanceSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
//     return 1.0f - InnerProductSIMD16ExtAVX512(pVect1v, pVect2v, qty_ptr);
// }

// #endif

// #if defined(USE_AVX)

// static float
// InnerProductSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
//     float PORTABLE_ALIGN32 TmpRes[8];
//     float *pVect1 = (float *) pVect1v;
//     float *pVect2 = (float *) pVect2v;
//     size_t qty = *((size_t *) qty_ptr);

//     size_t qty16 = qty / 16;


//     const float *pEnd1 = pVect1 + 16 * qty16;

//     __m256 sum256 = _mm256_set1_ps(0);

//     while (pVect1 < pEnd1) {
//         //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

//         __m256 v1 = _mm256_loadu_ps(pVect1);
//         pVect1 += 8;
//         __m256 v2 = _mm256_loadu_ps(pVect2);
//         pVect2 += 8;
//         sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

//         v1 = _mm256_loadu_ps(pVect1);
//         pVect1 += 8;
//         v2 = _mm256_loadu_ps(pVect2);
//         pVect2 += 8;
//         sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
//     }

//     _mm256_store_ps(TmpRes, sum256);
//     float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

//     return sum;
// }

// static float
// InnerProductDistanceSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
//     return 1.0f - InnerProductSIMD16ExtAVX(pVect1v, pVect2v, qty_ptr);
// }

// #endif

// #if defined(USE_SSE)

// static float
// InnerProductSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
//     float PORTABLE_ALIGN32 TmpRes[8];
//     float *pVect1 = (float *) pVect1v;
//     float *pVect2 = (float *) pVect2v;
//     size_t qty = *((size_t *) qty_ptr);

//     size_t qty16 = qty / 16;

//     const float *pEnd1 = pVect1 + 16 * qty16;

//     __m128 v1, v2;
//     __m128 sum_prod = _mm_set1_ps(0);

//     while (pVect1 < pEnd1) {
//         v1 = _mm_loadu_ps(pVect1);
//         pVect1 += 4;
//         v2 = _mm_loadu_ps(pVect2);
//         pVect2 += 4;
//         sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

//         v1 = _mm_loadu_ps(pVect1);
//         pVect1 += 4;
//         v2 = _mm_loadu_ps(pVect2);
//         pVect2 += 4;
//         sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

//         v1 = _mm_loadu_ps(pVect1);
//         pVect1 += 4;
//         v2 = _mm_loadu_ps(pVect2);
//         pVect2 += 4;
//         sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

//         v1 = _mm_loadu_ps(pVect1);
//         pVect1 += 4;
//         v2 = _mm_loadu_ps(pVect2);
//         pVect2 += 4;
//         sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
//     }
//     _mm_store_ps(TmpRes, sum_prod);
//     float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

//     return sum;
// }

// static float
// InnerProductDistanceSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
//     return 1.0f - InnerProductSIMD16ExtSSE(pVect1v, pVect2v, qty_ptr);
// }

// #endif

// #if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
// static DISTFUNC<float> InnerProductSIMD16Ext = InnerProductSIMD16ExtSSE;
// static DISTFUNC<float> InnerProductSIMD4Ext = InnerProductSIMD4ExtSSE;
// static DISTFUNC<float> InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtSSE;
// static DISTFUNC<float> InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtSSE;

// static float
// InnerProductDistanceSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
//     size_t qty = *((size_t *) qty_ptr);
//     size_t qty16 = qty >> 4 << 4;
//     float res = InnerProductSIMD16Ext(pVect1v, pVect2v, &qty16);
//     float *pVect1 = (float *) pVect1v + qty16;
//     float *pVect2 = (float *) pVect2v + qty16;

//     size_t qty_left = qty - qty16;
//     float res_tail = InnerProduct(pVect1, pVect2);
//     return 1.0f - (res + res_tail);
// }

// static float
// InnerProductDistanceSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
//     size_t qty = *((size_t *) qty_ptr);
//     size_t qty4 = qty >> 2 << 2;

//     float res = InnerProductSIMD4Ext(pVect1v, pVect2v, &qty4);
//     size_t qty_left = qty - qty4;

//     float *pVect1 = (float *) pVect1v + qty4;
//     float *pVect2 = (float *) pVect2v + qty4;
//     float res_tail = InnerProduct(pVect1, pVect2);

//     return 1.0f - (res + res_tail);
// }
// #endif

class InnerProductSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t dense_data_size_;
    size_t dense_dim_;

 public:
    InnerProductSpace(size_t dense_dim) {
        fstdistfunc_ = ComputeDistance;
// #if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
//     #if defined(USE_AVX512)
//         if (AVX512Capable()) {
//             InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX512;
//             InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX512;
//         } else if (AVXCapable()) {
//             InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
//             InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
//         }
//     #elif defined(USE_AVX)
//         if (AVXCapable()) {
//             InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
//             InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
//         }
//     #endif
//     #if defined(USE_AVX)
//         if (AVXCapable()) {
//             InnerProductSIMD4Ext = InnerProductSIMD4ExtAVX;
//             InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtAVX;
//         }
//     #endif

//         if (dim % 16 == 0)
//             fstdistfunc_ = InnerProductDistanceSIMD16Ext;
//         else if (dim % 4 == 0)
//             fstdistfunc_ = InnerProductDistanceSIMD4Ext;
//         else if (dim > 16)
//             fstdistfunc_ = InnerProductDistanceSIMD16ExtResiduals;
//         else if (dim > 4)
//             fstdistfunc_ = InnerProductDistanceSIMD4ExtResiduals;
// #endif
        dense_dim_ = dense_dim;
        dense_data_size_ = dense_dim * sizeof(vectordata_t);
    }

    size_t get_dense_data_size() {
        return dense_data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dense_dim_;
    }

~InnerProductSpace() {}
};

}  // namespace hnswlib
