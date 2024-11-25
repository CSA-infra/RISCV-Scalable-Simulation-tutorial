#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>

#define MIN(a,b)  ((a) < (b) ? (a) : (b))

typedef enum {FP8, FP16, FP32, FP64} data_type_e;

void init_random_tensor(void * tensor, data_type_e data_type, size_t nmemb);

void gemm(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n, int k);

void gemm_t(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n, int k);

void scale(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n);

void softmax(void * dst, void * src, data_type_e data_type, int m, int n);

int main(int argc, char ** argv) {

   if(argc != 4) {
      fprintf(stderr, "Usage: %s MODEL_SIZE SEQUENCE_LENGHT HEAD_COUNT\n", argv[0]);
      exit(EXIT_FAILURE);
   }
   struct timespec start, end;
   double time_elapsed_s;
   int dmodel = atoi(argv[1]);
   int h = atoi(argv[3]);
   int S = atoi(argv[2]);
   int dk = dmodel/h;
   int dv = dmodel/h;
   data_type_e data_type = FP32;

   fprintf(stdout, "Model size: %d, Sequence lenght: %d, head count: %d\n", dmodel, S, h);


   float * embeddings = NULL;
   float * Qw = NULL;
   float * Kw = NULL;
   float * Vw = NULL;
   float * Q = NULL;
   float * K = NULL;
   float * V = NULL;
   float * KQ = NULL;
   float * softmax_out = NULL;
   float * QKV = NULL;
   float scale_f = 1.0f/sqrtf(((float)dk));

   srand(time(NULL));

   clock_gettime(CLOCK_MONOTONIC, &start);

   embeddings = calloc(dmodel*S, sizeof(float));
   assert(embeddings);
   init_random_tensor(embeddings, FP32, dmodel*S);

   Qw = calloc(dmodel*dmodel, sizeof(float));
   assert(Qw);
   init_random_tensor(Qw, FP32, dmodel*dmodel);

   Kw = calloc(dmodel*dmodel, sizeof(float));
   assert(Kw);
   init_random_tensor(Kw, FP32, dmodel*dmodel);

   Vw = calloc(dmodel*dmodel, sizeof(float));
   assert(Vw);
   init_random_tensor(Vw, FP32, dmodel*dmodel);

   Q = calloc(S*dmodel, sizeof(float));
   assert(Q);
   memset(Q, 0, S*dmodel*sizeof(float));

   K = calloc(S*dmodel, sizeof(float));
   assert(K);
   memset(K, 0, S*dmodel*sizeof(float));

   V = calloc(S*dmodel, sizeof(float));
   assert(V);
   memset(V, 0, S*dmodel*sizeof(float));

   KQ = calloc(S*S, sizeof(float));
   assert(KQ);
   memset(KQ, 0, S*S*sizeof(float));

   softmax_out = calloc(S*S, sizeof(float));
   assert(softmax_out);
   memset(softmax_out, 0, S*S*sizeof(float));

   QKV = calloc(S*dmodel, sizeof(float));
   assert(QKV);
   memset(QKV, 0, S*dmodel*sizeof(float));

   clock_gettime(CLOCK_MONOTONIC, &end);

   time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
   printf("Init time: %.2f ms\n", time_elapsed_s * 1000);

   clock_gettime(CLOCK_MONOTONIC, &start);
   /* MHA */

   gemm(Q, embeddings, Qw, FP32, S, dmodel, dmodel);
   gemm(K, embeddings, Kw, FP32, S, dmodel, dmodel);
   gemm(V, embeddings, Vw, FP32, S, dmodel, dmodel);

   gemm_t(KQ, Q, K, FP32, S, dmodel, S);

   scale(KQ, KQ, ((void*)&scale_f), FP32, S, S);

   softmax(softmax_out, KQ, FP32, S, S);

   gemm(QKV, softmax_out, V, FP32, S, dmodel, S);

   clock_gettime(CLOCK_MONOTONIC, &end);

   time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
   printf("MHA execution time: %.2f ms\n", time_elapsed_s * 1000);

   free(embeddings);
   free(Qw);
   free(Kw);
   free(Vw);
   free(Q);
   free(K);
   free(V);
   free(KQ);
   free(softmax_out);
   free(QKV);


   return 0;
}

static size_t get_element_size(data_type_e type) {
   size_t size;
   switch (type) {
      case FP8:
         size = sizeof(uint8_t);
         break;
      case FP16:
         size = sizeof(uint16_t);
         break;
      case FP32:
         size = sizeof(uint32_t);
         break;
      case FP64:
         size = sizeof(uint64_t);
         break;
      default:
         size = -1;
         break;
   }

   assert(size > 0 && "data type unknown");

   return size;
}


static void init_random_tensor_fp32(float * tensor, size_t nmemb) {
   #pragma omp parallel for shared (tensor)
   for(int i = 0; i < nmemb; i++)
      tensor[i] = ((float)rand()/(float)(RAND_MAX)) * 10.0;
}

void init_random_tensor(void * tensor, data_type_e data_type, size_t nmemb) {
   assert(tensor);
   switch(data_type) {
      case FP32:
         init_random_tensor_fp32(((float*)tensor), nmemb);
         break;
      default:
         fprintf(stderr, "[%s:%d] Data type not supported\n", __func__, __LINE__);
         break;
   }
}

static void gemm_fp32(float * dst, const float * src1, const float * src2, int m, int n, int k) {
   const int bsize = 64;
   int ii0, ii1, ii2;
   int i0, i1, i2;
   float pp;
   #pragma omp parallel for shared (dst, src1, src2) private(i0,i1,i2,ii0,ii1,ii2,pp) collapse(3)
   for (ii0 = 0; ii0<m; ii0+=bsize) {
      for (ii1 = 0; ii1<n; ii1+=bsize) {
         for(ii2 = 0; ii2<k; ii2+=bsize) {
            for (i0 = ii0; i0 < MIN(ii0+bsize,m); i0++) {
               for (i1 = ii1; i1 < MIN(ii1+bsize,n); i1++) {
                  pp = 0;
                  for (i2 = ii2; i2 < MIN(ii2+bsize,k); i2++) {
                     pp += src1[i0*k+i2] * src2[i2*n+i1];
                  }
                  dst[i0*n+i1]+= pp;
               }
            }
         }
      }
   }
}

static void gemm_t_fp32(float * dst, const float * src1, const float * src2, int m, int n, int k) {
   const int bsize = 64;
   int ii0, ii1, ii2;
   int i0, i1, i2;
   float pp;
   #pragma omp parallel for shared (dst, src1, src2) private(i0,i1,i2,ii0,ii1,ii2,pp) collapse(3)
   for (ii0 = 0; ii0<m; ii0+=bsize) {
      for (ii1 = 0; ii1<n; ii1+=bsize) {
         for(ii2 = 0; ii2<k; ii2+=bsize) {
            for (i0 = ii0; i0 < MIN(ii0+bsize,m); i0++) {
               for (i1 = ii1; i1 < MIN(ii1+bsize,n); i1++) {
                  pp = 0;
                  for (i2 = ii2; i2 < MIN(ii2+bsize,k); i2++) {
                     pp += src1[i0*k+i2] * src2[i1*k+i2];
                  }
                  dst[i0*n+i1]+= pp;
               }
            }
         }
      }
   }
}

static void scale_fp32(float * dst, const float * src1, const float src2, int m, int n) {
   int i, j;
   #pragma omp parallel for shared (dst, src1) private(i)
   for(i = 0; i < m*n; i++)
      dst[i] = src1[i] * src2;
}

static void softmax_fp32(float * dst, const float * src, int m, int n) {
   int i, j;
   float max, sum;
   #pragma omp parallel for shared (dst) private(i, j, max, sum)
   for(i = 0; i < m; i++) {
      max = FLT_MIN;
      for(j = 0; j < n; j++)
         max = (max > src[i*n+j]) ? max : src[i*n+j];

      sum = 0.0;
      for(j = 0; j < n; j++) {
         const float e = expf(src[i*n+j] - max);
         sum += e;
         dst[i*n+j] = e;
      }

      for(j = 0; j < n; j++) {
         dst[i*n+j] *= sum;
      }
   }
}


void gemm(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n, int k) {
   assert(dst);
   assert(src1);
   assert(src2);
   switch(data_type) {
      case FP32:
         gemm_fp32(((float*)dst), ((float*)src1), ((float*)src2), m, n, k);
         break;
      default:
         fprintf(stderr, "[%s:%d] Data type not supported\n", __func__, __LINE__);
         break;
   }
}

void gemm_t(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n, int k) {
   assert(dst);
   assert(src1);
   assert(src2);
   switch(data_type) {
      case FP32:
         gemm_t_fp32(((float*)dst), ((float*)src1), ((float*)src2), m, n, k);
         break;
      default:
         fprintf(stderr, "[%s:%d] Data type not supported\n", __func__, __LINE__);
         break;
   }
}

void scale(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n) {
   assert(dst);
   assert(src1);
   assert(src2);
   switch(data_type) {
      case FP32:
         scale_fp32(((float*)dst), ((float*)src1), *((float*)src2), m, n);
         break;
      default:
         fprintf(stderr, "[%s:%d] Data type not supported\n", __func__, __LINE__);
         break;
   }
}

void softmax(void * dst, void * src, data_type_e data_type, int m, int n) {
   assert(dst);
   assert(src);
   switch(data_type) {
      case FP32:
         softmax_fp32(((float*)dst), ((float*)src), m, n);
         break;
      default:
         fprintf(stderr, "[%s:%d] Data type not supported\n", __func__, __LINE__);
         break;
   }
}
