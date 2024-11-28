#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <omp.h>

#define MIN(a,b)  ((a) < (b) ? (a) : (b))

//FP32
//#define DATATYPE 0
//FP64
//#define DATATYPE 1
//I8
//#define DATATYPE 2
//I16
//#define DATATYPE 3
//I32
#define DATATYPE 4

typedef enum {FP32, FP64, I8, I16, I32} data_type_e;

#if DATATYPE == 0
typedef float data_t;
static data_type_e data_type = FP32;
#define DATA_MIN FLT_MIN
#define TYPE_IS_FP
#elif DATATYPE == 1
typedef double data_t;
static data_type_e data_type = FP64;
#define DATA_MIN DBL_MIN
#define TYPE_IS_FP
#elif DATATYPE == 2
typedef int8_t data_t;
static data_type_e data_type = I8;
#define DATA_MIN CHAR_MIN
#define TYPE_IS_INT
#elif DATATYPE == 3
typedef int16_t data_t;
static data_type_e data_type = I16;
#define DATA_MIN SHRT_MIN
#define TYPE_IS_INT
#elif DATATYPE == 4
typedef int32_t data_t;
static data_type_e data_type = I32;
#define DATA_MIN INT_MIN
#define TYPE_IS_INT
#else
   #error Unsupported choice setting
#endif


void init_random_tensor(void * tensor, data_type_e data_type, size_t nmemb);

void gemm(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n, int k,
          int stride_0, int stride_1, int stride_2);

void gemm_t(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n, int k,
          int stride_0, int stride_1, int stride_2);

void scale(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n);

void add(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n);

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

   fprintf(stdout, "Model n_ranks: %d, Sequence lenght: %d, head count: %d\n", dmodel, S, h);

   data_t *embeddings = NULL;
   data_t * Qw = NULL;
   data_t * Kw = NULL;
   data_t * Vw = NULL;

   data_t * Qw_heads = NULL;
   data_t * Kw_heads = NULL;
   data_t * Vw_heads = NULL;

   data_t * Q = NULL;
   data_t * K = NULL;
   data_t * V = NULL;

   data_t * KQ = NULL;
   data_t * softmax_out = NULL;

   data_t * QKV = NULL;

   data_t * ATTNw = NULL;
   data_t * ATTNout = NULL;

   data_t scale_f = 1.0f/sqrtf(((data_t)dk));

   srand(time(NULL));

   clock_gettime(CLOCK_MONOTONIC, &start);

   embeddings = calloc(dmodel*S, sizeof(data_t));

   ATTNw = calloc(dmodel*dmodel, sizeof(data_t));

   init_random_tensor(embeddings, data_type, dmodel*S);
   init_random_tensor(ATTNw, data_type, dmodel*dmodel);
   Qw = calloc(dmodel*dmodel, sizeof(data_t));
   init_random_tensor(Qw, data_type, dmodel*dmodel);

   Kw = calloc(dmodel*dmodel, sizeof(data_t));
   init_random_tensor(Kw, data_type, dmodel*dmodel);

   Vw = calloc(dmodel*dmodel, sizeof(data_t));
   init_random_tensor(Vw, data_type, dmodel*dmodel);

   Q = calloc(S*dmodel, sizeof(data_t));
   memset(Q, 0, S*dmodel*sizeof(data_t));

   K = calloc(S*dmodel, sizeof(data_t));
   memset(K, 0, S*dmodel*sizeof(data_t));

   V = calloc(S*dmodel, sizeof(data_t));
   memset(V, 0, S*dmodel*sizeof(data_t));

   KQ = calloc(h*S*S, sizeof(data_t));
   memset(KQ, 0, h*S*S*sizeof(data_t));

   softmax_out = calloc(h*S*S, sizeof(data_t));
   memset(softmax_out, 0, h*S*S*sizeof(data_t));

   QKV = calloc(S*dmodel, sizeof(data_t));
   memset(QKV, 0, S*dmodel*sizeof(data_t));

   ATTNout = calloc(S*dmodel, sizeof(data_t));
   memset(ATTNout, 0, S*dmodel*sizeof(data_t));

   clock_gettime(CLOCK_MONOTONIC, &end);

   time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
   printf("Init time: %.2f ms\n", time_elapsed_s * 1000);

   clock_gettime(CLOCK_MONOTONIC, &start);
   /* MHA */

   gemm(Q, embeddings, Qw, data_type, S, dmodel, dmodel, dmodel, dmodel, dmodel);
   gemm(K, embeddings, Kw, data_type, S, dmodel, dmodel, dmodel, dmodel, dmodel);
   gemm(V, embeddings, Vw, data_type, S, dmodel, dmodel, dmodel, dmodel, dmodel);

   for(int i = 0; i < h; i++) {
      gemm_t(&KQ[S*S*i], &Q[dmodel/h*i], K, data_type, S, S, dmodel/h, S, dmodel, dmodel);
   }

   scale(KQ, KQ, ((void*)&scale_f), data_type, h*S, S);

   softmax(softmax_out, KQ, data_type, h*S, S);

   for(int i = 0; i < h; i++) {
      gemm(&QKV[dmodel/h*i], &softmax_out[S*S*i], &V[dmodel/h*i], data_type, S, dmodel, S, dmodel, S, dmodel);
   }

   gemm(ATTNout, QKV, ATTNw, data_type, S, dmodel, dmodel, dmodel, dmodel, dmodel);

   add(embeddings, ATTNout, embeddings, data_type, S, dmodel);


   clock_gettime(CLOCK_MONOTONIC, &end);

   time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
   const uint64_t flop_count = 4*2*S*dmodel*dmodel + (1+h)*2*S*S*dmodel + h*S*S + 7*h*S*S + S*dmodel;
   printf("MHA execution time: %.2f ms flop count: %ld\n", time_elapsed_s * 1000, flop_count);

   free(Qw);
   free(Kw);
   free(Vw);

   free(embeddings);
   free(Qw_heads);
   free(Kw_heads);
   free(Vw_heads);
   free(Q);
   free(K);
   free(V);
   free(KQ);
   free(softmax_out);
   free(QKV);
   free(ATTNw);
   free(ATTNout);

   return 0;
}

static size_t get_element_size(data_type_e type) {
   size_t size;
   switch (type) {
      case I8:
         size = sizeof(uint8_t);
         break;
      case I16:
         size = sizeof(uint16_t);
         break;
      case I32:
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


static void init_random_tensor_impl(data_t * tensor, size_t nmemb) {
   const data_t range = 10;
   #pragma omp parallel for shared (tensor)
   for(int i = 0; i < nmemb; i++)
      tensor[i] = ((data_t)rand()/(data_t)(RAND_MAX)) * range;
}

void init_random_tensor(void * tensor, data_type_e data_type, size_t nmemb) {
   assert(tensor);
   init_random_tensor_impl(((data_t*)tensor), nmemb);
}

static void gemm_impl(data_t * dst, const data_t * src1, const data_t * src2, int m, int n, int k,
      int stride_0, int stride_1, int stride_2) {
   const int bsize = 64;
   int ii0, ii1, ii2;
   int i0, i1, i2;
   data_t pp;
   #pragma omp parallel for shared (dst, src1, src2) private(i0,i1,i2,ii0,ii1,ii2,pp) collapse(3)
   for (ii0 = 0; ii0<m; ii0+=bsize) {
      for (ii1 = 0; ii1<n; ii1+=bsize) {
         for(ii2 = 0; ii2<k; ii2+=bsize) {
            for (i0 = ii0; i0 < MIN(ii0+bsize,m); i0++) {
               for (i1 = ii1; i1 < MIN(ii1+bsize,n); i1++) {
                  pp = 0;
                  for (i2 = ii2; i2 < MIN(ii2+bsize,k); i2++) {
                     pp += src1[i0*(stride_1)+i2] * src2[i2*(stride_2)+i1];
                  }
                  dst[i0*(stride_0)+i1]+= pp;
               }
            }
         }
      }
   }
}

static void gemm_t_impl(data_t * dst, const data_t * src1, const data_t * src2, int m, int n, int k,
      int stride_0, int stride_1, int stride_2) {
   const int bsize = 64;
   int ii0, ii1, ii2;
   int i0, i1, i2;
   data_t pp;
   #pragma omp parallel for shared (dst, src1, src2) private(i0,i1,i2,ii0,ii1,ii2,pp) collapse(3)
   for (ii0 = 0; ii0<m; ii0+=bsize) {
      for (ii1 = 0; ii1<n; ii1+=bsize) {
         for(ii2 = 0; ii2<k; ii2+=bsize) {
            for (i0 = ii0; i0 < MIN(ii0+bsize,m); i0++) {
               for (i1 = ii1; i1 < MIN(ii1+bsize,n); i1++) {
                  pp = 0;
                  for (i2 = ii2; i2 < MIN(ii2+bsize,k); i2++) {
                     pp += src1[i0*stride_1+i2] * src2[i1*stride_2+i2];
                  }
                  dst[i0*stride_0+i1]+= pp;
               }
            }
         }
      }
   }
}

static void scale_impl(data_t * dst, const data_t * src1, const data_t src2, int m, int n) {
   int i;
   #pragma omp parallel for shared (dst, src1) private(i)
   for(i = 0; i < m*n; i++)
      dst[i] = src1[i] * src2;
}

static void add_impl(data_t * dst, const data_t * src1, const data_t * src2, int m, int n) {
   int i;
   #pragma omp parallel for shared (dst, src1) private(i)
   for(i = 0; i < m*n; i++)
      dst[i] = src1[i] + src2[i];
}

static void softmax_impl(data_t * dst, const data_t * src, int m, int n) {
   int i, j;
   data_t max, sum;
   #pragma omp parallel for shared (dst) private(i, j, max, sum)
   for(i = 0; i < m; i++) {
      max = DATA_MIN;
      for(j = 0; j < n; j++)
         max = (max > src[i*n+j]) ? max : src[i*n+j];

      sum = 0.0;
      for(j = 0; j < n; j++) {
         const data_t e = expf(src[i*n+j] - max);
         sum += e;
         dst[i*n+j] = e;
      }

      for(j = 0; j < n; j++) {
         dst[i*n+j] *= sum;
      }
   }
}


void gemm(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n, int k,
          int stride_0, int stride_1, int stride_2) {
   assert(dst);
   assert(src1);
   assert(src2);
   gemm_impl(((data_t*)dst), ((data_t*)src1), ((data_t*)src2), m, n, k, stride_0, stride_1, stride_2);
}

void gemm_t(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n, int k,
          int stride_0, int stride_1, int stride_2) {
   assert(dst);
   assert(src1);
   assert(src2);
   gemm_t_impl(((data_t*)dst), ((data_t*)src1), ((data_t*)src2), m, n, k, stride_0, stride_1, stride_2);
}

void scale(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n) {
   assert(dst);
   assert(src1);
   assert(src2);
   scale_impl(((data_t*)dst), ((data_t*)src1), *((data_t*)src2), m, n);
}

void add(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n) {
   assert(dst);
   assert(src1);
   assert(src2);
   add_impl(((data_t*)dst), ((data_t*)src1), ((data_t*)src2), m, n);
}


void softmax(void * dst, void * src, data_type_e data_type, int m, int n) {
   assert(dst);
   assert(src);
   softmax_impl(((data_t*)dst), ((data_t*)src), m, n);
}
