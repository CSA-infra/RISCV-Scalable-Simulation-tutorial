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

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

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

void gemm(void * dst, void * src1, void * src2, data_type_e data_type, int heads, int m, int n, int k,
          int stride_0, int stride_1, int stride_2);
int main(int argc, char ** argv) {

   if(argc != 4) {
      fprintf(stderr, "Usage: %s M N K\n", argv[0]);
      exit(EXIT_FAILURE);
   }

   struct timespec start, end;
   double time_elapsed_s;
   int m = atoi(argv[1]);
   int n = atoi(argv[3]);
   int k = atoi(argv[2]);

   fprintf(stdout, "M: %d, N: %d, K: %d\n", m, n, k);

   data_t * A = NULL;
   data_t * B = NULL;
   data_t * C = NULL;

   srand(time(NULL));

   clock_gettime(CLOCK_MONOTONIC, &start);

   A = calloc(m*k, sizeof(data_t));
   B = calloc(k*n, sizeof(data_t));
   C = calloc(m*n, sizeof(data_t));


   init_random_tensor(A, data_type, m*k);
   init_random_tensor(B, data_type, n*k);

   clock_gettime(CLOCK_MONOTONIC, &end);

   time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
   printf("Init time: %.2f ms\n", time_elapsed_s * 1000);

   clock_gettime(CLOCK_MONOTONIC, &start);
   /* MHA */

   gemm(C, A, B, data_type, 1, m, n, k, m, n, k);

   clock_gettime(CLOCK_MONOTONIC, &end);

   time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
   const uint64_t flop_count = m*n*k*2;

   printf("Execution time: %.2f ms flop count: %lu\n", time_elapsed_s * 1000, flop_count);

   free(A);
   free(B);
   free(C);

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

static void gemm_impl(data_t * dst, const data_t * src1, const data_t * src2, int heads, int m, int n, int k,
      int stride_0, int stride_1, int stride_2) {
   const int bsize = TILE_SIZE;
   int ii0, ii1, ii2;
   int i0, i1, i2;
   int h;
   int start_head, stop_head;
   data_t pp;
   #pragma omp parallel for shared (dst, src1, src2) private(h,i0,i1,i2,ii0,ii1,ii2,pp) collapse(2)
   for (h=0; h < heads; h++) {
      for (ii0 = 0; ii0<m; ii0+=bsize) {
         for (ii1 = h*n; ii1<((h+1)*n); ii1+=bsize) {
            for(ii2 = 0; ii2<k; ii2+=bsize) {
               for (i0 = ii0; i0 < MIN(ii0+bsize,m); i0++) {
                  for (i1 = ii1; i1 < MIN(ii1+bsize,((h+1)*n)); i1++) {
                     pp = 0;
                     for (i2 = ii2; i2 < MIN(ii2+bsize,k); i2++) {
                        pp += src1[(i0+h*m)*(stride_1)+i2] * src2[i2*stride_2+i1];
                     }
                     dst[i0*(stride_0)+i1]+= pp;
                  }
               }
            }
         }
      }
   }
}

void gemm(void * dst, void * src1, void * src2, data_type_e data_type, int heads, int m, int n, int k,
          int stride_0, int stride_1, int stride_2) {
   assert(dst);
   assert(src1);
   assert(src2);
   assert(heads*n == stride_0);
   gemm_impl(((data_t*)dst), ((data_t*)src1), ((data_t*)src2), heads, m, n, k, stride_0, stride_1, stride_2);
}
