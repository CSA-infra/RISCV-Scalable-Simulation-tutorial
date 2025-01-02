#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

#define MIN(a,b)  ((a) < (b) ? (a) : (b))

#define WORLD MPI_COMM_WORLD

#define CHECK_RES


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
MPI_Datatype mpi_data_type = MPI_FLOAT;
static data_type_e data_type = FP32;
#define TYPE_IS_FP
#elif DATATYPE == 1
typedef double data_t;
MPI_Datatype mpi_data_type = MPI_DOUBLE;
static data_type_e data_type = FP64;
#define TYPE_IS_FP
#elif DATATYPE == 2
typedef int8_t data_t;
MPI_Datatype mpi_data_type = MPI_INT8_T;
static data_type_e data_type = I8;
#define TYPE_IS_INT
#elif DATATYPE == 3
typedef int16_t data_t;
MPI_Datatype mpi_data_type = MPI_INT16_T;
static data_type_e data_type = I16;
#define TYPE_IS_INT
#elif DATATYPE == 4
typedef int32_t data_t;
MPI_Datatype mpi_data_type = MPI_INT32_T;
static data_type_e data_type = I32;
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

#ifdef CHECK_RES

void gemm_ref(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n, int k,
          int stride_0, int stride_1, int stride_2);

void gemm_t_ref(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n, int k,
          int stride_0, int stride_1, int stride_2);

void scale_ref(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n);

void add_ref(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n);

void softmax_ref(void * dst, void * src, data_type_e data_type, int m, int n);

#if defined(TYPE_IS_INT)
static int cmp(data_t rhs, data_t lhs) {
   return (rhs) - (lhs);
}
static void print_diff(int count, int pos, data_t ref, data_t res) {
   printf("Difference %d at %d ref = %d mpi = %d\n", count, pos, ref, res);
}
#elif defined(TYPE_IS_FP)
static int cmp(data_t rhs, data_t lhs) {
   int ret = 0;
   data_t diff = roundf(rhs) - roundf(lhs);
   ret = diff > 1.0 ? 1 : 0;
   return ret;
}
static void print_diff(int count, int pos, data_t ref, data_t res) {
   printf("Difference %d at %d ref = %.4f mpi = %.4f\n", count, pos, ref, res);
}
#else
   #error Unsupported choice setting
#endif

#endif // CHECK_RES

int main(int argc, char ** argv) {
   const int root = 0;
   int n_ranks, rank;
   MPI_Request emb_req, Qw_req, Kw_req, Vw_req, attn_w_req;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(WORLD, &n_ranks);
   MPI_Comm_rank(WORLD, &rank);

   MPI_Datatype col, col_type;

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

   if((dmodel%n_ranks) != 0) {
      fprintf(stderr, "Error: dmodel must be a multiple of the number of ranks (dmodel: %d, rank: %d)\n", dmodel, rank);
      exit(EXIT_FAILURE);
   }

   fprintf(stdout, "Model n_ranks: %d, Sequence lenght: %d, head count: %d\n", dmodel, S, h);


   MPI_Type_vector(dmodel, dmodel/n_ranks, dmodel, mpi_data_type, &col);
   MPI_Type_commit(&col);
   MPI_Type_create_resized(col, 0, dmodel/n_ranks*sizeof(data_t), &col_type);
   MPI_Type_commit(&col_type);


   assert(n_ranks <= h && (h % n_ranks) == 0);

#ifdef CHECK_RES
   data_t * embeddings_ref = NULL;
#endif

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

   if(rank == root) {
      init_random_tensor(embeddings, data_type, dmodel*S);
      init_random_tensor(ATTNw, data_type, dmodel*dmodel);
#ifdef CHECK_RES
      embeddings_ref = calloc(S*dmodel,sizeof(data_t));
      memcpy(embeddings_ref, embeddings, S*dmodel*sizeof(data_t));
#endif
   }

   MPI_Ibcast(embeddings, dmodel*S, mpi_data_type, root, WORLD, &emb_req);
   MPI_Ibcast(ATTNw, dmodel*dmodel, mpi_data_type, root, WORLD, &attn_w_req);

   if(rank == root) {
      Qw = calloc(dmodel*dmodel, sizeof(data_t));
      init_random_tensor(Qw, data_type, dmodel*dmodel);

      Kw = calloc(dmodel*dmodel, sizeof(data_t));
      init_random_tensor(Kw, data_type, dmodel*dmodel);

      Vw = calloc(dmodel*dmodel, sizeof(data_t));
      init_random_tensor(Vw, data_type, dmodel*dmodel);
   }

   Qw_heads = calloc(dmodel*dmodel/n_ranks, sizeof(data_t));
   Kw_heads = calloc(dmodel*dmodel/n_ranks, sizeof(data_t));
   Vw_heads = calloc(dmodel*dmodel/n_ranks, sizeof(data_t));

   MPI_Iscatter(Qw, 1, col_type, Qw_heads, dmodel*dmodel/n_ranks, mpi_data_type, root, WORLD, &Qw_req);
   MPI_Iscatter(Kw, 1, col_type, Kw_heads, dmodel*dmodel/n_ranks, mpi_data_type, root, WORLD, &Kw_req);
   MPI_Iscatter(Vw, 1, col_type, Vw_heads, dmodel*dmodel/n_ranks, mpi_data_type, root, WORLD, &Vw_req);

   Q = calloc(S*dmodel/n_ranks, sizeof(data_t));
   memset(Q, 0, S*dmodel*sizeof(data_t)/n_ranks);

   K = calloc(S*dmodel/n_ranks, sizeof(data_t));
   memset(K, 0, S*dmodel*sizeof(data_t)/n_ranks);

   V = calloc(S*dmodel/n_ranks, sizeof(data_t));
   memset(V, 0, S*dmodel*sizeof(data_t)/n_ranks);

   KQ = calloc(h/n_ranks*S*S, sizeof(data_t));
   memset(KQ, 0, h/n_ranks*S*S*sizeof(data_t));

   softmax_out = calloc(h/n_ranks*S*S, sizeof(data_t));
   memset(softmax_out, 0, h/n_ranks*S*S*sizeof(data_t));

   QKV = calloc(S/n_ranks*dmodel, sizeof(data_t));
   memset(QKV, 0, S/n_ranks*dmodel*sizeof(data_t));

   ATTNout = calloc(S*dmodel, sizeof(data_t));
   memset(ATTNout, 0, S*dmodel*sizeof(data_t));

   MPI_Wait(&emb_req, MPI_STATUS_IGNORE);
   MPI_Wait(&Qw_req, MPI_STATUS_IGNORE);
   MPI_Wait(&Kw_req, MPI_STATUS_IGNORE);
   MPI_Wait(&Vw_req, MPI_STATUS_IGNORE);
   MPI_Wait(&attn_w_req, MPI_STATUS_IGNORE);

   clock_gettime(CLOCK_MONOTONIC, &end);

   time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
   printf("[rank: %d] Init time: %.2f ms\n", rank, time_elapsed_s * 1000);

   clock_gettime(CLOCK_MONOTONIC, &start);
   /* MHA */

   gemm(Q, embeddings, Qw_heads, data_type, S, dmodel/n_ranks, dmodel, dmodel/n_ranks, dmodel, dmodel/n_ranks);
   gemm(K, embeddings, Kw_heads, data_type, S, dmodel/n_ranks, dmodel, dmodel/n_ranks, dmodel, dmodel/n_ranks);
   gemm(V, embeddings, Vw_heads, data_type, S, dmodel/n_ranks, dmodel, dmodel/n_ranks, dmodel, dmodel/n_ranks);

   for(int i = 0; i < h/n_ranks; i++) {
      gemm_t(&KQ[S*S*i], &Q[dmodel/h*i], K, data_type, S, S, dmodel/h, S, dmodel/n_ranks, dmodel/n_ranks);
   }

   scale(KQ, KQ, ((void*)&scale_f), data_type, h/n_ranks*S, S);

   softmax(softmax_out, KQ, data_type, h/n_ranks*S, S);

   for(int i = 0; i < h/n_ranks; i++) {
      gemm(&QKV[dmodel/h*i], &softmax_out[S*S*i], &V[dmodel/h*i], data_type, S, dmodel/h, S, dmodel/n_ranks, S, dmodel/n_ranks);
   }

   gemm(ATTNout, QKV, ATTNw, data_type, S, dmodel, dmodel/n_ranks, dmodel, dmodel/n_ranks, dmodel);

   add(&ATTNout[S/n_ranks*rank*dmodel], &ATTNout[S/n_ranks*rank*dmodel], &embeddings[S/n_ranks*rank*dmodel], data_type, S/n_ranks, dmodel);


   MPI_Allreduce(ATTNout, embeddings, S*dmodel, mpi_data_type, MPI_SUM, WORLD);

   clock_gettime(CLOCK_MONOTONIC, &end);

   time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
   printf("[rank: %d] MHA execution time: %.2f ms\n", rank, time_elapsed_s * 1000);

#ifdef CHECK_RES
   if(rank == root) {
      data_t * Q_ref = calloc(S*dmodel, sizeof(data_t));
      assert(Q_ref);
      memset(Q_ref, 0, S*dmodel*sizeof(data_t));

      data_t * K_ref = calloc(S*dmodel, sizeof(data_t));
      assert(K_ref);
      memset(K_ref, 0, S*dmodel*sizeof(data_t));

      data_t * V_ref = calloc(S*dmodel, sizeof(data_t));
      assert(V_ref);
      memset(V_ref, 0, S*dmodel*sizeof(data_t));

      data_t * KQ_ref = calloc(S*S*h, sizeof(data_t));
      assert(KQ_ref);
      memset(KQ_ref, 0, h*S*S*sizeof(data_t));

      data_t * softmax_out_ref = calloc(h*S*S, sizeof(data_t));
      assert(softmax_out_ref);
      memset(softmax_out_ref, 0, h*S*S*sizeof(data_t));

      data_t * QKV_ref = calloc(S*dmodel, sizeof(data_t));
      assert(QKV_ref);
      memset(QKV_ref, 0, S*dmodel*sizeof(data_t));

      data_t * ATTNout_ref = calloc(S*dmodel, sizeof(data_t));
      assert(ATTNout_ref);
      memset(ATTNout_ref, 0, S*dmodel*sizeof(data_t));

      gemm_ref(Q_ref, embeddings_ref, Qw, data_type, S, dmodel, dmodel, dmodel, dmodel, dmodel);
      gemm_ref(K_ref, embeddings_ref, Kw, data_type, S, dmodel, dmodel, dmodel, dmodel, dmodel);
      gemm_ref(V_ref, embeddings_ref, Vw, data_type, S, dmodel, dmodel, dmodel, dmodel, dmodel);

      for(int i = 0; i < h; i++) {
         gemm_t_ref(&KQ_ref[S*S*i], &Q_ref[dmodel/h*i], K_ref, data_type, S, S, dmodel/h, S, dmodel, dmodel);
      }

      scale_ref(KQ_ref, KQ_ref, ((void*)&scale_f), data_type, S*h, S);

      softmax_ref(softmax_out_ref, KQ_ref, data_type, S*h, S);

      for(int i = 0; i < h; i++) {
         gemm_ref(&QKV_ref[dmodel/h*i], &softmax_out_ref[S*S*i], &V_ref[dmodel/h*i], data_type, S, dmodel/h, S, dmodel, S, dmodel);
      }

      gemm_ref(ATTNout_ref, QKV_ref, ATTNw, data_type, S, dmodel, dmodel, dmodel, dmodel, dmodel);

      add_ref(embeddings_ref, embeddings_ref, ATTNout_ref, data_type, S, dmodel);

      int count = 0;
      for(int i = 0; i < S*dmodel; i++)
         if(cmp(embeddings_ref[i], embeddings[i]) != 0)
               print_diff(count++, i, embeddings_ref[i], embeddings[i]);

      free(Q_ref);
      free(K_ref);
      free(V_ref);
      free(KQ_ref);
      free(softmax_out_ref);
      free(QKV_ref);
      free(embeddings_ref);
      free(ATTNout_ref);
   }
#endif

   if(rank == root) {
      free(Qw);
      free(Kw);
      free(Vw);
   }

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

   MPI_Finalize();
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

static void gemm_impl(data_t * dst, const data_t * src1, const data_t * src2, int m, int n, int k, int stride_0, int stride_1, int stride_2) {
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
                     pp += src1[i0*(stride_1)+i2] * src2[i2*stride_2+i1];
                  }
                  dst[i0*(stride_0)+i1]+= pp;
               }
            }
         }
      }
   }
}

static void gemm_t_impl(data_t * dst, const data_t * src1, const data_t * src2, int m, int n, int k, int stride_0, int stride_1, int stride_2) {
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
                     pp += src1[i0*(stride_1)+i2] * src2[i1*stride_2+i2];
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
      max = FLT_MIN;
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


void gemm(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n, int k, int stride_0, int stride_1, int stride_2) {
   assert(dst);
   assert(src1);
   assert(src2);
   gemm_impl(((data_t*)dst), ((data_t*)src1), ((data_t*)src2), m, n, k, stride_0, stride_1, stride_2);
}

void gemm_t(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n, int k, int stride_0, int stride_1, int stride_2) {
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

#ifdef CHECK_RES

static void gemm_ref_impl(data_t * dst, const data_t * src1, const data_t * src2, int m, int n, int k, int stride_0, int stride_1, int stride_2) {
   int i0 = 0, i1 = 0, i2 = 0;
   for (i0 = 0; i0 < m; ++i0) {
      for (i1 = 0; i1 < n; ++i1) {
         for (i2 = 0; i2 < k; ++i2) {
            dst[i0*(stride_0)+i1] += src1[i0*(stride_1)+i2] * src2[i2*stride_2+i1];
         }
      }
   }
}

static void gemm_t_ref_impl(data_t * dst, const data_t * src1, const data_t * src2, int m, int n, int k, int stride_0, int stride_1, int stride_2) {
   int i0 = 0, i1 = 0, i2 = 0;
   for (i0 = 0; i0 < m; ++i0) {
      for (i1 = 0; i1 < n; ++i1) {
         for (i2 = 0; i2 < k; ++i2) {
            dst[i0*stride_0+i1] += src1[i0*(stride_1)+i2] * src2[i1*stride_2+i2];
         }
      }
   }
}

static void scale_ref_impl(data_t * dst, const data_t * src1, const data_t src2, int m, int n) {
   int i, j;
   for(i = 0; i < m*n; i++)
      dst[i] = src1[i] * src2;
}

static void add_ref_impl(data_t * dst, const data_t * src1, const data_t * src2, int m, int n) {
   int i, j;
   for(i = 0; i < m*n; i++)
      dst[i] = src1[i] + src2[i];
}

static void softmax_ref_impl(data_t * dst, const data_t * src, int m, int n) {
   int i, j;
   data_t max, sum;
   for(i = 0; i < m; i++) {
      max = FLT_MIN;
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


void gemm_ref(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n, int k, int stride_0, int stride_1, int stride_2) {
   assert(dst);
   assert(src1);
   assert(src2);
   gemm_ref_impl(((data_t*)dst), ((data_t*)src1), ((data_t*)src2), m, n, k, stride_0, stride_1, stride_2);
}

void gemm_t_ref(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n, int k, int stride_0, int stride_1, int stride_2) {
   assert(dst);
   assert(src1);
   assert(src2);
   gemm_t_ref_impl(((data_t*)dst), ((data_t*)src1), ((data_t*)src2), m, n, k, stride_0, stride_1, stride_2);
}

void scale_ref(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n) {
   assert(dst);
   assert(src1);
   assert(src2);
   scale_ref_impl(((data_t*)dst), ((data_t*)src1), *((data_t*)src2), m, n);
}

void add_ref(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n) {
   assert(dst);
   assert(src1);
   assert(src2);
   add_ref_impl(((data_t*)dst), ((data_t*)src1), ((data_t*)src2), m, n);
}


void softmax_ref(void * dst, void * src, data_type_e data_type, int m, int n) {
   assert(dst);
   assert(src);
   softmax_ref_impl(((data_t*)dst), ((data_t*)src), m, n);
}

#endif
