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

typedef enum {FP8, FP16, FP32, FP64} data_type_e;

void init_random_tensor(void * tensor, data_type_e data_type, size_t nmemb);

void gemm(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n, int k);

void gemm_t(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n, int k);

void scale(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n);

void softmax(void * dst, void * src, data_type_e data_type, int m, int n);

#ifdef CHECK_RES

void gemm_ref(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n, int k);

void gemm_t_ref(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n, int k);

void scale_ref(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n);

void softmax_ref(void * dst, void * src, data_type_e data_type, int m, int n);

static int cmpf(float rhs, float lhs) {
   int ret = 0;
   float diff = roundf(rhs) - roundf(lhs);
   ret = diff > 1.0 ? 1 : 0;
   return ret;
}

#endif

int main(int argc, char ** argv) {
   const int root = 0;
   int n_ranks, rank;
   MPI_Request emb_req, Qw_req, Kw_req, Vw_req, *Q_req, K_req, V_req, QKV_req;

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
   data_type_e data_type = FP32;

   if((dmodel%n_ranks) != 0) {
      fprintf(stderr, "Error: dmodel must be a multiple of the number of ranks (dmodel: %d, rank: %d)\n", dmodel, rank);
      exit(EXIT_FAILURE);
   }

   fprintf(stdout, "Model n_ranks: %d, Sequence lenght: %d, head count: %d\n", dmodel, S, h);


   MPI_Type_vector(dmodel, dmodel/n_ranks, dmodel, MPI_FLOAT, &col);
   MPI_Type_commit(&col);
   MPI_Type_create_resized(col, 0, dmodel/n_ranks*sizeof(float), &col_type);
   MPI_Type_commit(&col_type);


   Q_req = calloc(n_ranks, sizeof(MPI_Request));

   float * embeddings = NULL;
   float * Qw = NULL;
   float * Kw = NULL;
   float * Vw = NULL;

   float * Qw_col = NULL;
   float * Kw_col = NULL;
   float * Vw_col = NULL;

   float * Q_col = NULL;
   float * Q_row = NULL;

   float * K_col = NULL;
   float * K_full = NULL;

   float * V_col = NULL;
   float * V_full = NULL;

   float * KQ = NULL;
   float * softmax_out = NULL;

   float * QKV_row = NULL;
   float * QKV_full = NULL;

   float scale_f = 1.0f/sqrtf(((float)dk));

   srand(time(NULL));

   clock_gettime(CLOCK_MONOTONIC, &start);

   embeddings = calloc(dmodel*S, sizeof(float));

   if(rank == root) {
      init_random_tensor(embeddings, FP32, dmodel*S);
   }

   MPI_Ibcast(embeddings, dmodel*S, MPI_FLOAT, root, WORLD, &emb_req);

   if(rank == root) {
      Qw = calloc(dmodel*dmodel, sizeof(float));
      init_random_tensor(Qw, FP32, dmodel*dmodel);

      Kw = calloc(dmodel*dmodel, sizeof(float));
      init_random_tensor(Kw, FP32, dmodel*dmodel);

      Vw = calloc(dmodel*dmodel, sizeof(float));
      init_random_tensor(Vw, FP32, dmodel*dmodel);

      QKV_full = calloc(S*dmodel, sizeof(float));
   }

   Qw_col = calloc(dmodel*dmodel/n_ranks, sizeof(float));
   memset(Qw_col, 0, dmodel*dmodel/n_ranks*sizeof(float));
   Kw_col = calloc(dmodel*dmodel/n_ranks, sizeof(float));
   Vw_col = calloc(dmodel*dmodel/n_ranks, sizeof(float));

   MPI_Iscatter(Qw, 1, col_type, Qw_col, dmodel*dmodel/n_ranks, MPI_FLOAT, root, WORLD, &Qw_req);
   MPI_Iscatter(Kw, 1, col_type, Kw_col, dmodel*dmodel/n_ranks, MPI_FLOAT, root, WORLD, &Kw_req);
   MPI_Iscatter(Vw, 1, col_type, Vw_col, dmodel*dmodel/n_ranks, MPI_FLOAT, root, WORLD, &Vw_req);

   Q_col = calloc(S*dmodel/n_ranks, sizeof(float));
   memset(Q_col, 0, S*dmodel*sizeof(float)/n_ranks);
   Q_row = calloc(S*dmodel/n_ranks, sizeof(float));

   K_col = calloc(S*dmodel/n_ranks, sizeof(float));
   memset(K_col, 0, S*dmodel*sizeof(float)/n_ranks);
   K_full = calloc(S*dmodel, sizeof(float));

   V_col = calloc(S*dmodel/n_ranks, sizeof(float));
   memset(V_col, 0, S*dmodel*sizeof(float)/n_ranks);
   V_full = calloc(S*dmodel, sizeof(float));
   assert(V_full);

   KQ = calloc(S/n_ranks*S, sizeof(float));
   memset(KQ, 0, S/n_ranks*S*sizeof(float));

   softmax_out = calloc(S/n_ranks*S, sizeof(float));
   memset(softmax_out, 0, S/n_ranks*S*sizeof(float));

   QKV_row = calloc(S/n_ranks*dmodel, sizeof(float));
   memset(QKV_row, 0, S/n_ranks*dmodel*sizeof(float));

   MPI_Wait(&emb_req, MPI_STATUS_IGNORE);
   MPI_Wait(&Qw_req, MPI_STATUS_IGNORE);
   MPI_Wait(&Kw_req, MPI_STATUS_IGNORE);
   MPI_Wait(&Vw_req, MPI_STATUS_IGNORE);

   clock_gettime(CLOCK_MONOTONIC, &end);

   time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
   printf("[rank: %d] Init time: %.2f ms\n", rank, time_elapsed_s * 1000);

   clock_gettime(CLOCK_MONOTONIC, &start);
   /* MHA */

   gemm(Q_col, embeddings, Qw_col, FP32, S, dmodel/n_ranks, dmodel);

   for(int r = 0; r < n_ranks; r++) {
      const int bsize = S/n_ranks*dmodel/n_ranks;
      MPI_Iscatter(Q_col, bsize, MPI_FLOAT, &Q_row[bsize*r], bsize, MPI_FLOAT, r, WORLD, &Q_req[r]);
   }

   gemm(K_col, embeddings, Kw_col, FP32, S, dmodel/n_ranks, dmodel);

   MPI_Iallgather(K_col, S*dmodel/n_ranks, MPI_FLOAT, K_full, 1, col_type, WORLD, &K_req );

   gemm(V_col, embeddings, Vw_col, FP32, S, dmodel/n_ranks, dmodel);

   MPI_Iallgather(V_col, S*dmodel/n_ranks, MPI_FLOAT, V_full, 1, col_type, WORLD, &V_req );

   for(int r = 0; r < n_ranks; r++)
      MPI_Wait(&Q_req[r], MPI_STATUS_IGNORE);
   MPI_Wait(&K_req, MPI_STATUS_IGNORE);

   gemm_t(KQ, Q_row, K_full, FP32, S/n_ranks, S, dmodel);

   scale(KQ, KQ, ((void*)&scale_f), FP32, S/n_ranks, S);

   softmax(softmax_out, KQ, FP32, S/n_ranks, S);

   MPI_Wait(&V_req, MPI_STATUS_IGNORE);

   gemm(QKV_row, softmax_out, V_full, FP32, S/n_ranks, dmodel, S);

   MPI_Gather(QKV_row, S/n_ranks*dmodel, MPI_FLOAT, QKV_full, S/n_ranks*dmodel, MPI_FLOAT, root, WORLD);

   clock_gettime(CLOCK_MONOTONIC, &end);

   time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
   printf("[rank: %d] MHA execution time: %.2f ms\n", rank, time_elapsed_s * 1000);

#ifdef CHECK_RES
   if(rank == root) {
      float * QKV_ref = calloc(S*dmodel, sizeof(float));
      assert(QKV_ref);
      memset(QKV_ref, 0, S*dmodel*sizeof(float));

      float * Q_ref = calloc(S*dmodel, sizeof(float));
      assert(Q_ref);
      memset(Q_ref, 0, S*dmodel*sizeof(float));

      float * K_ref = calloc(S*dmodel, sizeof(float));
      assert(K_ref);
      memset(K_ref, 0, S*dmodel*sizeof(float));

      float * V_ref = calloc(S*dmodel, sizeof(float));
      assert(V_ref);
      memset(V_ref, 0, S*dmodel*sizeof(float));

      float * KQ_ref = calloc(S*S, sizeof(float));
      assert(KQ_ref);
      memset(KQ_ref, 0, S*S*sizeof(float));

      float * softmax_out_ref = calloc(S*S, sizeof(float));
      assert(softmax_out_ref);
      memset(softmax_out_ref, 0, S*S*sizeof(float));

      gemm_ref(Q_ref, embeddings, Qw, FP32, S, dmodel, dmodel);
      gemm_ref(K_ref, embeddings, Kw, FP32, S, dmodel, dmodel);
      gemm_ref(V_ref, embeddings, Vw, FP32, S, dmodel, dmodel);

      gemm_t_ref(KQ_ref, Q_ref, K_ref, FP32, S, dmodel, S);

      scale_ref(KQ_ref, KQ_ref, ((void*)&scale_f), FP32, S, S);

      softmax_ref(softmax_out_ref, KQ_ref, FP32, S, S);

      gemm_ref(QKV_ref, softmax_out_ref, V_ref, FP32, S, dmodel, S);

      int cmp = 0;
      for(int i = 0; i < S; i++)
         for(int j = 0; j < dmodel; j++)
            if(cmpf(QKV_ref[i*S+j], QKV_full[i*S+j]) != 0)
               printf("Difference %d at %d ref = %.4f mpi = %.4f\n", cmp++, i*S+j, QKV_ref[i*S+j], QKV_full[i*S+j]);

      free(Q_ref);
      free(K_ref);
      free(V_ref);
      free(KQ_ref);
      free(softmax_out_ref);
      free(QKV_ref);
   }
#endif

   if(rank == root) {
      free(Qw);
      free(Kw);
      free(Vw);
      free(QKV_full);

   }

   free(Q_req);

   free(embeddings);
   free(Qw_col);
   free(Kw_col);
   free(Vw_col);
   free(Q_col);
   free(Q_row);
   free(K_col);
   free(K_full);
   free(V_col);
   free(V_full);
   free(KQ);
   free(softmax_out);
   free(QKV_row);

   MPI_Finalize();
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

#ifdef CHECK_RES

static void gemm_fp32_ref(float * dst, const float * src1, const float * src2, int m, int n, int k) {
   int i0 = 0, i1 = 0, i2 = 0;
   for (i0 = 0; i0 < m; ++i0) {
      for (i1 = 0; i1 < n; ++i1) {
         for (i2 = 0; i2 < k; ++i2) {
            dst[i0*n+i1] += src1[i0*k+i2] * src2[i2*n+i1];
         }
      }
   }
}

static void gemm_t_fp32_ref(float * dst, const float * src1, const float * src2, int m, int n, int k) {
   int i0 = 0, i1 = 0, i2 = 0;
   for (i0 = 0; i0 < m; ++i0) {
      for (i1 = 0; i1 < n; ++i1) {
         for (i2 = 0; i2 < k; ++i2) {
            dst[i0*n+i1] += src1[i0*k+i2] * src2[i1*k+i2];
         }
      }
   }
}

static void scale_fp32_ref(float * dst, const float * src1, const float src2, int m, int n) {
   int i, j;
   for(i = 0; i < m*n; i++)
      dst[i] = src1[i] * src2;
}

static void softmax_fp32_ref(float * dst, const float * src, int m, int n) {
   int i, j;
   float max, sum;
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


void gemm_ref(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n, int k) {
   assert(dst);
   assert(src1);
   assert(src2);
   switch(data_type) {
      case FP32:
         gemm_fp32_ref(((float*)dst), ((float*)src1), ((float*)src2), m, n, k);
         break;
      default:
         fprintf(stderr, "[%s:%d] Data type not supported\n", __func__, __LINE__);
         break;
   }
}

void gemm_t_ref(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n, int k) {
   assert(dst);
   assert(src1);
   assert(src2);
   switch(data_type) {
      case FP32:
         gemm_t_fp32_ref(((float*)dst), ((float*)src1), ((float*)src2), m, n, k);
         break;
      default:
         fprintf(stderr, "[%s:%d] Data type not supported\n", __func__, __LINE__);
         break;
   }
}

void scale_ref(void * dst, void * src1, void * src2, data_type_e data_type, int m, int n) {
   assert(dst);
   assert(src1);
   assert(src2);
   switch(data_type) {
      case FP32:
         scale_fp32_ref(((float*)dst), ((float*)src1), *((float*)src2), m, n);
         break;
      default:
         fprintf(stderr, "[%s:%d] Data type not supported\n", __func__, __LINE__);
         break;
   }
}

void softmax_ref(void * dst, void * src, data_type_e data_type, int m, int n) {
   assert(dst);
   assert(src);
   switch(data_type) {
      case FP32:
         softmax_fp32_ref(((float*)dst), ((float*)src), m, n);
         break;
      default:
         fprintf(stderr, "[%s:%d] Data type not supported\n", __func__, __LINE__);
         break;
   }
}

#endif
