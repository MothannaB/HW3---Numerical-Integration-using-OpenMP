// omp_integrate.c — Starter for OpenMP Numerical Integration
// Implements composite trapezoid & Simpson rules with OpenMP reductions,
// CLI flags per assignment, schedule(runtime) + omp_set_schedule, timing,
// and JSON output with best and median timings.
//
// Build:
//   gcc -O3 -march=native -fopenmp -o omp_integrate omp_integrate.c
//
// Example run (Simpson, sin, 2^24 panels, 8 threads):
//   ./omp_integrate --rule simp --func sin --a 0 --b 3.141592653589793 \
//                   --n 16777216 --threads 8 --work 1000 \
//                   --schedule static,1 --repeat 5 --warmups 2
//
// Notes:
//  - Simpson requires even n.
//  - Use --work K to increase compute intensity (compute-bound vs memory-bound).
//  - Use --schedule static[,chunk] | dynamic[,chunk] | guided[,chunk].
//  - Set OMP_PROC_BIND / OMP_PLACES in your environment for affinity experiments.
//
// This file is adapted to match the “OpenMP & Numerical Integration” assignment
// you provided (flags, goals, measurements).  (Ref: your assignment PDF)
//

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include <stdalign.h>

#ifndef CACHELINE
#define CACHELINE 64
#endif

typedef struct {
    alignas(CACHELINE) double v;
} padded_double;

// Helper to allocate an array of padded slots = number of threads
static padded_double* alloc_padded(int threads) {
    padded_double *arr = (padded_double*) aligned_alloc(CACHELINE, threads * sizeof(padded_double));
    if (!arr) {
        fprintf(stderr, "[error] alloc_padded failed\n");
        exit(1);
    }
    for (int i = 0; i < threads; ++i) arr[i].v = 0.0;
    return arr;
}



typedef enum { RULE_TRAP=0, RULE_SIMP=1 } rule_t;
typedef enum { F_SIN=0, F_EXPQUAD=1 } func_t;
typedef enum { ACCUM_REDUCTION=0, ACCUM_PADDED=1 } accum_t;

typedef struct {
    rule_t rule;
    func_t func;
    double a, b;
    long long n;
    int threads;
    int workK;
    int repeat;
    int warmups;
    // schedule settings parsed from --schedule:
    omp_sched_t sched_kind;
    int sched_chunk;
    accum_t accum; // NEW

} args_t;






// ---- Utility: median (for timing) ----
static int cmp_double_asc(const void *p1, const void *p2) {
    const double a = *(const double*)p1, b = *(const double*)p2;
    return (a > b) - (a < b);
}
static double median_double(double *x, int m) {
    qsort(x, (size_t)m, sizeof(double), cmp_double_asc);
    if (m <= 0) return NAN;
    if (m & 1) return x[m/2];
    return 0.5*(x[m/2 - 1] + x[m/2]);
}

// ---- Work amplifier to make f(x) compute-heavy when K>0 ----
static inline double heavy(double x, int K) {
    // Start from sin(x) so “sin on [0,pi]” remains the baseline integrand
    // while allowing extra arithmetic intensity via K. (Matches assignment intent.)
    double y = sin(x);
    // Constants chosen to avoid trivial compiler elimination while staying stable.
    const double c1 = 1.00000011920928955078125;    // ~1 + 2^-23
    const double c2 = 0.0000002384185791015625;     // ~2^-22
    const double c3 = 0.00000011920928955078125;    // ~2^-23
    for (int k = 0; k < K; ++k) {
        y = y * c1 + c2;
        y = y - c3 * y * y;
    }
    return y;
}

// ---- Integrand selector ----
static inline double f_eval(double x, func_t f, int K) {
    switch (f) {
        case F_SIN:     return heavy(x, K);        // sin-based, compute-heavy via K
        case F_EXPQUAD: return exp(-x*x);          // optional second function
        default:        return sin(x);
    }
}

// ---- Schedule parser: sets omp_set_schedule + returns kind/chunk ----
static void parse_schedule(const char *s, omp_sched_t *kind, int *chunk) {
    // Accept "static", "static,4", "dynamic,8", "guided,32"
    // Fallback: static, chunk=1
    char buf[64]; buf[0] = '\0';
    if (s) {
        strncpy(buf, s, sizeof(buf)-1);
        buf[sizeof(buf)-1] = '\0';
    }
    char *comma = strchr(buf, ',');
    if (comma) *comma = '\0';
    const char *kstr = (buf[0] ? buf : "static");
    const char *cstr = (comma ? (comma+1) : NULL);

    if      (!strcmp(kstr, "static"))  *kind = omp_sched_static;
    else if (!strcmp(kstr, "dynamic")) *kind = omp_sched_dynamic;
    else if (!strcmp(kstr, "guided"))  *kind = omp_sched_guided;
    else                               *kind = omp_sched_static;

    *chunk = (cstr ? atoi(cstr) : 1);
    if (*chunk <= 0) *chunk = 1;

    // Apply runtime schedule so loops can use schedule(runtime)
    omp_set_schedule(*kind, *chunk);
}

// ---- Integration kernels ----
static double integrate_trap(double a, double b, long long n, func_t f, int K) {
    const double h = (b - a) / (double)n;
    double s = 0.0;

    #pragma omp parallel for reduction(+:s) schedule(runtime)
    for (long long i = 1; i < n; ++i) {
        const double x = a + i * h;
        s += f_eval(x, f, K);
    }

    // endpoints with 1/2 weights
    const double res = h * (0.5 * f_eval(a, f, K) + s + 0.5 * f_eval(b, f, K));
    return res;
}

// static double integrate_trap_padded(double a, double b, long long n, func_t f, int K) {
//     const double h = (b - a) / (double)n;

//     padded_double *priv = alloc_padded(threads);

//     #pragma omp parallel
//     {
//         int tid = omp_get_thread_num();
//         double local = 0.0;

//         #pragma omp for schedule(runtime)
//         for (long long i = 1; i < n; ++i) {
//             const double x = a + i * h;
//             local += f_eval(x, f, K);
//         }
//         priv[tid].v = local;
//     }

//     // combine
//     double s = 0.0;
//     for (int i = 0; i < threads; ++i) s += priv[i].v;

//     free(priv);

//     const double res = h * (0.5 * f_eval(a, f, K) + s + 0.5 * f_eval(b, f, K));
//     return res;
// }



static double integrate_simp(double a, double b, long long n, func_t f, int K) {
    // Requires even n
    const double h = (b - a) / (double)n;
    double s_odd = 0.0, s_even = 0.0;

    // Sum odd indices
    #pragma omp parallel for reduction(+:s_odd) schedule(runtime)
    for (long long i = 1; i < n; i += 2) {
        const double x = a + i * h;
        s_odd += f_eval(x, f, K);
    }
    // Sum even indices
    #pragma omp parallel for reduction(+:s_even) schedule(runtime)
    for (long long i = 2; i < n; i += 2) {
        const double x = a + i * h;
        s_even += f_eval(x, f, K);
    }

    const double res = (h / 3.0) * (f_eval(a, f, K) + 4.0 * s_odd + 2.0 * s_even + f_eval(b, f, K));
    return res;
}


//start


static double integrate_trap_padded(double a, double b, long long n, func_t f, int K)
{
    const double h = (b - a) / (double)n;
    const int P = omp_get_max_threads();   // <- number of OpenMP threads to size the padded array
    padded_double *priv = alloc_padded(P);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double local = 0.0;

        #pragma omp for schedule(runtime)
        for (long long i = 1; i < n; ++i) {
            const double x = a + i * h;
            local += f_eval(x, f, K);
        }
        priv[tid].v = local;
    }

    double s = 0.0;
    for (int i = 0; i < P; ++i) s += priv[i].v;
    free(priv);

    return h * (0.5 * f_eval(a, f, K) + s + 0.5 * f_eval(b, f, K));
}

static double integrate_simp_padded(double a, double b, long long n, func_t f, int K)
{
    const double h = (b - a) / (double)n;
    const int P = omp_get_max_threads();
    padded_double *privOdd  = alloc_padded(P);
    padded_double *privEven = alloc_padded(P);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double locOdd  = 0.0, locEven = 0.0;

        #pragma omp for schedule(runtime) nowait
        for (long long i = 1; i < n; i += 2) {
            const double x = a + i * h;
            locOdd += f_eval(x, f, K);
        }

        #pragma omp for schedule(runtime)
        for (long long i = 2; i < n; i += 2) {
            const double x = a + i * h;
            locEven += f_eval(x, f, K);
        }

        privOdd[tid].v  = locOdd;
        privEven[tid].v = locEven;
    }

    double s_odd = 0.0, s_even = 0.0;
    for (int i = 0; i < P; ++i) { s_odd += privOdd[i].v; s_even += privEven[i].v; }
    free(privOdd);
    free(privEven);

    return (h/3.0) * (f_eval(a, f, K) + 4.0*s_odd + 2.0*s_even + f_eval(b, f, K));
}



//end

// static double integrate_simp_padded(double a, double b, long long n, func_t f, int K)
// {
//     // Simpson requires even n (your parse_args already enforces this)
//     const double h = (b - a) / (double)n;
//     const int P = omp_get_max_threads();

//     padded_double *privOdd  = alloc_padded(P);
//     padded_double *privEven = alloc_padded(P);

//     #pragma omp parallel
//     {
//         const int tid = omp_get_thread_num();
//         double locOdd  = 0.0;
//         double locEven = 0.0;

//         // odd indices
//         #pragma omp for schedule(runtime) nowait
//         for (long long i = 1; i < n; i += 2) {
//             const double x = a + i * h;
//             locOdd += f_eval(x, f, K);
//         }
//         // even indices
//         #pragma omp for schedule(runtime)
//         for (long long i = 2; i < n; i += 2) {
//             const double x = a + i * h;
//             locEven += f_eval(x, f, K);
//         }

//         // each thread writes to its own cache-line
//         privOdd[tid].v  = locOdd;
//         privEven[tid].v = locEven;
//     }

//     // sequential combine
//     double s_odd = 0.0, s_even = 0.0;
//     for (int i = 0; i < P; ++i) {
//         s_odd  += privOdd[i].v;
//         s_even += privEven[i].v;
//     }

//     free(privOdd);
//     free(privEven);

//     const double res = (h/3.0) * (f_eval(a, f, K) + 4.0*s_odd + 2.0*s_even + f_eval(b, f, K));
//     return res;
// }








// ---- CLI parsing & usage ----
static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --rule trap|simp --func sin|expquad --a A --b B --n N \\\n"
        "          --threads P --work K --repeat R --warmups W --schedule kind[,chunk]\n"
        "Defaults: --rule simp --func sin --a 0 --b pi --n 2^24 --threads 1 --work 0 --repeat 5 --warmups 1 --schedule static,1\n", prog);
}

static int parse_args(int argc, char **argv, args_t *A) {
    // defaults (aligned with assignment’s examples)
    A->rule = RULE_SIMP;
    A->func = F_SIN;
    A->a = 0.0;
    A->b = M_PI;
    A->n = (1LL<<24);
    A->threads = 1;
    A->workK = 0;
    A->repeat = 5;
    A->warmups = 1;
    A->sched_kind = omp_sched_static;
    A->sched_chunk = 1;
    A->accum = ACCUM_REDUCTION;   // default

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i],"--rule") && i+1<argc) {
            ++i;
            if (!strcmp(argv[i],"trap")) A->rule = RULE_TRAP;
            else                         A->rule = RULE_SIMP;
        } else if (!strcmp(argv[i],"--func") && i+1<argc) {
            ++i;
            if (!strcmp(argv[i],"sin"))      A->func = F_SIN;
            else if (!strcmp(argv[i],"expquad")) A->func = F_EXPQUAD;
            else A->func = F_SIN;
        } else if (!strcmp(argv[i],"--a") && i+1<argc) {
            A->a = atof(argv[++i]);
        } else if (!strcmp(argv[i],"--b") && i+1<argc) {
            A->b = atof(argv[++i]);
        } else if (!strcmp(argv[i],"--n") && i+1<argc) {
            A->n = atoll(argv[++i]);
        } else if (!strcmp(argv[i],"--threads") && i+1<argc) {
            A->threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i],"--work") && i+1<argc) {
            A->workK = atoi(argv[++i]);
        } else if (!strcmp(argv[i],"--repeat") && i+1<argc) {
            A->repeat = atoi(argv[++i]);
        } else if (!strcmp(argv[i],"--warmups") && i+1<argc) {
            A->warmups = atoi(argv[++i]);
        } else if (!strcmp(argv[i],"--schedule") && i+1<argc) {
            const char *s = argv[++i];
            parse_schedule(s, &A->sched_kind, &A->sched_chunk); // sets runtime schedule
        } else if (!strcmp(argv[i],"--accum") && i+1 < argc) {
            ++i;
            if (!strcmp(argv[i], "padded"))
                A->accum = ACCUM_PADDED;
            else
                A->accum = ACCUM_REDUCTION;
        }

        
        
        else {
            fprintf(stderr, "Unknown or bad arg: %s\n", argv[i]);
            usage(argv[0]);
            return 0;
        }
    }
    if (A->rule == RULE_SIMP && (A->n % 2LL)) {
        fprintf(stderr, "[error] Simpson requires even n (got n=%lld)\n", A->n);
        return 0;
    }


    return 1;
}

int main(int argc, char **argv) {
    args_t A;
    if (!parse_args(argc, argv, &A)) return 1;

    // Apply threads setting
    if (A.threads > 0) omp_set_num_threads(A.threads);

    const int runs = A.warmups + A.repeat;
    double best = 1e300;
    double *times = (double*)malloc(sizeof(double) * (A.repeat > 0 ? A.repeat : 1));
    int tcount = 0;

    double keep_res = 0.0;

    for (int r = 0; r < runs; ++r) {
        double t0 = omp_get_wtime();
        double res = 0.0;
        // if (A.rule == RULE_TRAP) {
        //     res = integrate_trap(A.a, A.b, A.n, A.func, A.workK);
        // } else {
        //     res = integrate_simp(A.a, A.b, A.n, A.func, A.workK);
        // }

        

        if (A.rule == RULE_TRAP) {
            if (A.accum == ACCUM_PADDED)
                res = integrate_trap_padded(A.a, A.b, A.n, A.func, A.workK);
            else
                res = integrate_trap(A.a, A.b, A.n, A.func, A.workK);      // your reduction version
        } else { // RULE_SIMP
            if (A.accum == ACCUM_PADDED)
                res = integrate_simp_padded(A.a, A.b, A.n, A.func, A.workK);
            else
                res = integrate_simp(A.a, A.b, A.n, A.func, A.workK);      // your reduction version
        }

        double t1 = omp_get_wtime();
        double dt = t1 - t0;

        if (r >= A.warmups) {
            if (dt < best) { best = dt; keep_res = res; }
            if (tcount < A.repeat) times[tcount++] = dt;
        }
    }

    double med = (tcount > 0 ? median_double(times, tcount) : NAN);
    free(times);

    // Print JSON: include both best_s and median_s to satisfy different experiment choices
    // printf("{\"rule\":\"%s\",\"func\":\"%s\",\"a\":%.17g,\"b\":%.17g,"
    //        "\"n\":%lld,\"threads\":%d,\"workK\":%d,"
    //        "\"best_s\":%.6f,\"median_s\":%.6f,\"result\":%.17g}\n",
    //        (A.rule==RULE_TRAP?"trap":"simp"),
    //        (A.func==F_SIN?"sin":"expquad"),
    //        A.a, A.b, A.n, A.threads, A.workK, best, med, keep_res);


    printf("{\"rule\":\"%s\",\"func\":\"%s\",\"a\":%.17g,\"b\":%.17g,"
       "\"n\":%lld,\"threads\":%d,\"workK\":%d,"
       "\"accum\":\"%s\",\"best_s\":%.6f,\"median_s\":%.6f,\"result\":%.17g}\n",
       (A.rule==RULE_TRAP?"trap":"simp"),
       (A.func==F_SIN?"sin":"expquad"),
       A.a, A.b, A.n, A.threads, A.workK,
       (A.accum==ACCUM_PADDED?"padded":"reduction"),
       best, med, keep_res);

    return 0;
}