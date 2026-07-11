#include <Rcpp.h>
using namespace Rcpp;

#ifdef _OPENMP
#include <omp.h>
#endif


// ============================================================
// Helper structures and functions
// ============================================================

// [[Rcpp::export]]
void set_threads_cpp(int n_threads) {
#ifdef _OPENMP
    omp_set_num_threads(n_threads);
#endif
}


inline Rcpp::List bad_result()
{
    Rcpp::List result;
    result["logLik_unpenalized"] = -1e20;
    result["logLik_penalized"] = -1e20;
    result["penalty"] = NA_REAL;
    // TODO: add newer timings
    result["timings"] = Rcpp::List::create(
        Rcpp::Named("build_G") = 0.0,
        Rcpp::Named("groups") = 0.0,
        Rcpp::Named("finalize") = 0.0
    );
    return result;
}
