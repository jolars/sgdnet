// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// SgdnetDense
Rcpp::List SgdnetDense(const Eigen::MatrixXd& x, const std::vector<double>& y, const Rcpp::List& control);
RcppExport SEXP _sgdnet_SgdnetDense(SEXP xSEXP, SEXP ySEXP, SEXP controlSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const Rcpp::List& >::type control(controlSEXP);
    rcpp_result_gen = Rcpp::wrap(SgdnetDense(x, y, control));
    return rcpp_result_gen;
END_RCPP
}
// SgdnetSparse
Rcpp::List SgdnetSparse(const Eigen::SparseMatrix<double>& x, const std::vector<double>& y, const Rcpp::List& control);
RcppExport SEXP _sgdnet_SgdnetSparse(SEXP xSEXP, SEXP ySEXP, SEXP controlSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::SparseMatrix<double>& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const Rcpp::List& >::type control(controlSEXP);
    rcpp_result_gen = Rcpp::wrap(SgdnetSparse(x, y, control));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_sgdnet_SgdnetDense", (DL_FUNC) &_sgdnet_SgdnetDense, 3},
    {"_sgdnet_SgdnetSparse", (DL_FUNC) &_sgdnet_SgdnetSparse, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_sgdnet(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
