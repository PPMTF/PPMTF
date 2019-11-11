#ifndef INCLUDED_PPXTF_HPP
#define INCLUDED_PPXTF_HPP

#include <utility>
#include <vector>
#include <map>
#include <tuple>
#include <string>

// eigen
#include <Eigen/Core>

using p_t = std::pair<int, int>;
using r_t = std::vector<std::map<p_t, double> >;
using mat_t = Eigen::MatrixXd;
using vec_mat_t = std::vector<mat_t>;

std::tuple<mat_t, mat_t> sample_Lam_X_mu_X(const mat_t& X, int N, int nu0, const mat_t& W0, double beta0, double mu0, int K);
std::tuple<mat_t, mat_t> calc_XYXY_XYR(int K, const mat_t& B, const mat_t& C, const std::map<p_t, double>& rt1);
mat_t multivariate_normal(const mat_t& mu, const mat_t& Sigma);
void save_parameters(int K, int ItrNum, const std::string& prefix, const std::string& name, const vec_mat_t& A, const mat_t& mu_A, const mat_t& Lam_A, const std::string& X = "");
void seeding(unsigned seed = 0);
std::tuple<r_t, r_t, r_t> ReadTrainTransTensor(int N, int M, int ZeroNum, const std::string& infile);
std::tuple<r_t, r_t, r_t> ReadTrainVisitTensor(int N, int M, int T, int ZeroNum, const std::string& infile);
mat_t random_mat(int m, int n);
void read_tensor(r_t& r1, r_t& r2, r_t& r3, const std::string& infile);
std::vector<int> random_index(int n);

#endif // INCLUDED_PPXTF_HPP
