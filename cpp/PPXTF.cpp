#include "PPXTF.hpp"

#include "common.hpp"
#include <cstdio>
#include <string>
#include <map>
#include <fstream>
#include <utility>
#include <vector>
#include <algorithm>
#include <random>
#include <tuple>
#include <utility>
#include <cassert>
#include <iostream>
#include <cstdlib>

// eigen
#include <Eigen/Core>

// stats
#define STATS_DONT_USE_OPENMP
#define STATS_ENABLE_EIGEN_WRAPPERS
//#define STATS_ENABLE_STDVEC_WRAPPERS
//#define STATS_ENABLE_INTERNAL_VEC_FEATURES
#include <stats.hpp>

using namespace Eigen;
using namespace std;

constexpr int BUF_SIZE = 1000;

using p_t = pair<int, int>;
using r_t = vector<map<p_t, double> >;
using mat_t = MatrixXd;
using vec_mat_t = vector<mat_t>;

std::mt19937_64 engine;

const map<int, double> Count2Rating = {
  {1,1.0},
  {2,2.0},
  {3,3.0},
  {4,4.0},
  {5,5.0},
  {6,6.0},
  {7,7.0},
  {8,8.0},
  {9,9.0},
  {10,10.0}
};

vector<int> seq(int n){
  vector<int> v(n);
  for(int i = 0; i < n; ++i){v[i] = i;}
  return v;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// random

vector<int> random_index(int n){
  vector<int> res = seq(n);
  shuffle(res.begin(), res.end(), engine);
  return res;
}

// m \times n matrix
// numbers are uniformly random and in the [0,1] range
mat_t random_mat(int m, int n){
  return 0.5 * (mat_t::Random(m, n).array() + 1);
}

// Cholesky decomposition
inline mat_t chol(const mat_t& A){
  return A.llt().matrixL();
}

// https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
// https://qiita.com/jhako/items/30f420033b5c126eedc1
// mu[K,1]
// Sigma[K,K]
// ret[K,1]
mat_t multivariate_normal(const mat_t& mu, const mat_t& Sigma){
  int K = Sigma.rows();
  assert(mu.rows() == K);
  const mat_t L = chol(Sigma); // L * L' = Sigma
  mat_t r = mat_t::Zero(K, 1);
  for(int i = 0; i < K; ++i){r(i, 0) = stats::rnorm(0, 1, engine);}
  return mu + L * r;
}

// https://www.math.wustl.edu/~sawyer/hmhandouts/Wishart.pdf
// scale[K, K]
// res[K, K]
mat_t wishart_rvs(int df, const mat_t& scale){
  int K = scale.rows();
  mat_t A = chol(scale); // A * A' = scale
  mat_t T = mat_t::Zero(K, K); // T * T' = B; B ~ W(I_d, d, n)
  for(int i = 1; i < K; ++i){ // T_{i,j} ~ N(0,1) for j < i
    for(int j = 0; j < i; ++j){
      T(i, j) = stats::rnorm(0 ,1, engine);
    }
  }
  for(int i = 0; i < K; ++i){ // T_{i,i} ~ \chi^2(df-i)
    T(i, i) = std::sqrt(stats::rchisq(df - i, engine));
  }
  mat_t AT = A * T;
  return AT * AT.transpose(); // A * B * A' ~ W(scale, d, n); AT * AT' = A * T * T' * A' = A * B * A'
}

void seeding(unsigned seed){
  //std::random_device seed_gen;
  //engine = std::mt19937_64(seed_gen());
  engine = std::mt19937_64(seed);
  srand(seed); // for eigen
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double count2rating(int count){
  auto it = Count2Rating.lower_bound(count);
  if(it == Count2Rating.end()){
    return Count2Rating.rbegin()->second;
  }else{
    return it->second;
  }
}

inline void write_to_tensor(r_t& r1, r_t& r2, r_t& r3,
			    int i, int j, int k,
			    double value){
  r1[i][p_t(j,k)] = value;
  r2[j][p_t(i,k)] = value;
  r3[k][p_t(i,j)] = value;
}

void read_tensor(r_t& r1, r_t& r2, r_t& r3, const string& infile){
  ifstream ifs(infile);
  string line;
  getline(ifs, line); // header
  while(getline(ifs, line)){
    auto row = split(line);
    assert(row.size() >= 4);
    int i = atoi(row[0].c_str());
    int j = atoi(row[1].c_str());
    int k = atoi(row[2].c_str());
    double value = count2rating(atoi(row[3].c_str()));
    write_to_tensor(r1, r2, r3, i, j, k, value);
  }
}

/*
  ###################### Read a training transition tensor ######################
  # [input1]: N -- Number of users
  # [input2]: M -- Number of POIs
  # [output1]: RT1_observed ([user_index]{poi_index_from, poi_index_to: value})
  # [output2]: RT2_observed ([poi_index_from]{user_index, poi_index_to: value})
  # [output3]: RT3_observed ([poi_index_to]{user_index, poi_index_from: value})
 */
tuple<r_t, r_t, r_t> ReadTrainTransTensor(int N, int M, int ZeroNum, const string& infile){
  r_t RT1_observed(N), RT2_observed(M), RT3_observed(M);
  // Read a training tensor --> (RT1,RT2,RT3)_observed
  read_tensor(RT1_observed, RT2_observed, RT3_observed, infile);
  // Randomly assign ZeroNum zero-values --> (RT1,RT2,RT3)_observed
  if(ZeroNum > 0){
    int MM = M * M;
    for(int user_index = 0; user_index < N; ++user_index){
      auto rand_index = random_index(MM);
      int zero_num = 0;
      for(int i = 0; i < MM; ++i){
	int poi_index_from = rand_index[i] % M;
	int poi_index_to = rand_index[i] / M;
	auto& a = RT1_observed[user_index];
	if(a.find(p_t(poi_index_from,poi_index_to)) == a.end()){
	  write_to_tensor(RT1_observed, RT2_observed, RT3_observed, user_index, poi_index_from, poi_index_to, 0);
	  zero_num += 1;
	  if(zero_num == ZeroNum){break;}
	}
      }
    }
  }else if(ZeroNum == -1){
    for(int user_index = 0; user_index < N; ++user_index){
      for(int poi_index_from = 0; poi_index_from < M; ++poi_index_from){
	for(int poi_index_to = 0; poi_index_to < M; ++poi_index_to){
	  auto& a = RT1_observed[user_index];
	  auto p = p_t(poi_index_from,poi_index_to);
	  if(a.find(p) == a.end()){
	    write_to_tensor(RT1_observed, RT2_observed, RT3_observed, user_index, poi_index_from, poi_index_to, 0);
	  }
	}
      }
    }
  }
  return forward_as_tuple(RT1_observed, RT2_observed, RT3_observed);
}
/*
  ######################## Read a training visit tensor #########################
  # [input1]: N -- Number of users
  # [input2]: M -- Number of POIs
  # [input3]: T (Number of time periods)
  # [output1]: RV1_observed ([user_index]{poi_index_from, time_id: value})
  # [output2]: RV2_observed ([poi_index_from]{user_index, time_id: value})
  # [output3]: RV3_observed ([time_id]{user_index, poi_index_from: value})
 */
tuple<r_t, r_t, r_t> ReadTrainVisitTensor(int N, int M, int T, int ZeroNum, const string& infile){
  r_t RV1_observed(N), RV2_observed(M), RV3_observed(T);
  // Read a training tensor --> (RV1,RV2,RV3)_observed
  read_tensor(RV1_observed, RV2_observed, RV3_observed, infile);
  // Randomly assign ZeroNum zero-values --> (RV1,RV2,RV3)_observed
  if(ZeroNum > 0){
    int MT = M * T;
    for(int user_index = 0; user_index < N; ++user_index){
      auto rand_index = random_index(MT);
      int zero_num = 0;
      for(int i = 0; i < MT; ++i){
	int poi_index_from = rand_index[i] % M;
	int time_id = rand_index[i] / M;
	auto& a = RV1_observed[user_index];
	if(a.find(p_t(poi_index_from,time_id)) == a.end()){
	  write_to_tensor(RV1_observed, RV2_observed, RV3_observed, user_index, poi_index_from, time_id, 0);
	  zero_num += 1;
	  if(zero_num == ZeroNum){break;}
	}
      }
    }
  }else if(ZeroNum == -1){
    for(int user_index = 0; user_index < N; ++user_index){
      for(int poi_index_from = 0; poi_index_from < M; ++poi_index_from){
	for(int time_id = 0; time_id < T; ++time_id){
	  auto& a = RV1_observed[user_index];
	  auto p = p_t(poi_index_from,time_id);
	  if(a.find(p) == a.end()){
	    write_to_tensor(RV1_observed, RV2_observed, RV3_observed, user_index, poi_index_from, time_id, 0);
	  }
	}
      }
    }
  }
  return forward_as_tuple(RV1_observed, RV2_observed, RV3_observed);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// io

void puts_mat(FILE* fout, const mat_t& A){
  int m = A.rows();
  int n = A.cols();
  assert(n > 0);
  for(int i = 0; i < m; ++i){
    fprintf(fout, "%lf", A(i,0));
    for(int j = 1; j < n; ++j){
      fprintf(fout, ",%lf", A(i,j));
    }
    fprintf(fout, "\n");
  }
}

void savetxt(const string& outfile, const mat_t& A){
  FILE* fp = fopen(outfile.c_str(), "w");
  puts_mat(fp, A);
  fclose(fp);
}

void save_parameters(int K, int ItrNum, const string& prefix, const string& name, const vec_mat_t& A, const mat_t& mu_A, const mat_t& Lam_A, const string& X){
  char buf[BUF_SIZE];
  sprintf(buf, "%s_K%d_Itr%d_%s%s.csv", prefix.c_str(), K, ItrNum, name.c_str(), X.c_str());
  savetxt(buf, A[ItrNum]);
  sprintf(buf, "%s_K%d_Itr%d_mu_%s%s.csv", prefix.c_str(), K, ItrNum, name.c_str(), X.c_str());
  savetxt(buf, mu_A);
  sprintf(buf, "%s_K%d_Itr%d_Lam_%s%s.csv", prefix.c_str(), K, ItrNum, name.c_str(), X.c_str());
  savetxt(buf, Lam_A);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ppmft

tuple<mat_t, mat_t> sample_Lam_X_mu_X(const mat_t& X, int N, int nu0, const mat_t& W0, double beta0, double mu0, int K){
  // Sample Lam_X
  // S_bar = np.sum([np.outer(A[itr, n, :], A[itr, n, :]) for n in range(N)], axis=0) / N
  mat_t S_bar = X.transpose() * X / N; // [K, K]
  // a_bar = np.sum(A[itr], axis=0) / N
  mat_t x_bar = X.colwise().sum() / N; // [1, K]
  // nu_ast = nu0 + n
  int nu_ast = nu0 + N; // 1
  // W0_ast = inv(inv(W0) + N * S_bar + (beta0 * N / (beta0 + N)) * np.outer(mu0 - a_bar, mu0 - a_bar))
  mat_t t = mat_t::Constant(1, K, mu0) - x_bar; // [1, K]
  mat_t W0_ast = (W0.inverse() + N * S_bar + (beta0 * N / (beta0 + N)) * (t.transpose() * t)).inverse(); // [K, K]
  // Lam_A = wishart.rvs(df=nu_ast, scale=W0_ast)
  //mat_t Lam_X = stats::rwish(W0_ast, nu_ast); // [K, K]
  mat_t Lam_X = wishart_rvs(nu_ast, W0_ast); // [K, K]

  // Sample mu_X
  // mu0_ast = (beta0 * mu0 + N * a_bar) / (beta0 + N)
  mat_t mu0_ast = ((mat_t::Constant(1, K, beta0 * mu0) + N * x_bar) / (beta0 + N)).transpose(); // [K, 1]
  // mu_A = multivariate_normal(mu0_ast, inv((beta0 + N) * Lam_A))
  mat_t mu_X = multivariate_normal(mu0_ast, ((beta0 + N) * Lam_X).inverse()); // [K, 1]
  return forward_as_tuple(Lam_X, mu_X);
}

tuple<mat_t, mat_t> calc_XYXY_XYR(int K, const mat_t& B, const mat_t& C, const map<p_t, double>& rt1){
  mat_t BC_BC = mat_t::Zero(K, K);
  mat_t BC_R = mat_t::Zero(1, K);
  for(auto kv : rt1){
    int i = kv.first.first, j = kv.first.second;
    double value = kv.second;
    mat_t bc_ij = B.row(i).array() * C.row(j).array(); // hadamard product
    mat_t bc_bc = bc_ij.transpose() * bc_ij; // [K,K]
    mat_t bc_r = bc_ij * value; // [1,K]
    BC_BC = BC_BC + bc_bc;
    BC_R = BC_R + bc_r;
  }
  return forward_as_tuple(BC_BC, BC_R.transpose());
}
