#include "common.hpp"
#include "PPXTF.hpp"
#include <sys/stat.h>
#include <Eigen/LU>
#include"ProcessingTime.hpp"

using namespace Eigen;
using namespace std;

/*
  #################################### PPMTF ####################################
  # [input1]: RT1_observed ([user_index]{poi_index_from, poi_index_to: value})
  # [input2]: RT2_observed ([poi_index_from]{user_index, poi_index_to: value})
  # [input3]: RT3_observed ([poi_index_to]{user_index, poi_index_from: value})
  # [input4]: RV1_observed ([user_index]{poi_index_from, time_id: value})
  # [input5]: RV2_observed ([poi_index_from]{user_index, time_id: value})
  # [input6]: RV3_observed ([time_id]{user_index, poi_index_from: value})
  # [input7]: A (N x K matrix)
  # [input8]: B (M x K matrix)
  # [input9]: C (M x K matrix)
  # [input10]: D (M x K matrix)
  # [input11]: N -- Number of users
  # [input12]: M -- Number of POIs
  # [input13]: T -- Number of time periods
  # [output]: A, B, C, D, mu_A, Lam_A, mu_B, Lam_B, mu_C, Lam_C, mu_D, Lam_D
 */
tuple<mat_t, mat_t, mat_t, mat_t, mat_t, mat_t, mat_t, mat_t>
ppmtf(const r_t& rt1, const r_t& rt2, const r_t& rt3, const r_t& rv1, const r_t& rv2, const r_t& rv3,
      vec_mat_t& A, vec_mat_t& B, vec_mat_t& C, vec_mat_t& D,
      int N, int M, int T, int K, int ItrNum, double Alp,
      const string& outfile_prefix){
  // Hyper-hyper parameters (same as [Salakhutdinov+, ICML08])
  double alpha = Alp;
  double beta0 = 2;
  double mu0 = 0;
  int nu0 = K;
  mat_t W0 = mat_t::Identity(K,K);
  
  mat_t Lam_A, mu_A, Lam_B, mu_B, Lam_C, mu_C, Lam_D, mu_D;
  //puts("Sampling:");
  for(int itr = 0; itr < ItrNum; ++itr){
    //    printf("itr:%d\n", itr);

    // Sample Lam_{A,B,C,D} & mu_{A,B,C,D}
    tie(Lam_A, mu_A) = sample_Lam_X_mu_X(A[itr], N, nu0, W0, beta0, mu0, K);
    tie(Lam_B, mu_B) = sample_Lam_X_mu_X(B[itr], M, nu0, W0, beta0, mu0, K);
    tie(Lam_C, mu_C) = sample_Lam_X_mu_X(C[itr], M, nu0, W0, beta0, mu0, K);
    tie(Lam_D, mu_D) = sample_Lam_X_mu_X(D[itr], T, nu0, W0, beta0, mu0, K);
    
    // Sample A
    A.push_back(mat_t::Zero(N, K));
    for(int n = 0; n < N; ++n){
      mat_t BC_BC, BC_R, BD_BD, BD_R;
      tie(BC_BC, BC_R) = calc_XYXY_XYR(K, B[itr], C[itr], rt1[n]);
      tie(BD_BD, BD_R) = calc_XYXY_XYR(K, B[itr], D[itr], rv1[n]);
      // lam_ast_inv = inv(Lam_A + alpha * BC_BC + alpha * BD_BD)
      mat_t lam_ast_inv = (Lam_A + alpha * BC_BC + alpha * BD_BD).inverse();
      //mu_ast = lam_ast_inv.dot((alpha * BC_R + alpha * BD_R + Lam_A.dot(mu_A.T)).T)
      mat_t mu_ast = lam_ast_inv * (alpha * BC_R + alpha * BD_R + Lam_A * mu_A);
      //mat_t x = multivariate_normal(mu_ast, lam_ast_inv);
      A[itr+1].row(n) = multivariate_normal(mu_ast, lam_ast_inv).transpose();
    }
    
    // Sample B
    B.push_back(mat_t::Zero(M, K));
    for(int i = 0; i < M; ++i){
      mat_t AC_AC, AC_R, AD_AD, AD_R;
      tie(AC_AC, AC_R) = calc_XYXY_XYR(K, A[itr+1], C[itr], rt2[i]);
      tie(AD_AD, AD_R) = calc_XYXY_XYR(K, A[itr+1], D[itr], rv2[i]);
      mat_t lam_ast_inv = (Lam_B + alpha * AC_AC + alpha * AD_AD).inverse();
      mat_t mu_ast = lam_ast_inv * (alpha * AC_R + alpha * AD_R + Lam_B * mu_B);
      B[itr+1].row(i) = multivariate_normal(mu_ast, lam_ast_inv).transpose();
    }

    // Sample C
    C.push_back(mat_t::Zero(M, K));
    for(int j = 0; j < M; ++j){
      mat_t AB_AB, AB_R;
      tie(AB_AB, AB_R) = calc_XYXY_XYR(K, A[itr+1], B[itr+1], rt3[j]);
      mat_t lam_ast_inv = (Lam_C + alpha * AB_AB).inverse();
      mat_t mu_ast = lam_ast_inv * (alpha * AB_R + Lam_C * mu_C);
      C[itr+1].row(j) = multivariate_normal(mu_ast, lam_ast_inv).transpose();
    }

    // Sample D
    D.push_back(mat_t::Zero(T, K));
    for(int j = 0; j < T; ++j){
      mat_t AB_AB, AB_R;
      tie(AB_AB, AB_R) = calc_XYXY_XYR(K, A[itr+1], B[itr+1], rv3[j]);
      mat_t lam_ast_inv = (Lam_D + alpha * AB_AB).inverse();
      mat_t mu_ast = lam_ast_inv * (alpha * AB_R + Lam_D * mu_D);
      D[itr+1].row(j) = multivariate_normal(mu_ast, lam_ast_inv).transpose();
    }

    /*
    // Output model parameters and hyper-parameters
    if(itr == 0 || itr % 10 == 9){
      string pre = outfile_prefix;
      save_parameters(K, itr+1, pre, "A", A, mu_A, Lam_A);
      save_parameters(K, itr+1, pre, "B", B, mu_B, Lam_B);
      save_parameters(K, itr+1, pre, "C", C, mu_C, Lam_C);
      save_parameters(K, itr+1, pre, "D", D, mu_D, Lam_D);
    }
    */
  }
  return forward_as_tuple(mu_A, Lam_A, mu_B, Lam_B, mu_C, Lam_C, mu_D, Lam_D);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]){
{
  ProcessingTime pt("PPMTF_process");
  if(argc < 3){
      printf("Usage: PPMTF_time [Dataset] [City] ([Alp (default:200)] [MaxNumTrans (default:100)] [MaxNumVisit (default:100)] [ItrNum (default:100)])\n");
      return -1;
  }

  // Fix a seed
  seeding();

  // ################################# Parameters ##################################
  // Dataset
  const string Dataset = argv[1];
  // City
//  const string City = "TK";
  const string City = argv[2];
  // Hyper-hyper parameter alpha
  double Alp = 200;
  string AlpStr = "200";
  if(argc >= 4){
	  Alp = atof(argv[3]);
	  AlpStr = argv[3];
  }
  // Maximum number of transitions per user (-1: infinity)
  int MaxNumTrans = 100;
  if(argc >= 5){
	  MaxNumTrans = atoi(argv[4]);
  }
  // Maximum number of POI visits per user (-1: infinity)
  int MaxNumVisit = 100;
  if(argc >= 6){
	  MaxNumVisit = atoi(argv[5]);
  }
  // Number of iterations in Gibbs sampling
  int ItrNum = 100;
  if(argc >= 7){
	  ItrNum = atoi(argv[6]);
  }

  // Training user index file (input)
  string TUserIndexFile = "../data/" + Dataset + "/tuserindex_%s.csv";
  // POI index file (input)
  string POIIndexFile = "../data/" + Dataset + "/POIindex_%s.csv";
  // Training transition tensor file (input)
  string TrainTransTensorFile = "../data/" + Dataset + "/traintranstensor_%s_mnt" + std::to_string(MaxNumTrans) + ".csv";
  // Training visit tensor file (input)
  string TrainVisitTensorFile = "../data/" + Dataset + "/trainvisittensor_%s_mnv" + std::to_string(MaxNumVisit) + ".csv";
  // Prefix of the model parameter file (output)
  const string OutDir = "../data/" + Dataset + "/PPMTF_" + City + "_alp" + AlpStr + "_mnt" + std::to_string(MaxNumTrans) + "_mnv" + std::to_string(MaxNumVisit);
  mkdir(OutDir.c_str(), 0755);
  const string ModelParameterFile = OutDir + "/modelparameter";

  // Number of time periods
  int T;
  if(Dataset.find("PF") == 0){
	  T = 30;
  }else if(Dataset.find("FS") == 0){
	  T = 12;
  }else{
	  cout << "Wrong Dataset\n";
	  exit(-1);
  }

  // Number of columns in model parameters (A, B, C)
//  const int K = 32;
  const int K = 16;
  // Number of zero elements for each user in a training tensor (-1: all)
  const int ZeroNum = 1000;
  
  // Replace %s with City
  TUserIndexFile = string_replace(TUserIndexFile, City);
  POIIndexFile = string_replace(POIIndexFile, City);
  TrainTransTensorFile = string_replace(TrainTransTensorFile, City);
  TrainVisitTensorFile = string_replace(TrainVisitTensorFile, City);
  
  // Number of training users --> N
  int N = line_num(TUserIndexFile) - 1;
  // Number of POIs --> M
  int M = line_num(POIIndexFile) - 1;
  
  // Read a training transition tensor
  //puts("Reading a training transition tensor.");
  r_t RT1_observed, RT2_observed, RT3_observed;
  tie(RT1_observed, RT2_observed, RT3_observed) = ReadTrainTransTensor(N, M, ZeroNum, TrainTransTensorFile);
  // Read a training visit tensor
  //puts("Reading a training visit tensor.");
  r_t RV1_observed, RV2_observed, RV3_observed;
  tie(RV1_observed, RV2_observed, RV3_observed) = ReadTrainVisitTensor(N, M, T, ZeroNum, TrainVisitTensorFile);
  
  // Initialize model parameters (A, B, C, D) by random values in [0,1)
  vec_mat_t A, B, C, D;
  A.push_back(random_mat(N, K));
  B.push_back(random_mat(M, K));
  C.push_back(random_mat(M, K));
  D.push_back(random_mat(T, K));
  
  // PPMTF--> A, B, C, D, mu_A, Lam_A, mu_B, Lam_B, mu_C, Lam_C, mu_D, Lam_D
  mat_t mu_A, Lam_A, mu_B, Lam_B, mu_C, Lam_C, mu_D, Lam_D;
  tie(mu_A, Lam_A, mu_B, Lam_B, mu_C, Lam_C, mu_D, Lam_D) = ppmtf(RT1_observed, RT2_observed, RT3_observed,
								  RV1_observed, RV2_observed, RV3_observed,
								  A, B, C, D, N, M, T, K, ItrNum, Alp, ModelParameterFile);
  
  // Output the final model parameters (A, B, C, D)
  // Output the final hyper-parameters (mu_A, Lam_A, mu_B, Lam_B, mu_C, Lam_C, mu_D, Lam_D)
  save_parameters(K, ItrNum, ModelParameterFile, "A", A, mu_A, Lam_A);
  save_parameters(K, ItrNum, ModelParameterFile, "B", B, mu_B, Lam_B);
  save_parameters(K, ItrNum, ModelParameterFile, "C", C, mu_C, Lam_C);
  save_parameters(K, ItrNum, ModelParameterFile, "D", D, mu_D, Lam_D);
  return 0;
}
}
