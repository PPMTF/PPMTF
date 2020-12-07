#include "common.hpp"
#include "PPXTF.hpp"
#include <sys/stat.h>
#include <Eigen/LU>

using namespace Eigen;
using namespace std;

/*
  #################################### PPMTF ####################################
  # [input1]: R1_observed ([mode1]{mode2, mode3: value})
  # [input2]: R2_observed ([mode2]{mode1, mode3: value})
  # [input3]: R3_observed ([mode3]{mode1, mode2: value})
  # [input4]: A (L1 x K matrix)
  # [input5]: B (L2 x K matrix)
  # [input6]: C (L3 x K matrix)
  # [input7]: L1 -- Length of mode1
  # [input8]: L2 -- Length of mode2
  # [input9]: L3 -- Length of mode3
  # [input10]: X -- tensor no.
  # [output]: A, B, C, mu_A, Lam_A, mu_B, Lam_B, mu_C, Lam_C
 */
tuple<mat_t, mat_t, mat_t, mat_t, mat_t, mat_t>
pptf(const r_t& r1, const r_t& r2, const r_t& r3,
     vec_mat_t& A, vec_mat_t& B, vec_mat_t& C,
     int L1, int L2, int L3, int K, int ItrNum, double Alp,
     const string& outfile_prefix, const string& X){
  // Hyper-hyper parameters (same as [Salakhutdinov+, ICML08])
  double alpha = Alp;
  double beta0 = 2;
  double mu0 = 0;
  int nu0 = K;
  mat_t W0 = mat_t::Identity(K,K);
  
  mat_t Lam_A, mu_A, Lam_B, mu_B, Lam_C, mu_C;
  puts("Gibbs Sampling:");
  for(int itr = 0; itr < ItrNum; ++itr){
    if(itr % 10 == 0) printf("itr:%d\n", itr);

    // Sample Lam_{A,B,C,D} & mu_{A,B,C,D}
    tie(Lam_A, mu_A) = sample_Lam_X_mu_X(A[itr], L1, nu0, W0, beta0, mu0, K);
    tie(Lam_B, mu_B) = sample_Lam_X_mu_X(B[itr], L2, nu0, W0, beta0, mu0, K);
    tie(Lam_C, mu_C) = sample_Lam_X_mu_X(C[itr], L3, nu0, W0, beta0, mu0, K);
    
    // Sample A
    A.push_back(mat_t::Zero(L1, K));
    for(int n = 0; n < L1; ++n){
      mat_t BC_BC, BC_R;
      tie(BC_BC, BC_R) = calc_XYXY_XYR(K, B[itr], C[itr], r1[n]);
      mat_t lam_ast_inv = (Lam_A + alpha * BC_BC).inverse();
      mat_t mu_ast = lam_ast_inv * (alpha * BC_R + Lam_A * mu_A);
      A[itr+1].row(n) = multivariate_normal(mu_ast, lam_ast_inv).transpose();
    }
    
    // Sample B
    B.push_back(mat_t::Zero(L2, K));
    for(int i = 0; i < L2; ++i){
      mat_t AC_AC, AC_R;
      tie(AC_AC, AC_R) = calc_XYXY_XYR(K, A[itr+1], C[itr], r2[i]);
      mat_t lam_ast_inv = (Lam_B + alpha * AC_AC).inverse();
      mat_t mu_ast = lam_ast_inv * (alpha * AC_R + Lam_B * mu_B);
      B[itr+1].row(i) = multivariate_normal(mu_ast, lam_ast_inv).transpose();
    }

    // Sample C
    C.push_back(mat_t::Zero(L3, K));
    for(int j = 0; j < L3; ++j){
      mat_t AB_AB, AB_R;
      tie(AB_AB, AB_R) = calc_XYXY_XYR(K, A[itr+1], B[itr+1], r3[j]);
      mat_t lam_ast_inv = (Lam_C + alpha * AB_AB).inverse();
      mat_t mu_ast = lam_ast_inv * (alpha * AB_R + Lam_C * mu_C);
      C[itr+1].row(j) = multivariate_normal(mu_ast, lam_ast_inv).transpose();
    }

    // Output model parameters and hyper-parameters
    if(itr == 0 || itr % 10 == 9){
      string pre = outfile_prefix;
      save_parameters(K, itr+1, pre, "A", A, mu_A, Lam_A, X);
      save_parameters(K, itr+1, pre, "B", B, mu_B, Lam_B, X);
      save_parameters(K, itr+1, pre, "C", C, mu_C, Lam_C, X);
    }
  }
  return forward_as_tuple(mu_A, Lam_A, mu_B, Lam_B, mu_C, Lam_C);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]){
  if(argc < 3){
      printf("Usage: PPITF [Dataset] [City] ([Alp (default:200)] [MaxNumTrans (default:100)] [MaxNumVisit (default:100)] [ItrNum (default:100)])\n");
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
  const string OutDir = "../data/" + Dataset + "/PPITF_" + City + "_alp" + AlpStr + "_mnt" + std::to_string(MaxNumTrans) + "_mnv" + std::to_string(MaxNumVisit);
  mkdir(OutDir.c_str(), 0755);
  const string ModelParameterFile = OutDir + "/modelparameter";

  // Number of time slots
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
  puts("Reading a training transition tensor.");
  r_t RT1_observed, RT2_observed, RT3_observed;
  tie(RT1_observed, RT2_observed, RT3_observed) = ReadTrainTransTensor(N, M, ZeroNum, TrainTransTensorFile);
  // Read a training visit tensor
  puts("Reading a training visit tensor.");
  r_t RV1_observed, RV2_observed, RV3_observed;
  tie(RV1_observed, RV2_observed, RV3_observed) = ReadTrainVisitTensor(N, M, T, ZeroNum, TrainVisitTensorFile);
  
  {
    // Initialize model parameters (AT, BT, CT) of the training transition tensor by random values in [0,1)
    vec_mat_t A, B, C;
    A.push_back(random_mat(N, K));
    B.push_back(random_mat(M, K));
    C.push_back(random_mat(M, K));
    // PPTF for the training transition tensor --> AT, BT, CT, mu_AT, Lam_AT, mu_BT, Lam_BT, mu_CT, Lam_CT
    mat_t mu_A, Lam_A, mu_B, Lam_B, mu_C, Lam_C;
    tie(mu_A, Lam_A, mu_B, Lam_B, mu_C, Lam_C) = pptf(RT1_observed, RT2_observed, RT3_observed,
						      A, B, C, N, M, M, K, ItrNum, Alp, ModelParameterFile, "T");
  }
  {
    // Initialize model parameters (AV, BV, CV) of the training visit tensor by random values in [0,1)
    vec_mat_t A, B, C;
    A.push_back(random_mat(N, K));
    B.push_back(random_mat(M, K));
    C.push_back(random_mat(T, K));
    // PPTF for the training visit tensor --> AV, BV, CV, mu_AV, Lam_AV, mu_BV, Lam_BV, mu_CV, Lam_CV
    mat_t mu_A, Lam_A, mu_B, Lam_B, mu_C, Lam_C;
    tie(mu_A, Lam_A, mu_B, Lam_B, mu_C, Lam_C) = pptf(RV1_observed, RV2_observed, RV3_observed,
						      A, B, C, N, M, T, K, ItrNum, Alp, ModelParameterFile, "V");
  }
  return 0;
}
