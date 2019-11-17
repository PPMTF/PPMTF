#include "common.hpp"
#include "PPXTF.hpp"

#include <cstdio>
#include <string>
#include <cstdlib>
#include <utility>

using namespace std;

#define rep(i,n) for(int (i)=0;(i)<(n);(i)++)

constexpr int BUF_SIZE = 1000;

inline string i2s(int x){
  char buf[BUF_SIZE];
  sprintf(buf, "%d", x);
  return buf;
}

mat_t load_mat(const string& infile){
  vector<vector<double> > mat;
  ifstream ifs(infile);
  string line;
  while(getline(ifs, line)){
    vector<double> row;
    for(const auto& field : split(line)){
      double x;
      sscanf(field.c_str(), "%lf", &x);
      row.push_back(x);
    }
    mat.push_back(row);
  }
  int m = mat.size();
  assert(m > 0);
  int n = mat[0].size();
  mat_t res = mat_t::Zero(m, n);
  for(int i = 0; i < m; ++i){
    assert(n == mat[i].size());
    for(int j = 0; j < n; ++j){
      res(i, j) = mat[i][j];
    }
  }
  return res;
}

mat_t ReadModelParameter(const string& prefix, int K, int ItrNum, const string& name){
  char buf[BUF_SIZE];
  sprintf(buf, "%s_Itr%d_%s.csv", prefix.c_str(), ItrNum, name.c_str());
  return load_mat(buf);
}

tuple<mat_t, mat_t, mat_t, mat_t> ReadModelParameters(const string& ModelParameterFile,
						      int K,
						      int ItrNum){
  mat_t A = ReadModelParameter(ModelParameterFile, K, ItrNum, "A");
  mat_t B = ReadModelParameter(ModelParameterFile, K, ItrNum, "B");
  mat_t C = ReadModelParameter(ModelParameterFile, K, ItrNum, "C");
  mat_t D = ReadModelParameter(ModelParameterFile, K, ItrNum, "D");
  return forward_as_tuple(A,B,C,D);
}

tuple<r_t, r_t> ReadTrainTransTensor2(int N, int M, int ZeroNum, const string& infile){
  r_t RT1_observed(N), RT2_observed(M), RT3_observed(M);
  r_t RT1_unobserved(N);
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
	  if(zero_num < ZeroNum){
	    RT1_observed[user_index][p_t(poi_index_from,poi_index_to)] = 0;
	  }else{
	    RT1_unobserved[user_index][p_t(poi_index_from,poi_index_to)] = 0;
	  }
	  zero_num += 1;
	  if(zero_num == 2*ZeroNum){break;}
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
	    RT1_observed[user_index][p] = 0;
	  }
	}
      }
    }
  }
  return forward_as_tuple(RT1_observed, RT1_unobserved);
}

tuple<r_t, r_t> ReadTrainVisitTensor2(int N, int M, int T, int ZeroNum, const string& infile){
  r_t RV1_observed(N), RV2_observed(M), RV3_observed(T);
  r_t RV1_unobserved(N);
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
	int time_slot = rand_index[i] / M;
	auto& a = RV1_observed[user_index];
	auto p = p_t(poi_index_from,time_slot);
	if(a.find(p) == a.end()){
	  if(zero_num < ZeroNum){
	    a[p] = 0;
	  }else{
	    RV1_unobserved[user_index][p] = 0;
	  }
	  zero_num += 1;
	  if(zero_num == 2*ZeroNum){break;}
	}
      }
    }
  }else if(ZeroNum == -1){
    for(int user_index = 0; user_index < N; ++user_index){
      for(int poi_index_from = 0; poi_index_from < M; ++poi_index_from){
	for(int time_slot = 0; time_slot < T; ++time_slot){
	  auto& a = RV1_observed[user_index];
	  auto p = p_t(poi_index_from,time_slot);
	  if(a.find(p) == a.end()){
	    a[p] = 0;
	  }
	}
      }
    }
  }
  return forward_as_tuple(RV1_observed, RV1_unobserved);
}


int main(int argc, char* argv[]){
  string DataSet;
  string City;

  if(argc < 3){
    printf("Usage: %s [Dataset] [City] ([Alp (default:200)] [MaxNumTrans (default:100)] [MaxNumVisit (default:100)] [ItrNum (default:100)])\n", argv[0]);
    return -1;
  }

  // # Dataset (PF/FS)
  if(argc > 1){DataSet = argv[1];}
  // # City
  if(argc > 2){City = argv[2];}
  // # Hyper-hyper parameter alpha
  double Alp = 200;
  string AlpStr = "200";
  if(argc >= 4){
    AlpStr = argv[3];
    sscanf(AlpStr.c_str(), "%lf", &Alp);
  }
  // # Maximum number of transitions per user (-1: infinity)
  int MaxNumTrans = 100;
  if(argc >= 5){sscanf(argv[4], "%d", &MaxNumTrans);}
  // # Maximum number of POI visits per user (-1: infinity)
  int MaxNumVisit = 100;
  if(argc >= 6){sscanf(argv[5], "%d", &MaxNumVisit);}
  // # Number of iterations in Gibbs sampling
  int ItrNum = 100;
  if(argc >= 7){sscanf(argv[6], "%d", &ItrNum);}
  // # Data directory
  string DataDir = "../data/" + DataSet + "/";
  // # Training user index file (input)
  string TUserIndexFile = DataDir + "tuserindex_%s.csv";
  // # POI index file (input)
  string POIIndexFile = DataDir + "POIindex_%s.csv";
  // # Training transition tensor file (input)
  string TrainTransTensorFile = DataDir + "traintranstensor_%s_mnt" + i2s(MaxNumTrans) + ".csv";
  // # Training visit tensor file (input)
  string TrainVisitTensorFile = DataDir + "trainvisittensor_%s_mnv" + i2s(MaxNumVisit) + ".csv";
  // # Prefix of the model parameter file (output)
  // #OutDir = DataDir + "PPMTF_" + City + "_alp" + AlpStr + "_mnt" + str(MaxNumTrans) + "_mnv" + str(MaxNumVisit) + "/"
  //string OutDir = DataDir + "PPMTF_" + City + "_alp" + AlpStr + "_mnt" + i2s(MaxNumTrans) + "_mnv" + i2s(MaxNumVisit) + "_py/";
  string OutDir = DataDir + "PPMTF_" + City + "_alp" + AlpStr + "_mnt" + i2s(MaxNumTrans) + "_mnv" + i2s(MaxNumVisit) + "/";
  string ModelParameterFile = OutDir + "modelparameter";
  // # Number of time periods
  int T;
  if(DataSet == "PF"){
    T = 30;
  }else if(DataSet == "FS"){
    T = 12;
  }else{
    fprintf(stderr, "Wrong Dataset");
    exit(-1);
  }
  // # Number of columns in model parameters (A, B, C, D)
  int K = 16;
  // # Count-to-rating table
  // defined in PPXTF.cpp
  // # Number of zero elements for each user in a training tensor (-1: all)
  int ZeroNum = 1000;

  // # Fix a seed
//  seeding(1);
  seeding();
  // Replace %s with City
  TUserIndexFile = string_replace(TUserIndexFile, City);
  POIIndexFile = string_replace(POIIndexFile, City);
  TrainTransTensorFile = string_replace(TrainTransTensorFile, City);
  TrainVisitTensorFile = string_replace(TrainVisitTensorFile, City);
  // Number of training users --> N
  int N = line_num(TUserIndexFile) - 1;
  // Number of POIs --> M
  int M = line_num(POIIndexFile) - 1;

  // # Read a training transition tensor
  puts("Reading a training transition tensor.");
  printf("%s\n", TrainTransTensorFile.c_str());
  r_t RT1_observed, RT1_unobserved;
  tie(RT1_observed, RT1_unobserved) = ReadTrainTransTensor2(N, M, ZeroNum, TrainTransTensorFile);

  // # Read a training visit tensor
  puts("Reading a training visit tensor.");
  printf("%s\n", TrainVisitTensorFile.c_str());
  r_t RV1_observed, RV1_unobserved;
  tie(RV1_observed, RV1_unobserved) = ReadTrainVisitTensor2(N, M, T, ZeroNum, TrainVisitTensorFile);

  // # Read model parameters
  puts("Reading model parameters.");
  printf("%s\n",ModelParameterFile.c_str());
  mat_t A, B, C, D;
  tie(A, B, C, D) = ReadModelParameters(ModelParameterFile, K, ItrNum);

  // # Compute l1 loss
  double l1_loss_RT_nonzero = 0.0;
  double l1_loss_RT_zero = 0.0;
  double l1_loss_RT_unobserved = 0.0;
  double l1_loss_RV_nonzero = 0.0;
  double l1_loss_RV_zero = 0.0;
  double l1_loss_RV_unobserved = 0.0;
  double RT_nonzero_num = 0;
  double RT_zero_num = 0;
  double RT_unobserved_num = 0;
  double RV_nonzero_num = 0;
  double RV_zero_num = 0;
  double RV_unobserved_num = 0;
  rep(n, N){
    for(const auto& it : RT1_observed[n]){
      auto p = it.first;
      int i = p.first;
      int j = p.second;
      double val = it.second;
      double rhs = abs(val - (A.row(n).array() * B.row(i).array() * C.row(j).array() ).sum());
      if(val > 0.0){
	l1_loss_RT_nonzero += rhs;
	RT_nonzero_num += 1;
      }else{
	l1_loss_RT_zero += rhs;
	RT_zero_num += 1;
      }
    }
    for(const auto& it : RT1_unobserved[n]){
      auto p = it.first;
      int i = p.first;
      int j = p.second;
      double val = it.second;
      double rhs = abs(val - (A.row(n).array() * B.row(i).array() * C.row(j).array() ).sum());
      l1_loss_RT_unobserved += rhs;
      RT_unobserved_num += 1;
    }
    for(const auto& it : RV1_observed[n]){
      auto p = it.first;
      int i = p.first;
      int j = p.second;
      double val = it.second;
      double rhs = abs(val - (A.row(n).array() * B.row(i).array() * D.row(j).array() ).sum());
      if(val > 0.0){
	l1_loss_RV_nonzero += rhs;
	RV_nonzero_num += 1;
      }else{
	l1_loss_RV_zero += rhs;
	RV_zero_num += 1;
      }
    }
    for(const auto& it : RV1_unobserved[n]){
      auto p = it.first;
      int i = p.first;
      int j = p.second;
      double val = it.second;
      double rhs = abs(val - (A.row(n).array() * B.row(i).array() * D.row(j).array() ).sum());
      l1_loss_RV_unobserved += rhs;
      RV_unobserved_num += 1;
    }
  }
  double l1_loss_RT = l1_loss_RT_nonzero + l1_loss_RT_zero + l1_loss_RT_unobserved;
  double l1_loss_RV = l1_loss_RV_nonzero + l1_loss_RV_zero + l1_loss_RV_unobserved;

  puts("[Training Transition Tensor]");
  printf("#nonzero: %lf #zero: %lf #unobserved: %lf\n",RT_nonzero_num,RT_zero_num,RT_unobserved_num);
  printf("l1_loss (nonzero): %lf l1_loss (zero): %lf l1_loss (unobserved): %lf\n",l1_loss_RT_nonzero,l1_loss_RT_zero,l1_loss_RT_unobserved);
  printf("l1_loss (sum): %lf\n",l1_loss_RT);
  puts("[Training Visit Tensor]");
  printf("#nonzero: %lf #zero: %lf #unobserved: %lf\n",RV_nonzero_num,RV_zero_num,RV_unobserved_num);
  printf("l1_loss (nonzero): %lf l1_loss (zero): %lf l1_loss (unobserved): %lf\n",l1_loss_RV_nonzero,l1_loss_RV_zero,l1_loss_RV_unobserved);
  printf("l1_loss (sum): %lf\n",l1_loss_RV);

  return 0;
}
