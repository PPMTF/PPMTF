#include "common.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <cassert>
#include <random>
#include <vector>
#include <utility>
#include <map>
#include <cmath>
#include <algorithm>

#define rep(i,n) for(int (i)=0;(i)<(n);(i)++)

// eigen
#include <Eigen/Core>

using namespace Eigen;
using namespace std;

constexpr int BUF_SIZE = 1000;

using p_t = pair<int, int>;
using r_t = vector<map<p_t, double> >;
using mat_t = MatrixXd;
using vec_t = VectorXd;
using vec_mat_t = vector<mat_t>;

std::mt19937_64 engine;//(seed_gen());

vector<int> seq(int n){
  vector<int> v(n);
  for(int i = 0; i < n; ++i){v[i] = i;}
  return v;
}

vector<int> random_index(int n){
  vector<int> res = seq(n);
  shuffle(res.begin(), res.end(), engine);
  return res;
}

void seeding(unsigned seed = 0){
  //std::random_device seed_gen;
  //engine = std::mt19937_64(seed_gen());
  engine = std::mt19937_64(seed);
}

string to_str(int x){
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
  sprintf(buf, "%s_K%d_Itr%d_%s.csv", prefix.c_str(), K, ItrNum, name.c_str());
  //read_file(buf);

  //mat_t tmp = load_mat(buf);
  //fprintf(stderr, "[%ld, %ld]\n", tmp.rows(), tmp.cols());

  return load_mat(buf);
}
/*
  ########################### Read model parameters #############################
  # [output1]: A (N x K matrix)
  # [output2]: B (M x K matrix)
  # [output3]: C (M x K matrix)
  # [output4]: D (T x K matrix)
*/
tuple<mat_t, mat_t, mat_t, mat_t> ReadModelParameters(const string& prefix, int K, int ItrNum){
  mat_t A = ReadModelParameter(prefix, K, ItrNum, "A");
  mat_t B = ReadModelParameter(prefix, K, ItrNum, "B");
  mat_t C = ReadModelParameter(prefix, K, ItrNum, "C");
  mat_t D = ReadModelParameter(prefix, K, ItrNum, "D");
  return forward_as_tuple(A, B, C, D);
}

/*
######################### Plausible deniability test ##########################
# [input1]: A (N x K matrix)
# [input2]: B (M x K matrix)
# [input3]: C (M x K matrix)
# [input4]: D (T x K matrix)
# [input5]: N -- Number of users
# [input6]: M -- Number of POIs
# [input7]: T -- Number of time periods
# [output1]: pass_test (N-dim vector)
# [output2]: loglikeli_train (N*TraceNum-dim vector)
# [output3]: loglikeli_verify (N*TraceNum x VUserNum matrix)
 */

using pass_test_t = vector<int>;

void PDTest_out1(const string& _outfile,
		 int N,
		 int TraceNum,
		 const pass_test_t& pass_test){
  //  string outfile = _outfile + "_";
  string outfile = _outfile;
  puts(outfile.c_str());
  FILE* fp = fopen(outfile.c_str(), "w");
  rep(user_index, N){
    rep(trace_no, TraceNum){
      int user_trace_no = user_index * TraceNum + trace_no;
//      fprintf(fp, "%lf\n", static_cast<double>(pass_test[user_trace_no]));
      fprintf(fp, "%d,%d\n", pass_test[user_trace_no + N * TraceNum], pass_test[user_trace_no]);
    }
  }
  int pass_test_sum = 0;
  for(int i = 0; i < N * TraceNum; i++){
    pass_test_sum += pass_test[i];
  }
  fprintf(fp, "-,-,%d,%d,%lf\n", pass_test_sum, N * TraceNum, float(pass_test_sum) / float(N * TraceNum));
  fclose(fp);
}
void PDTest_out2(const string& _outfile,
		 int N,
		 int TraceNum,
		 int VUserNum,
		 const mat_t& loglikeli_train,
		 const mat_t& loglikeli_verify){
  //  string outfile = _outfile + "_";
  string outfile = _outfile;
  puts(outfile.c_str());
  FILE* fp = fopen(outfile.c_str(), "w");
  rep(user_index, N){
    rep(trace_no, TraceNum){
      int user_trace_no = user_index * TraceNum + trace_no;
      fprintf(fp, "%lf", loglikeli_train(user_trace_no, 0));
      rep(n, VUserNum){
	fprintf(fp, ",%lf", loglikeli_verify(user_trace_no,n));
      }
      fprintf(fp, "\n");
    }
  }
  fclose(fp);
}

pass_test_t perform_pd_test(int N,
			    int TraceNum,
			    int VUserNum,
			    const vector<int>& verify_user,
			    const mat_t& loglikeli_train,
			    const mat_t& loglikeli_verify,
			    double ReqEps,
			    int Reqk){
  /*
  pass_test_t pass_test(N * TraceNum, 0);
  // # For each training user
  rep(user_index, N){
    // # For each trace
    rep(trace_no, TraceNum){
      int user_trace_no = user_index * TraceNum + trace_no;
      // # Initialization
      int k = 0;
      // # For each verifying user
      rep(n, VUserNum){
	// # Continue if the training user is the same with the verifying user
	if(user_index == verify_user[n]){continue;}
	// # Increase k if the inequality holds with ReqEps
	if((loglikeli_train(user_trace_no, 0) >= loglikeli_verify(user_trace_no,n) - ReqEps) &&
	   (loglikeli_train(user_trace_no, 0) <= loglikeli_verify(user_trace_no,n) + ReqEps)){
	  ++k;
	}
	// # Break if k reaches Reqk
	if(k == Reqk){
	  pass_test[user_trace_no] = 1;
	  break;
	}
      }
    }
  }
  */
  pass_test_t pass_test(N * TraceNum * 2, 0);
  // # For each training user
  int i_tra, i_ver;
  rep(user_index, N){
    // # For each trace
    rep(trace_no, TraceNum){
      int user_trace_no = user_index * TraceNum + trace_no;
      pass_test[user_trace_no + N * TraceNum] += 1;
      // # Find i s.t. [-(i+1)ReqEps < loglikeli_train[user_trace_no] <= -i ReqEps] --> i_tra
      i_tra = int(loglikeli_train(user_trace_no, 0) / ReqEps);
      // # For each verifying user
      rep(n, VUserNum){
	// # Continue if the training user is the same with the verifying user
	if(user_index == verify_user[n]){continue;}
	// # Find i s.t. [-(i+1)ReqEps < loglikeli_verify[user_trace_no,n] <= -i ReqEps] --> i_ver
	i_ver = int(loglikeli_verify(user_trace_no,n) / ReqEps);
	// # Increase k if the inequality holds with ReqEps (i_tra == i_ver)
	if(i_tra == i_ver){
	  pass_test[user_trace_no + N * TraceNum] += 1;
	}
	// # Pass test (and break (optional)) if k reaches Reqk
	if(pass_test[user_trace_no + N * TraceNum] == Reqk){
	  pass_test[user_trace_no] = 1;
//	  break;
	}
      }
    }
  }
  return pass_test;
}

int str2i(const string& s){
  int x;
  sscanf(s.c_str(), "%d", &x);
  return x;
}

double str2f(const string& s){
  double x;
  sscanf(s.c_str(), "%lf", &x);
  return x;
}

using trans_t = vector<vector<pair<int, int> > >;
using first_visit_t = vector<int>;

tuple<trans_t, first_visit_t, mat_t> read_syn_traces(const string& infile,
					     int TraceNum,
					     int T,
					     int TimInsNum,
					     int N){
  // init
  trans_t trans(N * TraceNum);
  first_visit_t first_visit(N * TraceNum, 0);
  mat_t loglikeli_train = mat_t::Zero(N * TraceNum, 1);
  int poi_index_prev = 0;
  //printf("%s\n", infile.c_str());
  ifstream ifs(infile);
  string line;
  getline(ifs, line); // header
  while(getline(ifs, line)){
    auto lst = split(line);
    int user_index = str2i(lst[0]);
    int trace_no = str2i(lst[1]);
    int time_slot = str2i(lst[2]);
    int time_ins = str2i(lst[3]);
    int poi_index = str2i(lst[4]);
    double ll = str2f(lst[6]);

    // # (user,trace)-pair no. --> user_trace_no
    int user_trace_no = user_index * TraceNum + trace_no;
    if(time_slot == 0 && time_ins == 0){
      // # Visited POI at the first time --> first_visit[user_trace_no] 
      first_visit[user_trace_no] = poi_index;
    }else{
      // # Transition from the previous POI to the current POI --> trans[user_trace_no]
      trans[user_trace_no].push_back(make_pair(poi_index_prev,poi_index));
    }
    // # Log-likelihood for the training trace --> loglikeli_train[user_trace_no]
    if(time_slot == T-1 && time_ins == TimInsNum-1){
      loglikeli_train(user_trace_no, 0) = ll;
    }
    poi_index_prev = poi_index;
  }
  return forward_as_tuple(trans, first_visit, loglikeli_train);
}

mat_t calc_ll(int VUserNum,
	      const vector<int>& verify_user,
	      int T, int M,
	      double VisThr,
	      double VisitDelta,
	      const mat_t& A, // N * K
	      const mat_t& B, // M * K
	      const mat_t& C, // M * K
	      const mat_t& D, // T * K
	      double TransDelta,
	      int N,
	      int TraceNum,
	      int TimInsNum,
	      const first_visit_t& first_visit,
	      const trans_t& trans){
  mat_t loglikeli_verify = mat_t::Zero(N*TraceNum,VUserNum);
  rep(n, VUserNum){
    int vn = verify_user[n];
    printf("%d, %d\n", n, vn);
    // # Initialization
    mat_t time_poi_dist = mat_t::Zero(T, M);
    mat_t time_poi_dist_sum = mat_t::Zero(T, 1);
    mat_t prop_mat = mat_t::Zero(M, M);
    mat_t same_trans = mat_t::Zero(T, M); //np.full((T,M), -1.0);
    for(int i = 0; i < T; ++i)for(int j = 0; j < M; ++j){same_trans(i, j) = -1.0;}
    mat_t trans_vec = mat_t::Zero(M, 1);
    
    // ################### Calculate the POI distributions ###################
    puts("POI distribution");
    rep(t, T){
      mat_t ad = A.row(vn).array() * D.row(t).array();
      rep(i, M){
	// # Elements in a sampled visit tensor --> time_poi_dist
	time_poi_dist(t,i) = (ad.array() * B.row(i).array()).sum();
	// # Assign VisitDelta for an element whose value is less than VisThr
	if(time_poi_dist(t,i) < VisThr){
	  time_poi_dist(t,i) = VisitDelta;
	}
      }
    }
    // # Normalize time_poi_dist
    rep(t, T){
      time_poi_dist_sum(t) = time_poi_dist.row(t).sum();
      if(time_poi_dist_sum(t) > 0){
	time_poi_dist.row(t) /= time_poi_dist_sum(t);
      }else{
	printf("Error: All probabilities are 0 for user %d and time %d\n", n, t);
	exit(1);
      }
    }
    // #################### Calculate the proposal matrix ####################
    puts("Proposal matrix");
    rep(i, M){
      mat_t ab = A.row(vn).array() * B.row(i).array();
      // # Elements in a sampled transition tensor (assign TransDelta for a small transition count) --> prop_mat
      rep(j, M){
	prop_mat(i,j) = max((ab.array() * C.row(j).array()).sum(), TransDelta);
      }
      // # Normalize prop_mat
      double row_sum = prop_mat.row(i).sum();
      prop_mat.row(i) /= row_sum;
    }
    // #################### Calculate the log-likelihood #####################
    puts("Calculating the log-likelihood");
    // # For each training user
    rep(user_index, N){
      // # Continue if the training user is the same with the verifying user
      if(user_index == vn){continue;}
      // # For each trace
      rep(trace_no, TraceNum){
	int user_trace_no = user_index * TraceNum + trace_no;
	// # For each time slot
	rep(t, T){
	  // # For each time instant
	  rep(ins, TimInsNum){
	    if(t == 0 && ins == 0){
	      // # Add the log-likelihood for the first POI
	      int poi_index = first_visit[user_trace_no];
	      loglikeli_verify(user_trace_no, n) = log(time_poi_dist(0,poi_index));
	    }else{
	      // # Add the log-likelihood for the subsequent POIs
	      int tim = t * TimInsNum + ins - 1;
	      int poi_index_pre = trans[user_trace_no][tim].first;
	      int poi_index = trans[user_trace_no][tim].second;
	      double trans_prob;
	      // # If the current POI is different from the previous POI
	      if(poi_index_pre != poi_index){
		// # Calculate the transition probability --> trans_prob
		double alpha = (time_poi_dist(t, poi_index) * prop_mat(poi_index, poi_index_pre))
		  / (time_poi_dist(t, poi_index_pre) * prop_mat(poi_index_pre, poi_index));
		trans_prob = prop_mat(poi_index_pre,poi_index) * min(1.0, alpha);
	      // # If the self-transition probability for the POI at time slot t has been computed
	      }else if(same_trans(t,poi_index_pre) != -1.0){
		// # Use the self-transition probability --> trans_prob
		trans_prob = same_trans(t,poi_index_pre);
	      // # If the self-transition probability for the POI at time slot t has NOT been computed
	      }else{
		// # Compute the self-transition probability for the POI --> same_trans[t,poi_index_pre]
		trans_vec(poi_index_pre, 0) = 0;
		rep(j, M){
		  if(poi_index_pre != j){
		    double alpha = (time_poi_dist(t, j) * prop_mat(j, poi_index_pre))
		      / (time_poi_dist(t, poi_index_pre) * prop_mat(poi_index_pre,j));
		    trans_vec(j, 0) = prop_mat(poi_index_pre,j) * min(1.0, alpha);
		  }
		}
		double row_sum = trans_vec.sum();
		same_trans(t,poi_index_pre) = 1 - row_sum;
		// # Use the self-transition probability --> trans_prob
		trans_prob = same_trans(t,poi_index_pre);
	      }
	      // # Add the log of the transition probability
	      loglikeli_verify(user_trace_no,n) += log(trans_prob);
	      //
	    }
	    //
	  }
	}
      }
    }
    //
  }
  return loglikeli_verify;
}

void PDTest(mat_t& A,
	    mat_t& B,
	    mat_t& C,
	    mat_t& D,
	    int N,
	    int M,
	    int T,
	    int VUserNum,
	    const string& PDTestResFile,
	    int K,
	    int ItrNum,
	    int TraceNum,
	    double ReqEps,
	    int Reqk,
	    const string& SynTraceFile,
	    int TimInsNum,
	    double VisThr, double VisitDelta, double TransDelta){
  // # Initialization
  
  // # Randomly assign verifying users --> verify_user
  auto verify_user = random_index(VUserNum);
  
  // # Read synthesized traces --> trans, first_visit, loglikeli_train
  trans_t trans;
  first_visit_t first_visit;
  mat_t loglikeli_train;
  tie(trans, first_visit, loglikeli_train) =
    read_syn_traces(SynTraceFile + "_K" + to_str(K) + "_Itr" + to_str(ItrNum) + ".csv",
		    TraceNum, T, TimInsNum, N);
  
  // # Calculate the log-likelihood for each verifying user --> loglikeli_verify
  mat_t loglikeli_verify = calc_ll(VUserNum, verify_user, T, M, VisThr, VisitDelta, A, B, C, D, TransDelta, N, TraceNum, TimInsNum, first_visit, trans);

  // # Perform the PD test --> pass_test
  pass_test_t pass_test = perform_pd_test(N,TraceNum, VUserNum,
				    verify_user, loglikeli_train, loglikeli_verify,
				    ReqEps, Reqk);
  
  // # Output the PD test results (pass_test)
  PDTest_out1(PDTestResFile + "_K" + to_str(K) + "_Itr" + to_str(ItrNum) + ".csv",
	      N, TraceNum, pass_test);

  // # Output the PD test results (loglikeli_train, loglikeli_verify)
  PDTest_out2(PDTestResFile + "_K" + to_str(K) + "_Itr" + to_str(ItrNum) + "_loglikeli.csv",
	      N, TraceNum, VUserNum,
	      loglikeli_train, loglikeli_verify);
}

int main(int argc, char* argv[]){
  if(argc < 3){
    fprintf(stderr, "Usage: %s [Dataset] [City] ([ReqEps (default:1.0)] [Reqk (default:10)] [VUserNum (default:-1)] [TraceNum (default:10)] [Alp (default:200)] [MaxNumTrans (default:100)] [MaxNumVisit (default:100)] [ItrNum (default:100)])\n", argv[0]);
    exit(1);
  }
  // # Dataset (PF/FS)
  string DataSet = argv[1];
  // # City
  string City = argv[2];

  // # Required epsilon in plausible deniability
  double ReqEps = 1.0;
  if(argc >= 4){
    sscanf(argv[3], "%lf", &ReqEps);
  }

  // # Required k in plausible deniability
  int Reqk = 10;
  if(argc >= 5){
    sscanf(argv[4], "%d", &Reqk);
  }

  // # Number of users for verifying PD (-1: all)
  int VUserNum = -1;
  if(argc >= 6){
    sscanf(argv[5], "%d", &VUserNum);
  }

  // # Number of traces per user
  int TraceNum = 10;
  if(argc >= 7){
    sscanf(argv[6], "%d", &TraceNum);
  }

  // # Hyper-hyper parameter alpha
  double Alp = 200;
  string AlpStr = "200";
  if(argc >= 8){
    sscanf(argv[7], "%lf", &Alp);
    AlpStr = argv[7];
  }

  // # Maximum number of transitions per user (-1: infinity)
  int MaxNumTrans = 100;
  if(argc >= 9){
    sscanf(argv[8], "%d", &MaxNumTrans);
  }

  // # Maximum number of POI visits per user (-1: infinity)
  int MaxNumVisit = 100;
  if(argc >= 10){
    sscanf(argv[9], "%d", &MaxNumVisit);
  }

  // # Number of iterations in Gibbs sampling
  int ItrNum = 100;
  if(argc >= 11){
    sscanf(argv[10], "%d", &ItrNum);
  }
  
  // # Data directory
  string DataDir = "../data/" + DataSet + "/";

  // # Training user index file (input)
  string TUserIndexFile = DataDir + "tuserindex_%s.csv";
  // # POI index file (input)
  string POIIndexFile = DataDir + "POIindex_%s.csv";

  // # Model parameter directory
//  string ModelParameterDir = DataDir + "PPMTF_" + City + "_alp" + AlpStr + "_mnt" + to_str(MaxNumTrans) + "_mnv" + to_str(MaxNumVisit) + "_py/";
  string ModelParameterDir = DataDir + "PPMTF_" + City + "_alp" + AlpStr + "_mnt" + to_str(MaxNumTrans) + "_mnv" + to_str(MaxNumVisit) + "/";
  // # Prefix of the model parameter file (input)
  string ModelParameterFile = ModelParameterDir + "modelparameter";
  // # Prefix of the synthesized trace file (input)
  string SynTraceFile = ModelParameterDir + "syntraces";

  // # Prefix of the PD test result file (output)
  string PDTestResFile = ModelParameterDir + "pdtest_res";

  // # Name of the model parameter A
  //string ParamA = "A";

  // # Number of time slots
  int T = -1;
  if(DataSet.find("PF") == 0){T = 30;}
  if(DataSet.find("FS") == 0){T = 12;}
  if(T == -1){
    fprintf(stderr, "Wrong Dataset\n");
    exit(1);
  }

  // # Number of time instants per time slot
  int TimInsNum = -1;
  if(DataSet.find("PF") == 0){TimInsNum = 1;}
  if(DataSet.find("FS") == 0){TimInsNum = 2;}
  assert(TimInsNum >= 0);

  // # Number of columns in model parameters (A, B, C)
  int K = 16;
  // # Threshold for a visit count
  double VisThr = 0;
  // # Minimum value of a visit count
  double VisitDelta = 0.00000001;
  // # Minimum value of a transition count
  double TransDelta = 0.00000001;

  // #################################### Main #####################################
  // # Fix a seed
  seeding(1);

  // # Replace XX with City
  TUserIndexFile = string_replace(TUserIndexFile, City);
  POIIndexFile = string_replace(POIIndexFile, City);

  // # Number of training users --> N
  int N = line_num(TUserIndexFile) - 1;
  // # Number of POIs --> M
  int M = line_num(POIIndexFile) - 1;

  // # Number of users for verifying PD (-1: all)
  if(VUserNum == -1){VUserNum = N-1;}

  // # Read model parameters
  mat_t A, B, C, D;
  tie(A, B, C, D) = ReadModelParameters(ModelParameterFile, K, ItrNum);

  // # Plausible deniability test
  PDTest(A, B, C, D, N, M, T,
	 VUserNum, PDTestResFile,
	 K, ItrNum, TraceNum,
	 ReqEps, Reqk,
	 SynTraceFile,
	 TimInsNum,
	 VisThr, VisitDelta, TransDelta);
  //////////////////pass_test, loglikeli_train, loglikeli_verify = PDTest(A, B, C, D, N, M, T)

//  printf("%s\n", DataSet.c_str());
//  printf("%s\n", City.c_str());
  return 0;
}
