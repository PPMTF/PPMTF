#include "common.hpp"
#include <string>
#include <cassert>
#include <map>
#include <vector>
#include <tuple>
#include <algorithm>
#include <random>
#include <math.h>
#include"ProcessingTime.hpp"

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
std::uniform_real_distribution<> unifrom_real_dist(0.0, 1.0);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double uniform_real_rand(){
  return unifrom_real_dist(engine);
}

void seeding(unsigned seed = 0){
  //std::random_device seed_gen;
  //engine = std::mt19937_64(seed_gen());
  engine = std::mt19937_64(seed);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

map<int, string> read_poi_dic(const string& infile){
  map<int, string> res;
  ifstream ifs(infile);
  string line;
  getline(ifs, line); // header
  while(getline(ifs, line)){
    auto row = split(line);
    assert(row.size() >= 3);
    int id, index;
    sscanf(row[0].c_str(), "%d", &id);
    sscanf(row[1].c_str(), "%d", &index);
    string category = row[2];
    res[index] = category;
    //printf("<%d><%d><%s>\n", id, index, category.c_str());
  }
  return res;
}

mat_t read_train_tensor(const string& infile, int m, int n){
  mat_t res = mat_t::Zero(m, n);
  ifstream ifs(infile);
  string line;
  while(getline(ifs, line)){
    auto row = split(line);
    assert(row.size() >= 3);
    int from, to;
    sscanf(row[1].c_str(), "%d", &from);
    sscanf(row[2].c_str(), "%d", &to);
    res(from, to) = 1;
  }
  return res;
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
  //read_file(buf);
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

int sample_from_dist(const vec_t& df){
  int n = df.size();
  double rnd = uniform_real_rand();
  double prob_sum = 0;
  for(int i = 0; i < n; ++i){
    prob_sum += df(i);
    if(prob_sum >= rnd){return i;}
  }
  assert(0);
}


/*
  ############################## Synthesize traces ##############################
  # [input1]: A (N x K matrix)
  # [input2]: B (M x K matrix)
  # [input3]: D (T x K matrix)
  # [input4]: N -- Number of users
  # [input5]: M -- Number of POIs
  # [input6]: T -- Number of time slots
  # [input7]: poi_dic ({poi_index: category})
 */
void SynTraces(mat_t& A,
	       mat_t& B,
	       mat_t& C, // added
	       mat_t& D,
	       int N, int M, int T,
	       map<int, string>& poi_dic,
	       int K, 
	       int ReadTrans,
	       int ReadVisit,
	       double VisThr,
	       double VisitDelta,
	       double TransDelta,
	       int TraceNum,
	       int TimInsNum,
	       const string& SynTraceFile,
	       int ItrNum,
	       const string& TrainTransTensorFile,
	       const string& TrainVisitTensorFile){
  char buf[BUF_SIZE];
  sprintf(buf, "%s_Itr%d.csv", SynTraceFile.c_str(), ItrNum);
  string outfile(buf);
  FILE* fp = fopen(outfile.c_str(), "w");

  // Output header information
  fprintf(fp, "user,trace_no,time_slot,time_instant,poi_index,category,loglikeli\n");

  // Read transitions from TrainTransTensorFile --> trans
  mat_t trans = mat_t::Zero(M, M);
  if(ReadTrans == 1){trans = read_train_tensor(TrainTransTensorFile, M, M);}

  // Read visits from TrainVisitTensorFile --> visit
  mat_t visit = mat_t::Zero(T, M);
  if(ReadVisit == 1){visit = read_train_tensor(TrainVisitTensorFile, T, M);}
  
  // For each user
  for(int n = 0; n < N; ++n){
    //    printf("%d\n", n);
    // Initialization
    mat_t time_poi_dist = mat_t::Zero(T, M);
    mat_t time_poi_dist_sum = mat_t::Zero(T, 1);
    mat_t prop_mat = mat_t::Zero(M, M);
    mat_t trans_vec = mat_t::Zero(M, 1);

    // ################### Calculate the POI distributions ###################
    for(int t = 0; t < T; ++t){
      mat_t ad = A.row(n).array() * D.row(t).array();
      for(int i = 0; i < M; ++i){
	// Elements in a sampled visit tensor --> time_poi_dist
	time_poi_dist(t, i) = (ad.array() * B.row(i).array()).sum();
	// Assign VisitDelta for an element whose value is less than VisThr
	if(time_poi_dist(t, i) < VisThr){
	  time_poi_dist(t, i) = VisitDelta;
	}
	// Assign VisitDelta if there is no visits for time t & user i
	if(ReadVisit == 1 && visit(t,i) == 0){
	  time_poi_dist(t, i) = VisitDelta;
	}
      }
    }
    // Normalize time_poi_dist
    for(int t = 0; t < T; ++t){
      time_poi_dist_sum(t, 0) = time_poi_dist.row(t).sum();
      if(time_poi_dist_sum(t, 0) > 0){
	time_poi_dist.row(t) /= time_poi_dist_sum(t, 0);
      }else{
	fprintf(stderr, "Error: All probabilities are 0 for user %d and time %d\n", n, t);
	exit(1);
      }
    }

    // #################### Calculate the proposal matrix ####################
    //    puts("Calculating the proposal matrix ...");
    for(int i = 0; i < M; ++i){
      mat_t ab = A.row(n).array() * B.row(i).array();
      // Elements in a sampled transition tensor (assign TransDelta for a small transition count) --> prop_mat
      for(int j = 0; j < M; ++j){
	// prop_mat[i,j] = max(np.sum(ab * C[j, :]), TransDelta)
	prop_mat(i,j) = max((ab.array() * C.row(j).array()).sum(), TransDelta);
	// Assign TransDelta if there is no transitions between i and j
	if(ReadTrans == 1 && trans(i,j) == 0){
	  prop_mat(i,j) = TransDelta;
	}
      }
      // Normalize prop_mat
      double row_sum = prop_mat.row(i).sum();
      prop_mat.row(i) /= row_sum;
    }

    // ########################## Synthesize traces ##########################
    //    puts("Synthesizing traces ...");
    int poi_index_pre = 0;
    // For each trace
    for(int trace_no = 0; trace_no < TraceNum; ++trace_no){
	  double loglikeli = 0.0;
      // For each time slot
      for(int t = 0; t < T; ++t){
	// For each time instant
	for(int ins = 0; ins < TimInsNum; ++ins){
	  int poi_index = -1;
	  // Initial time slot and initial event
	  if(t == 0 && ins == 0){
	    // Randomly sample POI from the POI distribution
	    poi_index = sample_from_dist(time_poi_dist.row(t));
		loglikeli += log(time_poi_dist(t, poi_index));
	  }else{
	    // ##### Transform poi_index_pre into poi_index via MH (Metropolis-Hastings) ######
	    // Calculate the transition vector --> trans_vec
	    trans_vec(poi_index_pre,0) = 0;
	    for(int j = 0; j < M; ++j){
	      if(poi_index_pre != j){
		double alpha = (time_poi_dist(t, j) * prop_mat(j, poi_index_pre)) / (time_poi_dist(t, poi_index_pre) * prop_mat(poi_index_pre, j));
		trans_vec(j,0) = prop_mat(poi_index_pre, j) * min(1.0, alpha);
	      }
	    }
	    double row_sum = trans_vec.sum();
	    trans_vec(poi_index_pre,0) = 1 - row_sum;

	    // Transform poi_index_pre into poi_index via trans_vec
	    poi_index = sample_from_dist(trans_vec);
		loglikeli += log(trans_vec(poi_index, 0));
	  }
	  // Output an initial location ([user, trace_no, time_slot, time_instant, poi_index, category])
	  fprintf(fp, "%d,%d,%d,%d,%d,%s,%f\n", n, trace_no, t, ins, poi_index, poi_dic[poi_index].c_str(), loglikeli);

	  // Save the previous poi_index
	  poi_index_pre = poi_index;
	}
      }
    }
  }
  fclose(fp);
}

int main(int argc, char *argv[]){
{
  ProcessingTime pt("SynData_PPMTF_process");
  if(argc < 3){
      printf("Usage: SynData_PPMTF_time [Dataset] [City] ([TraceNum (default:10)] [Alp (default:200)] [MaxNumTrans (default:100)] [MaxNumVisit (default:100)] [ItrNum (default:100)])\n");
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

  // Number of traces per user
  int TraceNum = 10;
  if(argc >= 4){
    TraceNum = atoi(argv[3]);
  }
  // Hyper-hyper parameter alpha
  string AlpStr = "200";
  if(argc >= 5){
	  AlpStr = argv[4];
  }
  // Maximum number of transitions per user (-1: infinity)
  int MaxNumTrans = 100;
  if(argc >= 6){
	  MaxNumTrans = atoi(argv[5]);
  }
  // Maximum number of POI visits per user (-1: infinity)
  int MaxNumVisit = 100;
  if(argc >= 7){
	  MaxNumVisit = atoi(argv[6]);
  }
  // Number of iterations in Gibbs sampling
  int ItrNum = 100;
  if(argc >= 8){
	  ItrNum = atoi(argv[7]);
  }

  // Training user index file (input)
  string TUserIndexFile = "../data/" + Dataset + "/tuserindex_%s.csv";
  // POI index file (input)
  string POIIndexFile = "../data/" + Dataset + "/POIindex_%s.csv";
  // Training transition tensor file (input)
  string TrainTransTensorFile = "../data/" + Dataset + "/traintranstensor_%s.csv";
  // Training visit tensor file (input)
  string TrainVisitTensorFile = "../data/" + Dataset + "/trainvisittensor_%s.csv";

  // Prefix of the model parameter file (output)
  const string OutDir = "../data/" + Dataset + "/PPMTF_" + City + "_alp" + AlpStr + "_mnt" + std::to_string(MaxNumTrans) + "_mnv" + std::to_string(MaxNumVisit);
  // Prefix of the model parameter file (output)
  const string ModelParameterFile = OutDir + "/modelparameter";
  // Prefix of the synthesized trace file (output)
  const string SynTraceFile = OutDir + "/syntraces";

  // Name of the model parameter A
  //const string ParamA = "A";

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
  // Threshold for a visit count
  const double VisThr = 0;
  // Minimum value of a visit count
  const double VisitDelta = 0.00000001;
  // Minimum value of a transition count
  const double TransDelta = 0.00000001;
  // Read trans from TrainTransTensorFile (1:yes, 0:no)
  const int ReadTrans = 0;
  // Read visits from TrainVisitTensorFile (1:yes, 0:no)
  const int ReadVisit = 0;

  // Number of time instants per time slot
  int TimInsNum;
  if(Dataset.find("PF") == 0){
	  TimInsNum = 1;
  }else if(Dataset.find("FS") == 0){
	  TimInsNum = 2;
  }else{
	  cout << "Wrong Dataset\n";
	  exit(-1);
  }
//  const int TimInsNum = 1;
  
  // Replace %s with City
  TUserIndexFile = string_replace(TUserIndexFile, City);
  POIIndexFile = string_replace(POIIndexFile, City);
  TrainTransTensorFile = string_replace(TrainTransTensorFile, City);
  TrainVisitTensorFile = string_replace(TrainVisitTensorFile, City);

  // Number of training users --> N
  int N = line_num(TUserIndexFile) - 1;
  // Number of POIs --> M
  int M = line_num(POIIndexFile) - 1;

  // Read the POI index file --> poi_dic ({poi_index: category})
  auto poi_dic = read_poi_dic(POIIndexFile);

  // Read model parameters
  mat_t A, B, C, D;
  tie(A, B, C, D) = ReadModelParameters(ModelParameterFile, K, ItrNum);

  // Synthesize traces
  SynTraces(A, B, C, D, N, M, T, poi_dic,
	    K, ReadTrans, ReadVisit, VisThr, VisitDelta, TransDelta, TraceNum, TimInsNum, SynTraceFile, ItrNum, TrainTransTensorFile, TrainVisitTensorFile);

  return 0;
}
}
