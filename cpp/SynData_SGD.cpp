#include "common.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <random>
#include <map>
#include <cassert>

#include <sys/stat.h>

// eigen
#include <Eigen/Core>

#define rep(i,n) for(int (i)=0;(i)<(n);(i)++)

using namespace Eigen;
using namespace std;

constexpr int BUF_SIZE = 1000;

//using p_t = pair<int, int>;
//using r_t = vector<map<p_t, double> >;
using mat_t = MatrixXd;
//using vec_t = VectorXd;
//using vec_mat_t = vector<mat_t>;

std::mt19937_64 engine;//(seed_gen());
std::uniform_real_distribution<> unifrom_real_dist(0.0, 1.0);

using init_prob_t = vector<double>;

double uniform_real_rand(){
  return unifrom_real_dist(engine);
}

void seeding(unsigned seed = 0){
  //std::random_device seed_gen;
  //engine = std::mt19937_64(seed_gen());
  engine = std::mt19937_64(seed);
}

map<int, string> read_poi_dic(const string& infile){
  map<int, string> res;
  ifstream ifs(infile);
  string line;
  getline(ifs, line); // header
  while(getline(ifs, line)){
    auto row = split(line);
    assert(row.size() >= 3);
    int index;
    sscanf(row[1].c_str(), "%d", &index);
    res[index] = row[2];
  }
  return res;
}

bool dir_exists(const string& path){
  struct stat sb;
  return stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
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

template<typename T>
struct sparse_mat{
  int m_, n_;
  map<pair<int,int>, T> mat_;
  map<int, T> row_sum_;
  sparse_mat(int m, int n):m_(m),n_(n),mat_(),row_sum_(){}
  void set(int i, int j, T val){
    auto key = make_pair(i,j);
    if(row_sum_.find(i) == row_sum_.end()){
      row_sum_[i] = 0;
    }else{
      row_sum_[i] -= mat_[key];
    }
    mat_[key] = val;
    row_sum_[i] += val;
  }
  T get(int i, int j){
    auto key = make_pair(i, j);
    if(mat_.find(key) == mat_.end()){return 0;}
    return mat_[key];
  }
  T row_sum(int i){
    if(row_sum_.find(i) == row_sum_.end()){return 0;}
    return row_sum_[i];
  }
};

using smat_t = sparse_mat<double>;

/*
################## Train parameters in the generative model ###################
# [output1]: init_prob (M-dim vector)
# [output2]: trans_count ({(time_slot, poi_index_from, poi_index_to): count})
# [output3]: trans_prob (T x (M x M matrix))
# [output4]: copy_train_data (N x CopyNum matrix)
 */
tuple<init_prob_t, map<tuple<int,int,int>, int>, vector<smat_t>, map<pair<int,int>, int>>
TrainParam(int M, int T, int TimeType, int CopyNum, int MaxTimInt,
		const string& TrainTraceFile){
  // # Read a training trace file
  vector<int> init_count(M, 0);
  int init_count_sum = 0;
  map<tuple<int,int,int>, int> trans_count;
  map<pair<int,int>, int> copy_train_data;
  {
    string infile = TrainTraceFile;
    ifstream ifs(infile);
    string line;
    getline(ifs, line); // header
    int user_index_prev = -1;
    int poi_index_prev = 0;
    int unixtime_prev = 0;
    int time_ins_prev = 0;
    while(getline(ifs, line)){
      auto lst = split(line);
      int user_index = str2i(lst[0]);
      int poi_index = str2i(lst[1]);
      int unixtime = str2f(lst[3]);
      int ho = str2i(lst[5]);
      // # Time slot and time instant --> time_slot, time_ins
      int time_slot, time_ins;
      if(TimeType == 1){
	int x = str2i(lst[6]);
	int mi = 2;
	if(x >= 40){
	}else if(x >= 20){
	  mi = 1;
	}else{
	  mi = 0;
	}
	time_slot = 3 * (ho - 9) + mi;
	time_ins = time_slot;
      }else if(TimeType == 2){
	time_slot = ho/2;
	time_ins = ho;
      }else{
	fprintf(stderr, "Wrong TimeType.\n");
	exit(1);
      }
      // # Update copy_train_data for the first CopyNum time instants
      if(time_ins < CopyNum){
	copy_train_data[make_pair(user_index, time_ins)] = poi_index;
      }
      // # Update trans_count if the event and the previous event are from the same user
      if(user_index == user_index_prev){
	// # Consider only temporally-continuous locations within MaxTimInt for a transition
	if( (MaxTimInt == -1 || (unixtime - unixtime_prev <= MaxTimInt) ) && (time_ins - time_ins_prev == 1) ){
	  auto key = make_tuple(time_slot, poi_index_prev, poi_index);
	  if(trans_count.find(key) == trans_count.end()){trans_count[key] = 0;}
	  trans_count[key] += 1;
	}
      }
      // # Update init_count if the event is in the first time slot
      if(time_slot == 0){init_count[poi_index] += 1; ++init_count_sum;}
      user_index_prev = user_index;
      poi_index_prev = poi_index;
      unixtime_prev = unixtime;
      time_ins_prev = time_ins;
    }
  }
  
  // # Make an init probability vector --> init_prob
  init_prob_t init_prob(M, 0);
  rep(i, M){
    init_prob[i] = static_cast<double>(init_count[i]) / init_count_sum;
  }
  
  // # Make a count sum vector --> count_sum
  mat_t count_sum = mat_t::Zero(T, M);
  for(const auto& it : trans_count){
    const auto& key = it.first;
    auto& counts = it.second;
    auto& time_slot = get<0>(key);
    auto& poi_index_prev = get<1>(key);
    count_sum(time_slot, poi_index_prev) += counts;
  }
  
  // # Make a transition probability matrix --> trans_prob
  vector<smat_t> trans_prob;
  rep(i, T){trans_prob.push_back(smat_t(M, M));}
  for(const auto& it : trans_count){
    const auto& key = it.first;
    auto& counts = it.second;
    auto& time_slot = get<0>(key);
    auto& poi_index_prev = get<1>(key);
    auto& poi_index = get<2>(key);
    auto cs = count_sum(time_slot, poi_index_prev);
    if(cs > 0){
      trans_prob[time_slot].set(poi_index_prev, poi_index, counts / cs);
    }
  }
  
  // # Assign a uniform distribution for a row of zero-values --> trans_prob
  rep(time_slot, T){
    rep(i, M){
      if(trans_prob[time_slot].row_sum(i) == 0){
	rep(j, M){
	  trans_prob[time_slot].set(i, j, 1.0 / M);
	}
      }
    }
  }
  return forward_as_tuple(init_prob, trans_count, trans_prob, copy_train_data);
}

/*
############################## Synthesize traces ##############################
# [input1]: init_prob (M-dim vector)
# [input2]: trans_prob (T x (M x M matrix))
# [input3]: copy_train_data (N x CopyNum matrix)
# [input4]: poi_dic ({poi_index: category})
 */
void SynTraces(init_prob_t& init_prob,
	       vector<smat_t>& trans_prob,
	       map<pair<int,int>, int>& copy_train_data,
	       map<int, string>& poi_dic,
	       const string& SynTraceFile, int CopyNum,
	       int N, int M, int T, int TraceNum, int TimInsNum){
  char buf[BUF_SIZE];
  sprintf(buf, "%s_cn%d.csv", SynTraceFile.c_str(), CopyNum);
  // # Output header information
  string outfile = buf;
  FILE* fp = fopen(outfile.c_str(), "w");
  fprintf(fp, "user,trace_no,time_slot,time_instant,poi_index,category\n");
  // # For each user
  rep(n, N){
    if(n % 1000 == 0){printf("Synthesized traces of %d users.\n", n);}
    // ########################## Synthesize traces ##########################
    int poi_index_pre = 0;
    // # For each trace
    rep(trace_no, TraceNum){
      // # For each time slot
      rep(t, T){
	// # For each time instant
	rep(ins, TimInsNum){
	  int tim = t * TimInsNum + ins;
	  int poi_index = -1;
	  if(CopyNum == 0 && tim == 0){
	    // # Randomly sample POI from init_prob
	    double rnd = uniform_real_rand();
	    double prob_sum = 0;
	    rep(i, M){
	      prob_sum += init_prob[i];
	      if(prob_sum >= rnd){poi_index = i; break;}
	    }
	  }else if(tim < CopyNum && copy_train_data.find(make_pair(n,tim)) != copy_train_data.end()){
	    // # Copy the POI from the training data
	    poi_index = copy_train_data[make_pair(n,tim)];
	  }else{
	    // # Randomly sample POI from trans_prob
	    // # Transform poi_index_pre into poi_index via trans_vec
	    double rnd = uniform_real_rand();
	    double prob_sum = 0;
	    rep(j, M){
	      prob_sum += trans_prob[t].get(poi_index_pre,j);
	      if(prob_sum >= rnd){poi_index = j; break;}
	    }
	  }
	  assert(poi_index >= 0);
	  // # Output an initial location ([user, trace_no, time_slot, time_instant, poi_index, category])
	  fprintf(fp, "%d,%d,%d,%d,%d,%s\n", n, trace_no, t, ins, poi_index, poi_dic[poi_index].c_str());
	  // # Save the previous poi_index
	  poi_index_pre = poi_index;
	}
      }
    }
  }
  fclose(fp);
}

int main(int argc, char* argv[]){
  if(argc < 3){
    fprintf(stderr, "Usage: %s [Dataset] [City] ([TraceNum (default:10)] [CopyNum (default:0)])\n", argv[0]);
    exit(1);
  }
  // # Dataset (PF/FS)
  string DataSet = argv[1];
  // # City
  string City = argv[2];
  // # Number of traces per user
  int TraceNum = 10;
  if(argc >= 4){sscanf(argv[3], "%d", &TraceNum);}
  // # Number of locations copied from the training trace (DataSet = PF)
  int CopyNum = 0;
  if(DataSet == "PF" && argc >= 5){sscanf(argv[4], "%d", &CopyNum);}

  // # Data directory
  string DataDir = "../data/" + DataSet + "/";
  // # Training user index file (input)
  string TUserIndexFile = DataDir + "tuserindex_%s.csv";
  // # POI index file (input)
  string POIIndexFile = DataDir + "POIindex_%s.csv";
  // # Training trace file (input)
  string TrainTraceFile = DataDir + "traintraces_%s.csv";
  // # Type of time slots (1: 9-19h, 20min, 2: 2 hours, )
  int TimeType = 1;
  if(DataSet.find("PF") == 0){
  }else if(DataSet.find("FS") == 0){
    TimeType = 2;
  }else{
    fprintf(stderr, "Wrong Dataset\n");
    exit(1);
  }
  // # Output directory
  string OutDir = DataDir + "SGD_" + City + "/";
  if(!dir_exists(OutDir)){
    //fprintf(stderr, "%s does not exist.\n", OutDir.c_str());
    mode_t mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
    int res = mkdir(OutDir.c_str(), mode);
  }

  // # Prefix of the synthesized trace file (output)
  string SynTraceFile = OutDir + "syntraces";
  
  // # Number of time instants per time slot
  int TimInsNum = 1;
  if(DataSet.find("PF") == 0){
  }else if(DataSet.find("FS") == 0){
    TimInsNum = 2;
  }else{
    exit(1);
  }
  
  // # Maximum time interval between two temporally-continuous locations (sec) (-1: none)
  int MaxTimInt = -1;
  if(DataSet.find("PF") == 0){
  }else if(DataSet.find("FS") == 0){
    MaxTimInt = 7200;
  }else{
    fprintf(stderr, "Wrong Dataset\n");
    exit(1);
  }
  
  // # Fix a seed
  seeding(1);
  // # Replace %s with City
  TUserIndexFile = string_replace(TUserIndexFile, City);
  POIIndexFile = string_replace(POIIndexFile, City);
  TrainTraceFile = string_replace(TrainTraceFile, City);
  // # Number of training users --> N
  int N = line_num(TUserIndexFile) - 1;
  // # Number of POIs --> M
  int M = line_num(POIIndexFile) - 1;
  // # Number of time slots --> T
  int T = 30;
  if(TimeType == 1){
  }else if(TimeType == 2){
    T = 12;
  }else{
    fprintf(stderr, "Wrong TimeType.\n");
    exit(1);
  }
  // # Read the POI index file --> poi_dic ({poi_index: category})
  auto poi_dic = read_poi_dic(POIIndexFile);
  
  // # Train parameters in the generative model
  init_prob_t init_prob;
  map<tuple<int,int,int>, int> trans_count;
  vector<smat_t> trans_prob;
  map<pair<int,int>, int> copy_train_data;
  //fprintf(stderr, "M =%d, T = %d\n", M, T);
  tie(init_prob, trans_count, trans_prob, copy_train_data) = TrainParam(M, T, TimeType, CopyNum, MaxTimInt, TrainTraceFile);
  
  // # Synthesize traces
  SynTraces(init_prob, trans_prob, copy_train_data, poi_dic,
	    SynTraceFile, CopyNum, N, M, T, TraceNum, TimInsNum);
  return 0;
}
