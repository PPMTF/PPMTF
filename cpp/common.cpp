#include "common.hpp"


std::string string_replace(const std::string& format, const std::string& src){
  constexpr int BUF_SIZE = 1000;
  char buf[BUF_SIZE];
  sprintf(buf, format.c_str(), src.c_str());
  return std::string(buf);
}

int line_num(const std::string& filename){
  std::ifstream f;
  std::string line;
  f.open(filename);
  int count = 0;
  while(getline(f, line)){++count;}
  f.close();
  return count;
}

std::vector<std::string> split(const std::string& line, char del){
  std::istringstream iss(line);
  std::string field;
  std::vector<std::string> res;
  while(getline(iss, field, del)){res.push_back(field);}
  return res;
}
