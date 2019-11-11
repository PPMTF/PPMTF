#ifndef INCLUDED_COMMON_H
#define INCLUDED_COMMON_H

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

std::string string_replace(const std::string& format, const std::string& src);
int line_num(const std::string& filename);
std::vector<std::string> split(const std::string& line, char del = ',');

#endif // INCLUDED_COMMON_H
