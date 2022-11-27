#include <iostream>
#include <iomanip>
#include <fstream>
#include "geometry.hpp"

using std::cin;
using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::getline;
using std::setw;
using std::string;

int main(int argc,char** argv){
  ifstream geometryfile(argv[1]);
  double coffset = 0;
  auto f=geometry::get_crystfel_geometry(geometryfile,coffset);
  for (double fs,ss;cin >> fs >> ss;){
    string rest;
    getline(cin,rest);
    double x,y,z;
    f(fs,ss,x,y,z=0);
    cout << setw(8) << x << " " << setw(8) << y << " " << setw(8) <<  z
         << " " << rest << endl;
  }
}

