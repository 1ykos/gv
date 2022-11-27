#include <cmath>
#include <execution>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_map>
#include <sstream>
#include <string>

#include "partiality.hpp"
#include "wmath.hpp"

using std::abs;
using std::array;
using std::cerr;
using std::cin;
using std::cout;
using std::endl;
using std::fill;
using std::fixed;
using std::get;
using std::getline;
using std::ifstream;
using std::isinf;
using std::isnan;
using std::isfinite;
using std::istream;
using std::lower_bound;
using std::make_tuple;
using std::max_element;
using std::nan;
using std::normal_distribution;
using std::numeric_limits;
using std::ofstream;
using std::random_device;
using std::round;
using std::unordered_map;
using std::unordered_set;
using std::setprecision;
using std::setw;
using std::sort;
using std::stod;
using std::streamsize;
using std::string;
using std::stringstream;
using std::swap;
using std::to_string;
using std::transform;
using std::tuple;
using std::vector;
using std::stoull;

using dlib::abs;
using dlib::cholesky_decomposition;
using dlib::identity_matrix;
using dlib::is_finite;
using dlib::length;
using dlib::length_squared;
using dlib::matrix;
using dlib::normalize;
using dlib::ones_matrix;
using dlib::squared;
using dlib::sum;
using dlib::tmp;
using dlib::trans;
using dlib::zeros_matrix;

using partiality::IDX;
using partiality::crystl;
using partiality::deserialize_crystls;
using partiality::deserialize_crystl;
using partiality::deserialize_sources;
using partiality::predict;
using partiality::predict_integrated;
using partiality::source;

using whash::patchmap;

using wmath::clip;
using wmath::mean_variance;
using wmath::signum;
using wmath::pow;

constexpr double pi          = 3.14159265358979323846;

int main(int argc,char** argv) {
  const double a = 1.0/16;
  // this is a property of the detector
  const double g = 9.9866655;
  //patchmap<size_t,size_t> reduce_n;
  for (size_t counter=0;cin;++counter) {
    auto sources = deserialize_sources(cin);
    if (!cin) break;
    auto crystl  = deserialize_crystl(cin);
    if (!cin) break;
    double b = pow(g,-2),c = pow(g,-2);
    cin.read(reinterpret_cast<char*>(&b),sizeof(double));
    if (!cin) break;
    cin.read(reinterpret_cast<char*>(&c),sizeof(double));
    if (!cin) break;
    //crystl.a*=1e3;
    //crystl.U = trans(crystl.U);
    //crystl.R = trans(crystl.R);
    //cerr << crystl.R;
    //cerr << crystl.peak;
    //cerr << crystl.mosaicity << " " << crystl.strain << " " << crystl.a << " "
    //     << crystl.b << endl;
    uint64_t m;
    cin.read(reinterpret_cast<char*>(&m),sizeof(uint64_t));
    int32_t h,k,l;
    float i,s;
    for (size_t j=0;j!=m;++j) {
      cin.read(reinterpret_cast<char*>(&h),sizeof(int32_t));
      cin.read(reinterpret_cast<char*>(&k),sizeof(int32_t));
      cin.read(reinterpret_cast<char*>(&l),sizeof(int32_t));
      cin.read(reinterpret_cast<char*>(&i),sizeof(float));
      cin.read(reinterpret_cast<char*>(&s),sizeof(float));
      //cerr << h << " " << k << " " << l << " " << i << " " << s << endl;
      const IDX idx{h,k,l};
      const matrix<double,3,1> dhkl{1.0*h,1.0*k,1.0*l};
      const matrix<double,3,1> x = crystl.R*dhkl;
      //cerr << h << " " << k << " " << l << " " << i << " " << s << " "
      //     << length(x) << endl;
      //if (length(x)>4.25) continue;
      const double lambda = -(2*x(2))/length_squared(x);
      const double k0 = 1.0/0.132835;
      matrix<double,3,1> y = x;
      y(2)+=k0;
      const double length_kout = length(y);
      y = k0*normalize(y);
      y(2)-=k0;
      const double offset = length(x-y);
      //cout << trans(x);
      //cout << trans(y);
      cout << length(x) << " " << ((length_kout>k0)?1:-1)*offset
           << " " << i << " " << s << " " << lambda << endl;
      //if (length(x)>6) continue;
      //if (length(x)>2) continue;
      //
      //
    }
  }
}
