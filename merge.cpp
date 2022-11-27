#include <cmath>
#include <execution>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_map>
#include <sstream>
#include <string>

#include <dlib/optimization.h>

#include "asu.hpp"
#include "encode.hpp"
#include "geometry.hpp"
#include "partiality.hpp"
#include "wmath.hpp"
#include "wmath_optimisation.hpp"

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
using wmath::count_stop_strategy;
using wmath::mean_variance;
using wmath::signum;
using wmath::pow;

using SYMMETRY::decode;
using SYMMETRY::get_point_group;
using SYMMETRY::reduce_encode;

constexpr double pi          = 3.14159265358979323846;

double inline universal_distribution(const double x)
{
  const double a=0.000508212398787459;
  return a/(pi*(1+a*a*x*x));
}

tuple<double,tuple<double,double>> nlog_exponential(
    const double x,
    const double l
    )
{
  if (x<0) return {0,{0,0}};
  const double v = x/l + log(l);
  return {v,{1.0/l,1/l-x/pow(l,2u)}};
}

inline tuple<double,matrix<double,2,1>,matrix<double,2,2>> gauss
(
  const double x,
  const double m,
  const double v
)
{
  const double e = pow((x-m),2)/v;
        double t = exp(-0.5*e)/(sqrt(2*pi*v));
  if (isnan(t)||isinf(t)) t = 0;
  return
  {
    t,
    {
      (x-m)/v*t,
      t/(2*v)*(e-1)
    },
    {
      t*(pow((x-m)/v,2)-1.0/v),
      t*(pow((x-m)/v,3)-3*(x-m)/pow(v,2))/2,
      t*(pow((x-m)/v,3)-3*(x-m)/pow(v,2))/2,
      t*0.25*(pow(e,2)-6*e+3)/pow(v,2)
    }
  };
}

void test_gauss() {
  const double e = 1e-3;
  const double x = 1.7;
  const double m = 3.1;
  const double v = 1.3;
  cerr << get<1>(gauss(x,m,v))(0) << " "
       << (get<0>(gauss(x,m+e,v))-get<0>(gauss(x,m-e,v)))/(2*e) << endl;
  cerr << get<1>(gauss(x,m,v))(1) << " "
       << (get<0>(gauss(x,m,v+e))-get<0>(gauss(x,m,v-e)))/(2*e) << endl;
  cerr << get<2>(gauss(x,m,v))(0,0) << " "
       << (get<1>(gauss(x,m+e,v))(0)-get<1>(gauss(x,m-e,v))(0))/(2*e) << endl;
  cerr << get<2>(gauss(x,m,v))(0,1) << " "
       << (get<1>(gauss(x,m,v+e))(0)-get<1>(gauss(x,m,v-e))(0))/(2*e) << endl;
  cerr << get<2>(gauss(x,m,v))(1,1) << " "
       << (get<1>(gauss(x,m,v+e))(1)-get<1>(gauss(x,m,v-e))(1))/(2*e) << endl;
}

template<class outlier>
const inline tuple<double,matrix<double,3,1>,matrix<double,3,3>> llg
(
  const double x,
  const double m,
  const double v,
  const double a,
  const outlier& o
)
{
  const double p_o  = o(abs(x));
  const auto [p_g,dg,hg]  = gauss(x,m,v+0.5);
  const double p    = a*p_o + (1-a)*p_g;
  const double llg  = -log(p);
  if (isnan(p)||llg>44||llg<0||abs(x)>1.8446744e+19)
    return {44,zeros_matrix<double>(3,1),zeros_matrix<double>(3,3)};
  matrix<double,3,1> J;
  J(0) = (1-a)*dg(0);
  J(1) = (1-a)*dg(1);
  J(2) = p_o - p_g; 
  matrix<double,3,3> H;
  H(0,0) = (1-a)*hg(0,0);
  H(0,1) = H(1,0) = (1-a)*hg(0,1);
  H(0,2) = H(2,0) = -dg(0);
  H(1,1) = (1-a)*hg(1,1);
  H(1,2) = H(2,1) = -dg(1);
  H(2,2) = 0;
  H = (J*trans(J)/p-H)/p;
  J = -J/p;
  return {llg,J,H};
}

void test_llg() {
  const double eps = 1e-2;
  const double d = 10;
  const matrix<double,3,1> x{17,pow(10,2),0.2};
  const auto& o = universal_distribution;
  const auto [llg_value,nabla,hessian] = llg(d,x(0),x(1),x(2),o);
  for (size_t i=0;i!=3;++i) {
    matrix<double,3,1> xp = x;xp(i)+=eps;
    matrix<double,3,1> xm = x;xm(i)-=eps;
    cerr << nabla(i) << " "
         << (get<0>(llg(d,xp(0),xp(1),xp(2),o))
            -get<0>(llg(d,xm(0),xm(1),xm(2),o)))/(2*eps) << endl;
    for (size_t j=0;j!=3;++j) {
      cerr << hessian(i,j) << " "
           << (get<1>(llg(d,xp(0),xp(1),xp(2),o))(j)
              -get<1>(llg(d,xm(0),xm(1),xm(2),o))(j))/(2*eps) << " ";
    }
    cerr << endl;
  }
}

template<long n>
void set_crystal_vector(
    const struct crystl& crystl,
    matrix<double,n,1>& x
)
{
  x( 0) = crystl.R(0,0);
  x( 1) = crystl.R(0,1);
  x( 2) = crystl.R(0,2);
  x( 3) = crystl.R(1,0);
  x( 4) = crystl.R(1,1);
  x( 5) = crystl.R(1,2);
  x( 6) = crystl.R(2,0);
  x( 7) = crystl.R(2,1);
  x( 8) = crystl.R(2,2);
  x( 9) = crystl.mosaicity;
  x(10) = crystl.peak(0,0);
  x(11) = crystl.peak(0,1);
  x(12) = crystl.peak(0,2);
  x(13) = crystl.peak(1,1);
  x(14) = crystl.peak(1,2);
  x(15) = crystl.peak(2,2);
  x(16) = crystl.strain;
  x(17) = crystl.a;
  x(18) = crystl.b;
}

template<long n>
void set_crystal_from_vector(
    const matrix<double,n,1>& x,
    struct crystl& crystl
)
{
   crystl.R(0,0)    = x( 0);
   crystl.R(0,1)    = x( 1);
   crystl.R(0,2)    = x( 2);
   crystl.R(1,0)    = x( 3);
   crystl.R(1,1)    = x( 4);
   crystl.R(1,2)    = x( 5);
   crystl.R(2,0)    = x( 6);
   crystl.R(2,1)    = x( 7);
   crystl.R(2,2)    = x( 8);
   crystl.mosaicity = x( 9);
   crystl.peak(0,0) = x(10);
   crystl.peak(0,1) = x(11);
   crystl.peak(0,2) = x(12);
   crystl.peak(1,1) = x(13);
   crystl.peak(1,2) = x(14);
   crystl.peak(2,2) = x(15);
   crystl.strain    = x(16);
   crystl.a         = x(17);
   crystl.b         = x(18);
}

tuple<
  matrix<double,2,1>,matrix<double,6,2>,matrix<double,6,6>> prediction_mv(
    const double g,
    const double flx,
    const double wvn,
    const double bnd,
    const double b,
    const double c
)
{
  matrix<double,2,1> mv;
  mv(0) = flx*wvn*g; // there are no second derivatives to be found here
  mv(1) = pow(g,2)*(b+c*abs(flx))*abs(flx)*(pow(wvn,2)+pow(bnd,2));
  matrix<double,6,2> J;
  J(0,0) = flx*wvn;                                                  // mu_dg
  J(1,0) = wvn*g;                                                    // mu_dflx
  J(2,0) = flx*g;                                                    // mu_dwvn
  J(3,0) = 0;                                                        // mu_dbnd
  J(4,0) = 0;                                                        // mu_db
  J(5,0) = 0;                                                        // mu_dc
  J(0,1) = 2*g*(b+c*abs(flx))*abs(flx)*(pow(wvn,2)+pow(bnd,2));      // var_dg
  J(1,1) =
    signum(flx)*(pow(g,2)*(pow(wvn,2)+pow(bnd,2))*(2*c*abs(flx)+b)); // var_dflx
  J(2,1) = 2*abs(flx)*(c*abs(flx)+b)*pow(g,2)*wvn;                   // var_dwvn
  J(3,1) = 2*bnd*abs(flx)*(c*abs(flx)+b)*pow(g,2);                   // var_dbnd
  J(4,1) = abs(flx)*pow(g,2)*(pow(wvn,2)+pow(bnd,2));                // var_db
  J(5,1) = pow(flx,2)*pow(g,2)*(pow(wvn,2)+pow(bnd,2));              // var_dc
  matrix<double,6,6> H;
  H(0,0) = 2*abs(flx)*(c*abs(flx)+b)*(pow(wvn,2)+pow(bnd,2));
  H(0,1) = H(1,0) = 2*signum(flx)*(c*abs(flx)+b)*g*(pow(wvn,2)+pow(bnd,2))+2*c*flx*g*(pow(wvn,2)+pow(bnd,2));
  H(0,2) = H(2,0) = 4*abs(flx)*(c*abs(flx)+b)*g*wvn;
  H(0,3) = H(3,0) = 4*bnd*abs(flx)*(c*abs(flx)+b)*g;
  H(0,4) = H(4,0) = 2*abs(flx)*g*(pow(wvn,2)+pow(bnd,2));
  H(0,5) = H(5,0) = 2*pow(flx,2)*g*(pow(wvn,2)+pow(bnd,2));
  H(1,1) = 2*c*pow(g,2)*(pow(wvn,2)+pow(bnd,2));
  H(1,2) = H(2,1) = 2*pow(g,2)*(wvn*signum(flx)*(c*abs(flx)+b)+wvn*c*flx);
  H(1,3) = H(3,1) = 2*pow(g,2)*(bnd*signum(flx)*(c*abs(flx)+b)+bnd*c*flx); 
  H(1,4) = H(4,1) = signum(flx)*pow(g,2)*(pow(wvn,2)+pow(bnd,2));
  H(1,5) = H(5,1) = 2*flx*pow(g,2)*(pow(wvn,2)+pow(bnd,2));
  H(2,2) = 2*abs(flx)*(c*abs(flx)+b)*pow(g,2);
  H(2,3) = H(3,2) = 0;
  H(2,4) = H(3,2) = 2*abs(flx)*pow(g,2)*wvn;
  H(2,5) = H(3,2) = 2*pow(flx,2)*pow(g,2)*wvn;
  H(3,3) = 2*abs(flx)*(c*abs(flx)+b)*pow(g,2);
  H(3,4) = H(4,3) = 2*bnd*abs(flx)*pow(g,2);
  H(3,5) = H(5,3) = 2*bnd*pow(flx,2)*pow(g,2);
  H(4,4) = H(4,5) = H(5,4) = H(5,5) = 0;
  return {mv,J,H};
}

struct prediction_proposer{
  std::mt19937 gen;
  std::normal_distribution<double> d;
  const inline matrix<double,19,1> operator()(matrix<double,19,1> x) {
    //cerr << "proposing" << endl;
    const double epsilon = 5e-6;
    struct crystl crystl;
    set_crystal_from_vector(x,crystl);
    {
      double eps = epsilon*pow(det(crystl.R),1.0/3);
      for (size_t j=0;j!=3;++j) {
        for (size_t i=0;i!=3;++i) {
          crystl.R(j,i)+=eps*d(gen);
        }
      }
    }
    {
      double eps = epsilon*crystl.mosaicity+1e-8;
      crystl.mosaicity+=eps*d(gen);
    }
    {
      double eps = epsilon*pow(det(crystl.peak),1.0/3);
      for (size_t j=0;j!=3;++j) {
        for (size_t i=j;i!=3;++i) {
          crystl.peak(j,i)+=eps*d(gen);
        }
      }
    }
    {
      double eps = epsilon*abs(crystl.strain)+1e-9;
      crystl.strain+=eps*d(gen);
    }
    {
      double eps = epsilon*crystl.a;
      crystl.a+=eps*d(gen);
    }
    {
      double eps = 10*epsilon*pow(det(crystl.peak),2.0/3);
      crystl.b+=eps*d(gen);
    }
    set_crystal_vector(crystl,x);
    return x;
  }
};

struct prediction_bc_proposer{
  prediction_proposer propose;
  const inline matrix<double,21,1> operator()(matrix<double,21,1> x) {
    matrix<double,19,1> _x;
    for (size_t i=0;i!=19;++i) _x(i) = x(i);
    _x = propose(_x);
    for (size_t i=0;i!=19;++i) x(i) = _x(i);
    const double epsilon = 1e-6;
    double eps;
    eps = epsilon*abs(x(19))+epsilon;
    x(19)+=eps*propose.d(propose.gen);
    x(19) = abs(x(19));
    eps = epsilon*abs(x(20))+epsilon;
    x(20)+=eps*propose.d(propose.gen);
    x(20) = abs(x(20));
    return x; 
  }
};

struct initial_scaling_target {
  const double& g;
  const vector<struct source>& sources;
  const vector<tuple<IDX,double,double>>& data;
  //const inline tuple<double,matrix<double,19,1>>
  const double
  operator()(const matrix<double,19,1>& x) const {
    struct crystl crystl;
    set_crystal_from_vector(x,crystl);
    double value = 0;
    matrix<double,19,1> J = zeros_matrix<double>(19,1);
    for (auto it=data.begin();it!=data.end();++it) {
      const auto [flx,wvn,bnd] =
        predict_integrated(
            get<0>(*it),
            sources,
            crystl
            );
      const auto [mv,_nabla_mv,_hessian_v] =
        prediction_mv(1.0,flx,wvn,bnd,0.0,0.0);
      const auto [llg,_nabla_exp] =
        nlog_exponential(abs(get<1>(*it)),g+sqrt(get<2>(*it))/2
            +(isnan(mv(0))?0:mv(0)));
      value += llg;
    }
    return value;
  }
};

struct merge_target{
  const double& a;
  const vector<tuple<
    size_t, // crystal
    IDX,    // index
    double, // intensity
    double, // variance
    double, // flx
    double, // wvn
    double, // bnd
    double, // b (error model)
    double  // c (error model)
  >>& data;
  const tuple<double,matrix<double,1,1>,matrix<double,1,1>>
  operator()(const matrix<double,1,1>& x) const {
    matrix<double,1,1> J{0};
    matrix<double,1,1> H{0};
    double value = 0;
    // number,intensity,variance,scalefactor
    for (const auto& [n,idx,i,v,flx,wvn,bnd,b,c] : data) {
      const auto [mv,_nabla_mv,_hessian_v] = // hessian_m = 0
        prediction_mv(1.0,x(0)*flx,wvn,bnd,b,c);
      matrix<double,2,1> nabla_mv;
      nabla_mv(0) = flx*_nabla_mv(1,0); // m_dx
      nabla_mv(1) = flx*_nabla_mv(1,1); // v_dx
      const double hessian_v = pow(flx,2)*_hessian_v(1,1);
      const auto [l,nabla,_hessian] =
        llg(
          i,
          mv(0),
          mv(1)+v,
          a,
           universal_distribution
        );
      matrix<double,2,2> hessian;
      hessian(0,0) = _hessian(0,0);
      hessian(0,1) = _hessian(0,1);
      hessian(1,0) = _hessian(1,0);
      hessian(1,1) = _hessian(1,1);
      J(0)+= nabla_mv(0)*nabla(0)+nabla_mv(1)*nabla(1);
      {
      const double tmp = trans(nabla_mv)*hessian*nabla_mv+hessian_v*nabla(1);
      if (isfinite(tmp)) H(0)+= tmp;
      }
      value += l;
    }
    if (isnan(x(0))||isinf(x(0))||isnan(value))
      return {1e300,zeros_matrix<double>(1,1),zeros_matrix<double>(1,1)};
    return {value,J,H};
  }
};

struct refinement_target {
  const double& a;
  const size_t& pointgroup;
  const vector<struct source>& sources;
  //const patchmap<size_t,size_t>& reduce_n;
  const unordered_map<size_t,size_t>& reduce_n;
  const vector<tuple<IDX,double,double>>& data;
  const vector<double>& intensities;
  const double
  operator()(const matrix<double,21,1>& x) const {
    struct crystl crystl;
    set_crystal_from_vector(x,crystl);
    const double b = abs(x(19));
    const double c = abs(x(20));
    double value = 0;
    matrix<double,21,1> J = zeros_matrix<double>(21,1);
    for (const auto [idx,i,v] : data) {
      size_t reduced = reduce_encode
        (get<0>(idx),get<1>(idx),get<2>(idx),pointgroup);
      if (reduce_n.count(reduced)==0) continue;
      const double m = intensities[reduce_n.at(reduced)];
      const auto [flx,wvn,bnd] =
        predict_integrated(
            idx,
            sources,
            crystl
            );
      const auto [mv,_nabla_mv,_hessian_v] =
        prediction_mv(1.0,m*flx,wvn,bnd,b,c);
      const auto [llg_value,_llg_nabla,_llg_hessian] =
        llg(i,mv(0),v+mv(1),a,universal_distribution);
      value += llg_value;
    }
    return value;
  }
};

struct cell_proposer{
  std::mt19937 gen;
  std::normal_distribution<double> d;
  const inline matrix<double,9,1> operator()(matrix<double,9,1> x) {
    matrix<double,3,3> R;
    R(0,0) = x(0);
    R(0,1) = x(1);
    R(0,2) = x(2);
    R(1,0) = x(3);
    R(1,1) = x(4);
    R(1,2) = x(5);
    R(2,0) = x(6);
    R(2,1) = x(7);
    R(2,2) = x(8);
    const double epsilon = 3e-7;
    double eps = epsilon*pow(det(R),1.0/3);
    for (size_t i=0;i!=9;++i) x(i)+=eps*d(gen);
    return x;
  }
};

struct cell_refinement_target {
  const double& a;
  const size_t& pointgroup;
  const vector<struct source>& sources;
  //const patchmap<size_t,size_t>& reduce_n;
  const unordered_map<size_t,size_t>& reduce_n;
  const vector<tuple<IDX,double,double>>& data;
  const vector<double>& intensities;
  const struct crystl& old_crystl;
  const double& b;
  const double& c;
  const double
  operator()(const matrix<double,9,1>& x) const {
    struct crystl crystl = old_crystl;
    crystl.R(0,0) = x(0);
    crystl.R(0,1) = x(1);
    crystl.R(0,2) = x(2);
    crystl.R(1,0) = x(3);
    crystl.R(1,1) = x(4);
    crystl.R(1,2) = x(5);
    crystl.R(2,0) = x(6);
    crystl.R(2,1) = x(7);
    crystl.R(2,2) = x(8);
    double value = 0;
    for (const auto [idx,i,v] : data) {
      size_t reduced = reduce_encode
        (get<0>(idx),get<1>(idx),get<2>(idx),pointgroup);
      if (reduce_n.count(reduced)==0) continue;
      const double m = intensities[reduce_n.at(reduced)];
      const auto [flx,wvn,bnd] =
        predict_integrated(
            idx,
            sources,
            crystl
            );
      const auto [mv,_nabla_mv,_hessian_v] =
        prediction_mv(1.0,m*flx,wvn,bnd,b,c);
      const auto [llg_value,_llg_nabla,_llg_hessian] =
        llg(i,mv(0),v+mv(1),a,universal_distribution);
      value += llg_value;
    }
    //cerr << value << endl;
    return value;
  }
};

struct peak_proposer{
  std::mt19937 gen;
  std::normal_distribution<double> d;
  const inline matrix<double,11,1> operator()(matrix<double,11,1> x) {
    //cerr << "proposing" << endl;
    const double epsilon = 1e-3;
    matrix<double,3,3> peak;
    peak(0,0) = x(0);
    peak(0,1) = x(1);
    peak(0,2) = x(2);
    peak(1,0) = x(3);
    peak(1,1) = x(4);
    peak(1,2) = x(5);
    peak(2,0) = x(6);
    peak(2,1) = x(7);
    peak(2,2) = x(8);
    {
      double eps = epsilon*pow(det(peak),1.0/3);
      for (size_t i=0;i!=9;++i) x(i)+=eps*d(gen);
    }
    {
      double eps = epsilon*abs(x( 9))+1e-8;
      x( 9)+=eps*d(gen);
    }
    {
      double eps = epsilon*abs(x(10))+1e-8;
      x(10)+=eps*d(gen);
    }
    return x;
  }
};

struct peak_refinement_target {
  const double& a;
  const size_t& pointgroup;
  const vector<struct source>& sources;
  //const patchmap<size_t,size_t>& reduce_n;
  const unordered_map<size_t,size_t>& reduce_n;
  const vector<tuple<IDX,double,double>>& data;
  const vector<double>& intensities;
  const struct crystl& old_crystl;
  const double& b;
  const double& c;
  const double
  operator()(const matrix<double,11,1>& x) const {
    struct crystl crystl = old_crystl;
    crystl.peak(0,0) = x( 0);
    crystl.peak(0,1) = x( 1);
    crystl.peak(0,2) = x( 2);
    crystl.peak(1,0) = x( 3);
    crystl.peak(1,1) = x( 4);
    crystl.peak(1,2) = x( 5);
    crystl.peak(2,0) = x( 6);
    crystl.peak(2,1) = x( 7);
    crystl.peak(2,2) = x( 8);
    crystl.mosaicity = x( 9);
    crystl.strain    = x(10);
    double value = 0;
    for (const auto [idx,i,v] : data) {
      size_t reduced = reduce_encode
        (get<0>(idx),get<1>(idx),get<2>(idx),pointgroup);
      if (reduce_n.count(reduced)==0) continue;
      const double m = intensities[reduce_n.at(reduced)];
      const auto [flx,wvn,bnd] =
        predict_integrated(
            idx,
            sources,
            crystl
            );
      const auto [mv,_nabla_mv,_hessian_v] =
        prediction_mv(1.0,m*flx,wvn,bnd,b,c);
      const auto [llg_value,_llg_nabla,_llg_hessian] =
        llg(i,mv(0),v+mv(1),a,universal_distribution);
      value += llg_value;
    }
    //cerr << value << endl;
    return value;
  }
};

struct scaling_target {
  const double& a;
  const size_t& pointgroup;
  const vector<struct source>& sources;
  //const patchmap<size_t,size_t>& reduce_n;
  const unordered_map<size_t,size_t>& reduce_n;
  const vector<tuple<IDX,double,double>>& data;
  const vector<double>& intensities;
  const struct crystl& crystl;
  // crystl.a crystl.b error.b error.c
  const tuple<double,matrix<double,4,1>>
  operator()(const matrix<double,4,1>& x) const {
    //cerr << "scaling operator begin" << endl;
    struct crystl xcrystl = crystl;
    xcrystl.a = x(0);
    xcrystl.b = x(1);
    const double b = abs(x(2));
    const double c = abs(x(3));
    double value = 0;
    matrix<double,4,1> J = zeros_matrix<double>(4,1);
    for (const auto [idx,i,v] : data) {
      size_t reduced = reduce_encode
        (get<0>(idx),get<1>(idx),get<2>(idx),pointgroup);
      if (reduce_n.count(reduced)==0) continue;
      const double m = intensities[reduce_n.at(reduced)];
      const auto [flx,wvn,bnd] =
        predict_integrated(
            idx,
            sources,
            xcrystl
            );
      const matrix<double,3,1> dhkl
        {1.0*get<0>(idx),1.0*get<1>(idx),1.0*get<2>(idx)};
      const double dflx_da = flx/xcrystl.a;
      const double dflx_db = -0.5*length_squared(xcrystl.R*dhkl)*flx;
      const auto [mv,nabla_mv,hessian_v] =
        prediction_mv(1.0,m*flx,wvn,bnd,b,c);
      const auto [llg_value,llg_nabla,llg_hessian] =
        llg(i,mv(0),v+mv(1),a,universal_distribution);
      //cerr << get<0>(idx) << " "
      //     << get<1>(idx) << " "
      //     << get<2>(idx) << " "
      //     << llg_value << endl;
      value += llg_value;
      J(0)+=dflx_da*m*(nabla_mv(1,0)*llg_nabla(0)+nabla_mv(1,1)*llg_nabla(1));
      J(1)+=dflx_db*m*(nabla_mv(1,0)*llg_nabla(1)+nabla_mv(1,1)*llg_nabla(1));
      J(2)+=nabla_mv(4,1)*llg_nabla(1);
      J(3)+=nabla_mv(5,1)*llg_nabla(1);
    }
    //cerr << value << endl;
    J(2)*=signum(x(2));
    J(3)*=signum(x(3));
    //cerr << setprecision(16) << value << endl;
    return {value,J};
  }
};

int main(int argc,char** argv) {
  //test_llg();
  //return 0;
  size_t spacegroup = 1, pointgroup = 1;
  if (argc>1) pointgroup=get_point_group(spacegroup=stoull(argv[1]));
  const double a = 1.0/16;
  // this is a property of the detector
  const double g = 9.9866655;
  //patchmap<size_t,size_t> reduce_n;
  unordered_map<size_t,size_t> reduce_n;
  vector<size_t> n_reduce;
  // ...{...{index,intensity,variance,scalefactor,variance_errormodel}...}...
  vector<vector<tuple<IDX,double,double>>> rows;
  // ...{...{n_crystal,intensity,variance}...}...
  vector<vector<tuple<
    size_t,
    IDX,
    double,
    double,
    double,
    double,
    double,
    double,
    double
  >>> cols;
  vector<double> intensities;
  vector<tuple<vector<source>,tuple<crystl,double,double>>> parameters;
  //const auto subset = [](const size_t& i){return (i%2)==0;};
  //const auto subset = [](const size_t& i){return (i%2)==1;};
  const auto subset = [](const size_t& i){return true;};
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
    if (subset(counter))
      parameters.emplace_back(sources,make_tuple(crystl,b,c));
    uint64_t m;
    cin.read(reinterpret_cast<char*>(&m),sizeof(uint64_t));
    cerr << setw(8) << parameters.size() << setw(8) << m << endl;
    if (subset(counter)) {
      rows.push_back(vector<tuple<IDX,double,double>>{});
      rows.back().reserve(m);
    }
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
      if (length(x)>6) continue;
      //if (length(x)>2) continue;
      if (subset(counter)) {
        rows.back().emplace_back(idx,i,pow(s,2));
        const size_t reduced = reduce_encode(h,k,l,pointgroup);
        if (reduce_n.count(reduced)==0) {
          reduce_n[reduced] = n_reduce.size();
          n_reduce.push_back(reduced);
          cols.emplace_back(
              vector<tuple<
                size_t,IDX,double,double,double,double,double,double,double
                >>{});
        }
        cols[reduce_n[reduced]].emplace_back(
          rows.size()-1,idx,i,pow(s,2),0.0,0.0,0.0,0.0,0.0
          );
      }
    }
    if (subset(counter)) rows.back().shrink_to_fit();
    //if (rows.size()==(1u<<15)) break;
    //if (rows.size()==3158) break;
    //if (rows.size()==1u<<17) break;
    //if (rows.size()==327680) break;
  }
  cerr << "number of crystals = " << rows.size() << endl;
  parameters.shrink_to_fit();
  n_reduce.shrink_to_fit();
  rows.shrink_to_fit();
  cols.shrink_to_fit();
  for (auto& row : rows) row.shrink_to_fit();
  if constexpr (false) {
  cerr << "begin initial scaling" << endl;
  transform(
      std::execution::par_unseq,
      parameters.begin(),parameters.end(),
      rows.begin(),
      parameters.begin(),
      [&g]
      (
        const tuple<vector<source>,tuple<crystl,double,double>>& parameters,
        const vector<tuple<IDX,double,double>>& data
      )
      {
        const auto& [sources,crystl_a_b] = parameters;
        auto [crystl,b,c] = crystl_a_b;
        matrix<double,19,1> x;
        set_crystal_vector(crystl,x);
        const initial_scaling_target target{g,sources,data};
        int best_i = 0;
        double best_value = target(x);
        for (int i=-4;i<=4;++i) {
          matrix<double,19,1> _x = x;
          _x(17) = x(17)*pow(2,i);
          //if (get<0>(target(_x))<best_value) {
          if (target(_x)<best_value) {
            //best_value = get<0>(target(_x));
            best_value = target(_x);
            best_i = i;
          }
        }
        x(17)*=pow(2,best_i);
        prediction_proposer propose;
        find_min_numerical
          (
            x,
            target,
            propose,
            count_stop_strategy{pow(2ul,8ul),pow(2ul,8ul)}
          );
        cerr << "." ;
        //cerr << endl ;
        //cerr << trans(x);
        set_crystal_from_vector(x,crystl);
        return make_tuple(sources,make_tuple(crystl,b,c));
      });
  }
  if constexpr (false) {
  for (const auto& [sources,crystl_a_b] : parameters) {
    const auto& [crystl,b,c] = crystl_a_b;
    cout << crystl.R
         <<crystl.peak(0,0)<<" "<<crystl.peak(0,1)<<" "<<crystl.peak(0,2)<<endl
                                <<crystl.peak(1,1)<<" "<<crystl.peak(1,2)<<endl
                                                       <<crystl.peak(2,2)<<endl;
    cout << crystl.mosaicity << " " << crystl.strain << " "
         << crystl.a         << " " << crystl.b      << " "
         << b                << " " << c             << endl;
  }
  return 0;
  }
  intensities.resize(cols.size());
  ifstream intensity_file("intensities");
  if (intensity_file.is_open()) {
    //patchmap<size_t,void> wereset;
    unordered_set<size_t> wereset;
    for (string line;getline(intensity_file,line);) {
      stringstream ss(line);
      int h,k,l;
      double i;
      ss >> h >> k >> l >> i;
      const size_t reduced = reduce_encode(h,k,l,pointgroup);
      if (reduce_n.count(reduced)==0) continue;
      intensities[reduce_n[reduced]]=i;
      wereset.insert(reduce_n[reduced]);
    }
    for (size_t i=0;i!=intensities.size();++i) {
      if (wereset.count(i)==0) intensities[i]=g;
    }
  } else {
    for (auto it=intensities.begin();it!=intensities.end();++it) *it=g;
  }
  double targetsum;
  for (size_t i=0;i!=256;++i) {
    cerr << "update scaling factor and variance from error model in columns"
         << endl;
    ofstream premerge("premerge");
    for (size_t n=0;n!=cols.size();++n) {
      const size_t reduced = n_reduce[n];
      const auto [h,k,l] = decode(reduced,pointgroup);
      const double& m = intensities[n];
      premerge << reduced << " " << h << " " << k << " " << l << endl;
      for (size_t i=0;i!=cols[n].size();++i) {
        const auto& idx = get<1>(cols[n][i]);
        const auto& [sources,crystl_b_c] = parameters[get<0>(cols[n][i])];
        const auto& [crystl,b,c] = crystl_b_c;
        auto [flx,wvn,bnd] = predict_integrated(idx,sources,crystl);
        auto [_flx,_wvn,_bnd] =
          predict_integrated_onlyscaling(idx,sources[0],crystl);
        const matrix<double,2,1> mv = get<0>(prediction_mv(
              1.0,m*flx,wvn,bnd,abs(b),abs(c)));
        const matrix<double,2,1> _mv = get<0>(prediction_mv(
              1.0,m*_flx,_wvn,_bnd,abs(b),abs(c)));
        get<4>(cols[n][i]) = flx;
        get<5>(cols[n][i]) = wvn;
        get<6>(cols[n][i]) = bnd;
        get<7>(cols[n][i]) = abs(b);
        get<8>(cols[n][i]) = abs(c);
        premerge << get<0>(cols[n][i]) << " "
                 << get<0>(idx) << " "
                 << get<1>(idx) << " "
                 << get<2>(idx) << " "
                 << get<2>(cols[n][i]) << " "
                 << get<3>(cols[n][i]) << " "
                 << mv(0) << " "
                 << mv(1) << " "
                 << get<0>(gauss(
                       get<2>(cols[n][i]),
                       mv(0),
                       mv(1)+get<3>(cols[n][i])
                       )) << " "
                 << universal_distribution(abs(get<2>(cols[n][i]))) << " "
                 << _mv(0) << " "
                 << _mv(1) << " "
                 << '\n';
      }
    }
    targetsum = 0.0;
    constexpr bool skipmerge = false;
    if constexpr (skipmerge) {
      for (auto it=intensities.begin();it!=intensities.end();++it) *it=g;
    } else {
      cerr << "merge" << endl;
      transform(
          std::execution::par_unseq,
          cols.begin(),cols.end(),
          intensities.begin(),
          intensities.begin(),
          [&a](
            const vector<tuple<
                size_t,IDX,double,double,double,double,double,double,double
                >>& data,
             const double& m) {
            matrix<double,1,1> x{m};
            merge_target target{a,data};
            //if constexpr (true){
            //  target(x);
            //  return x(0);
            //}
            //matrix<double,1,1> xp{1.7*m+1e-8};
            //matrix<double,1,1> xm{1.7*m-1e-8};
            //cerr << (get<0>(target(xp))-get<0>(target(xm)))/2e-8 << " " 
            //     << get<1>(target(xp))(0) << endl;
            //cerr << (get<1>(target(xp))(0)-get<1>(target(xm))(0))/2e-8 << " " 
            //     << get<2>(target(xp))(0) << endl;
            //cerr << endl;
            int best_i = 0;
            double best_value = get<0>(target(x));
            for (int i=-256;i<=256;++i) {
              matrix<double,1,1> _x = x;
              _x(0) = x(0)*pow(pow(2,1.0/16),i);
              if (get<0>(target(_x))<best_value) {
                best_value = get<0>(target(_x));
                best_i = i;
              }
            }
            x(0)*=pow(pow(2,1.0/16),best_i);
            find_min
              (
                x,
                target,
                count_stop_strategy{1024,1024},
                1e-9
              );
            return x(0);
          });
      ofstream intensity_file("intensities_"+to_string(i));
      for (auto it=reduce_n.begin();it!=reduce_n.end();++it) {
        const size_t reduced = it->first;
        const size_t n       = it->second;
        merge_target target{a,cols[n]};
        const double& m = intensities[n];
        matrix<double,1,1> x{m};
        const double sigma = sqrt(abs(1.0/get<2>(target(x))(0)));
        auto idx = decode(reduced,pointgroup);
        intensity_file
          << get<0>(idx) << " "
          << get<1>(idx) << " "
          << get<2>(idx) << " "
          << m << "  "
          << sigma << " "
          << cols[n].size() << " "
          << get<2>(target(x))(0) << endl;
      }
      targetsum = 0.0;
      for (auto it=reduce_n.begin();it!=reduce_n.end();++it) {
        const size_t reduced = it->first;
        const size_t n       = it->second;
        merge_target target{a,cols[n]};
        matrix<double,1,1> x{intensities[n]};
        targetsum+=get<0>(target(x));
      }
      cerr << "total target = " << targetsum << endl;
    }
    constexpr bool skipscale = false;
    if constexpr (!skipscale) {
    cerr << "scale" << endl;
    transform(
        std::execution::par_unseq,
        parameters.begin(),parameters.end(),
        rows.begin(),
        parameters.begin(),
        [&a,&pointgroup,&reduce_n,&intensities]
        (
          const tuple<vector<struct source>,
                      tuple<struct crystl,double,double>
                >& parameters,
          const vector<tuple<IDX,double,double>>& data
        )
        {
          const auto& [sources,crystl_b_c] = parameters;
          auto [crystl,b,c] = crystl_b_c;
          matrix<double,4,1> x{crystl.a,crystl.b,b,c};
          scaling_target target
            {a,pointgroup,sources,reduce_n,data,intensities,crystl};
          //if constexpr (true){
          //  target(x);
          //  return make_tuple(sources,make_tuple(crystl,b,c));
          //}
          if constexpr (false) {
            for (size_t i=0;i!=4;++i) {
              matrix<double,4,1> xp = x;
              xp(i)*=1.1;
              matrix<double,4,1> xm = x;
              xm(i)*=0.9;
              cerr << get<1>(target(x))(i) << " "
                   << (get<0>(target(xp))-get<0>(target(xm)))/(xp(i)-xm(i))
                   << endl;
            }
            cerr << endl;
          }
          //cerr << "before optimisation : " << get<0>(target(x)) << endl;
          find_min
            (
              x,
              target,
              count_stop_strategy{pow(2ul,8ul),pow(2u,8ul)},
              1e-12
            );
          //cerr << "after optimisation : " << get<0>(target(x)) << endl;
          crystl.a = x(0);
          crystl.b = x(1);
          b = abs(x(2));
          c = abs(x(3));
          cerr << ".";
          //cerr << endl;
          return make_tuple(sources,make_tuple(crystl,b,c));
        });
    cerr << endl;
    }
    for (size_t n=0;n!=cols.size();++n) {
      const size_t reduced = n_reduce[n];
      const auto [h,k,l] = decode(reduced,pointgroup);
      const double& m = intensities[n];
      for (size_t i=0;i!=cols[n].size();++i) {
        const auto& idx = get<1>(cols[n][i]);
        const auto& [sources,crystl_b_c] = parameters[get<0>(cols[n][i])];
        const auto& [crystl,b,c] = crystl_b_c;
        auto [flx,wvn,bnd] = predict_integrated(idx,sources,crystl);
        const matrix<double,2,1> mv = get<0>(prediction_mv(
              1.0,m*flx,wvn,bnd,abs(b),abs(c)));
        get<4>(cols[n][i]) = flx;
        get<5>(cols[n][i]) = wvn;
        get<6>(cols[n][i]) = bnd;
        get<7>(cols[n][i]) = abs(b);
        get<8>(cols[n][i]) = abs(c);
      }
    }
    targetsum = 0.0;
    for (auto it=reduce_n.begin();it!=reduce_n.end();++it) {
      const size_t reduced = it->first;
      const size_t n       = it->second;
      merge_target target{a,cols[n]};
      matrix<double,1,1> x{intensities[n]};
      targetsum+=get<0>(target(x));
    }
    cerr << "total target = " << targetsum << endl;
    //return 0;
    cerr << "refine" << endl;
    transform(
        std::execution::par_unseq,
        parameters.begin(),parameters.end(),
        rows.begin(),
        parameters.begin(),
        [&a,&pointgroup,&reduce_n,&intensities]
        (
          const tuple<vector<struct source>,
                      tuple<struct crystl,double,double>
                >& parameters,
          const vector<tuple<IDX,double,double>>& data
        )
        {
          const auto& [sources,crystl_b_c] = parameters;
          auto [crystl,b,c] = crystl_b_c;
          {
            matrix<double,9,1> x;
            x(0) = crystl.R(0,0);
            x(1) = crystl.R(0,1);
            x(2) = crystl.R(0,2);
            x(3) = crystl.R(1,0);
            x(4) = crystl.R(1,1);
            x(5) = crystl.R(1,2);
            x(6) = crystl.R(2,0);
            x(7) = crystl.R(2,1);
            x(8) = crystl.R(2,2);
            cell_proposer propose;
            find_min_numerical
              (
                x,
                cell_refinement_target
                  {a,pointgroup,sources,reduce_n,data,intensities,crystl,abs(b),abs(c)},
                propose,
                count_stop_strategy{pow(2ul,8ul),pow(2u,8ul)}
              );
            crystl.R(0,0) = x(0);
            crystl.R(0,1) = x(1);
            crystl.R(0,2) = x(2);
            crystl.R(1,0) = x(3);
            crystl.R(1,1) = x(4);
            crystl.R(1,2) = x(5);
            crystl.R(2,0) = x(6);
            crystl.R(2,1) = x(7);
            crystl.R(2,2) = x(8);
          }
          {
            matrix<double,11,1> x;
            x( 0) = crystl.peak(0,0);
            x( 1) = crystl.peak(0,1);
            x( 2) = crystl.peak(0,2);
            x( 3) = crystl.peak(1,0);
            x( 4) = crystl.peak(1,1);
            x( 5) = crystl.peak(1,2);
            x( 6) = crystl.peak(2,0);
            x( 7) = crystl.peak(2,1);
            x( 8) = crystl.peak(2,2);
            x( 9) = crystl.mosaicity;
            x(10) = crystl.strain;
            peak_proposer propose;
            find_min_numerical
              (
                x,
                peak_refinement_target
                  {a,pointgroup,sources,reduce_n,data,intensities,crystl,abs(b),abs(c)},
                propose,
                count_stop_strategy{pow(2ul,8ul),pow(2u,8ul)}
              );
            crystl.peak(0,0) = x( 0);
            crystl.peak(0,1) = x( 1);
            crystl.peak(0,2) = x( 2);
            crystl.peak(1,0) = x( 3);
            crystl.peak(1,1) = x( 4);
            crystl.peak(1,2) = x( 5);
            crystl.peak(2,0) = x( 6);
            crystl.peak(2,1) = x( 7);
            crystl.peak(2,2) = x( 8);
            crystl.mosaicity = x( 9);
            crystl.strain    = x(10);
          }
          cerr << ".";
          //cerr << endl;
          return make_tuple(sources,make_tuple(crystl,abs(b),abs(c)));
        });
    cerr << endl; 
    for (size_t n=0;n!=cols.size();++n) {
      const size_t reduced = n_reduce[n];
      const auto [h,k,l] = decode(reduced,pointgroup);
      const double& m = intensities[n];
      for (size_t i=0;i!=cols[n].size();++i) {
        const auto& idx = get<1>(cols[n][i]);
        const auto& [sources,crystl_b_c] = parameters[get<0>(cols[n][i])];
        const auto& [crystl,b,c] = crystl_b_c;
        auto [flx,wvn,bnd] = predict_integrated(idx,sources,crystl);
        const matrix<double,2,1> mv = get<0>(prediction_mv(
              1.0,m*flx,wvn,bnd,abs(b),abs(c)));
        get<4>(cols[n][i]) = flx;
        get<5>(cols[n][i]) = wvn;
        get<6>(cols[n][i]) = bnd;
        get<7>(cols[n][i]) = abs(b);
        get<8>(cols[n][i]) = abs(c);
      }
    }
    targetsum = 0.0;
    for (auto it=reduce_n.begin();it!=reduce_n.end();++it) {
      const size_t reduced = it->first;
      const size_t n       = it->second;
      merge_target target{a,cols[n]};
      matrix<double,1,1> x{intensities[n]};
      targetsum+=get<0>(target(x));
    }
    cerr << "total target = " << targetsum << endl;
    ofstream crystls_file("crystls_"+to_string(i));
    crystls_file << setprecision(10);
    if constexpr (true) { // binary - ascii - switch 
      for (const auto& [sources,crystl_b_c] : parameters) {
        const auto& [crystl,b,c] = crystl_b_c;
        crystls_file
          << "< "
          <<crystl.R(0,0)   <<" "<<crystl.R(0,1)   <<" "<<crystl.R(0,2)   <<'\n'
          <<crystl.R(1,0)   <<" "<<crystl.R(1,1)   <<" "<<crystl.R(1,2)   <<'\n'
          <<crystl.R(2,0)   <<" "<<crystl.R(2,1)   <<" "<<crystl.R(2,2)   <<'\n'
          <<crystl.peak(0,0)<<" "<<crystl.peak(0,1)<<" "<<crystl.peak(0,2)<<'\n'
                                 <<crystl.peak(1,1)<<" "<<crystl.peak(1,2)<<'\n'
                                                        <<crystl.peak(2,2)<<'\n'
          << crystl.mosaicity << " " << crystl.strain << " "
          << crystl.a         << " " << crystl.b      << " "
          << abs(b)           << " " << abs(c)        << endl;
      }
    } else {
      for (const auto& [sources,crystl_b_c] : parameters) {
        const auto& [crystl,b,c] = crystl_b_c;
        serialize_crystl(crystl,crystls_file);
        const double _b = abs(b);
        const double _c = abs(c);
        crystls_file.write(reinterpret_cast<const char*>(&_b),sizeof(double));
        crystls_file.write(reinterpret_cast<const char*>(&_c),sizeof(double));
      }
    }
  }
}
