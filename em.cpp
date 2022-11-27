#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "wmath.hpp"
#include "patchmap.hpp"

using std::abs;
using std::accumulate;
using std::array;
using std::cerr;
using std::cin;
using std::cout;
using std::endl;
using std::exponential_distribution;
using std::fill;
using std::fixed;
using std::get;
using std::getline;
using std::ifstream;
using std::imag;
using std::isinf;
using std::isnan;
using std::istream;
using std::lower_bound;
using std::max_element;
using std::normal_distribution;
using std::numeric_limits;
using std::ofstream;
using std::random_device;
using std::real;
using std::round;
using std::setprecision;
using std::setw;
using std::sort;
using std::stod;
using std::stoull;
using std::streamsize;
using std::string;
using std::stringstream;
using std::swap;
using std::to_string;
using std::tuple;
using std::unordered_map;
using std::uniform_real_distribution;
using std::vector;

using wmath::clip;
using wmath::destructive_median;
using wmath::mean_variance;
using wmath::pow;

using whash::patchmap;

constexpr double pi = 3.141592653589793;

constexpr double iphi = (sqrt(5)-1)/2;

struct cauchy{
  double x0 = 0;
  double gamma = 1;
  double operator()(const double& x) const {
    return 1/(pi*gamma*(1+pow((x-x0)/gamma,2)));
  }
  double cdf(const double& x) const {
    return atan((x-x0)/gamma)/pi+0.5;
  }
};

struct gauss{
  double m = 0.0;
  double v = 1.0;
  double operator()(const double& x, const double &m, const double &v) const {
    return exp(-0.5*(pow(x-m,2)/v+log(2*pi*v)));
  }
  double operator()(const double& x) const {
    return operator()(x,m,v);
  }
  double operator()(const double& x,const double p) const {
    return operator()(x,m,v+p);
  }
};

int main(int argc,char** argv) {
  patchmap<size_t,vector<tuple<size_t,double,double>>> observations;
  vector<array<double,3>> assignments; // { gauss 0 , gauss 1 , outlier }
  double outlier_probability = 1.0/16;
  array<double,3> model_probabilities;
  model_probabilities[0] = 0.5*(1-outlier_probability);
  model_probabilities[1] = 0.5*(1-outlier_probability);
  model_probabilities[2] =        outlier_probability;
  cauchy outlier_distribution;
  {
    double a = iphi;
    vector<double> flatabsx;
    size_t m;
    for (string line;getline(cin,line);) {
      size_t n;
      int32_t h,k,l;
      stringstream ss(line);
      ss >> n >> h >> k >> l;
      if (ss.eof()) {
        m = n;
        continue;
      }
      while (assignments.size()<n+1) {
        assignments.push_back({
               a *(1-outlier_probability),
            (1-a)*(1-outlier_probability),
                     outlier_probability
            });
        a+=iphi;
        a-=(a>1);
      }
      double i, v0, e, v1;
      ss >> i >> v0 >> e >> v1;
      if (!ss) {
        cerr << "could not read line:" << endl;
        cerr << line << endl;
        //return 1;
      }
      if (e*e>0) {
        const double x = i/e;
        const double v = (v0+v1)/(e*e);
        if ((!isnan(x))&&(!isnan(v))&&(!isinf(x))&&(!isinf(v))) {
          observations[m].emplace_back(n,x,v);
          flatabsx.push_back(abs(x));
        }
      }
    }
    outlier_distribution.gamma = destructive_median(
        flatabsx.begin(),
        flatabsx.end());
  }
  cerr << assignments.size() << " datasets" << endl;
  cerr << "gamma = " << outlier_distribution.gamma << endl;
  for (size_t i=0;i!=assignments.size();++i) {
    assignments[i][0]+=0.01*   (i%2) *(1-outlier_probability);
    assignments[i][1]+=0.01*(1-(i%2))*(1-outlier_probability);
    assignments[i][0]/=1.01;
    assignments[i][1]/=1.01;
  }
  patchmap<size_t,array<gauss,2>> theta;
  for (size_t i=0;i!=256;++i) {
    //cerr << "M step" << endl;
    for (auto it=observations.begin();it!=observations.end();++it) {
      double sumw0=0,mean0=0,var0=0;
      double sumw1=0,mean1=0,var1=0;
      for (auto [n,x,v] : it->second) {
        double o = outlier_probability*outlier_distribution(x);
        if (o<=0) continue;
        double p0 = (1-outlier_probability)*theta[it->first][0](x,v);
        if (isnan(p0)||isinf(p0)) p0=0; 
        double p1 = (1-outlier_probability)*theta[it->first][1](x,v);
        if (isnan(p1)||isinf(p1)) p1=0; 
        const double w0 = assignments[n][0]*p0/(p0+o)/v;
        if (w0>0.0) mean_variance(x,w0,sumw0,mean0,var0);
        const double w1 = assignments[n][1]*p1/(p1+o)/v;
        if (w1>0.0) mean_variance(x,w1,sumw1,mean1,var1);
        //cerr << n << " " << x << " " << v << " " << w0 << " " << w1 << endl;
        //cerr << theta[it->first][0](x,v) << " "
        //     << theta[it->first][1](x,v) << " "
        //     << outlier_distribution(x) << endl;
      }
      //cerr << mean0 << " " << var0 << " " << mean1 << " " << var1 << endl;
      //cerr << endl;
      if ((!isnan(mean0))&&(!isnan(var0))&&(var0>0)) {
        theta[it->first][0].m = mean0;
        theta[it->first][0].v = var0;
      }
      if ((!isnan(mean1))&&(!isnan(var1))&&(var1>0)) {
        theta[it->first][1].m = mean1;
        theta[it->first][1].v = var1;
      }
    }
    //cerr << "E step" << endl;
    for (auto & a : assignments ) a[0]=a[1]=a[2]=0;
    for (auto it=observations.begin();it!=observations.end();++it) {
      double t=1;
      for (size_t i=1;i!=2;++i) {
        for (auto [n,x,v] : it->second) {
          const double p0 = model_probabilities[0]*theta[it->first][0](x,v);
          if (!(isnan(p0)||isinf(p0))) if (i==0) {
            t+=p0;
          } else {
            assignments[n][0]+=p0/t;
          }
          const double p1 = model_probabilities[1]*theta[it->first][1](x,v);
          if (!(isnan(p1)||isinf(p1))) if (i==0) {
            t+=p1;
          } else {
            assignments[n][1]+=p1/t;
          }
          const double p2 = model_probabilities[2]*outlier_distribution(x);
          if (!(isnan(p2)||isinf(p2))) if (i==0) {
            t+=p2;
          } else {
            assignments[n][2]+=p2/t;
          }
          /*if (i==1) {
            cerr << n << " " << x << " " << v << " "
                 << p0/t << " " << p1/t << " " << p2/t << endl;
          }*/
        }
      }
    }
    model_probabilities[0]=model_probabilities[1]=model_probabilities[2]=0;
    for (auto & a : assignments ) {
      model_probabilities[0]+=a[0];
      model_probabilities[1]+=a[1];
      model_probabilities[2]+=a[2];
      double c = a[0]+a[1]+a[2];
      if (c) {
        a[0]/=c;
        a[1]/=c;
        a[2]/=c;
      } else {
        a[0]=1.0/3;
        a[1]=1.0/3;
        a[2]=1.0/3;
      }
    }
    {
      double c =
         model_probabilities[0]
        +model_probabilities[1]
        +model_probabilities[2];
      model_probabilities[0]/=c;
      model_probabilities[1]/=c;
      model_probabilities[2]/=c;
    }
    for (auto & a : assignments) {
      cerr << a[0] << " " << a[1] << " " << a[2] << endl;
    }
    cerr << endl;
    /*
    cerr << model_probabilities[0] << " "
         << model_probabilities[1] << " "
         << model_probabilities[2] << endl;*/
  }
  for (auto & a : assignments) {
    cout << a[0] << " " << a[1] << " " << a[2] << endl;
  }
}
