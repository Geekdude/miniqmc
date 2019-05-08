#ifndef KOKKOS_MULTI_BSPLINE_H
#define KOKKOS_MULTI_BSPLINE_H
#include <Kokkos_Core.hpp>
#include <vector>
#include <assert.h>

// at first crack just use LayoutLeft
// later will want to experiment with a tiled layout with
// a static tile size (32, 64, etc)

// could imagine a specialization for complex types
// that split the coefficients into real and imaginary parts
namespace qmcplusplus
{

template<typename p, typename valType, typename coefType>
void doEval_v(p x, p y, p z, valType& vals, coefType& coefs,
	      Kokkos::View<p[3]>& gridStarts, Kokkos::View<p[3]>& delta_invs,
	      Kokkos::View<p[16]>& A44, int blockSize = 32);

template<typename p, typename multiPosType, typename valType, typename coefType>
void doMultiEval_v(multiPosType& pos, valType& vals, coefType& coefs,
		   Kokkos::View<p[3]>& gridStarts, Kokkos::View<p[3]>& delta_invs,
		   Kokkos::View<p[16]>& A44, int blockSize = 32);

template<typename p, typename multiPosType, typename valType, typename coefType>
void doMultiEval_v2d(multiPosType& pos, valType& vals, coefType& coefs,
		     Kokkos::View<p[3]>& gridStarts, Kokkos::View<p[3]>& delta_invs,
		     Kokkos::View<p[16]>& A44, int blockSize = 32);

template<typename p, typename valType, typename gradType, typename hessType,typename coefType>
void doEval_vgh(p x, p y, p z, valType& vals, gradType& grad,
		hessType& hess, coefType& coefs,
		Kokkos::View<p[3]>& gridStarts, Kokkos::View<p[3]>& delta_invs,
		Kokkos::View<p[16]>& A44, Kokkos::View<p[16]>& dA44,
		Kokkos::View<p[16]>& d2A44, int blockSize = 32);

template<typename p, typename multiPosType, typename valType, typename gradType, typename hessType,typename coefType>
void doMultiEval_vgh(multiPosType& pos, valType& vals, gradType& grad, 
		     hessType& hess, Kokkos::View<int*>& isValidMap, int numValid, coefType& coefs,
		     Kokkos::View<p[3]>& gridStarts, Kokkos::View<p[3]>& delta_invs,
		     Kokkos::View<p[16]>& A44, Kokkos::View<p[16]>& dA44,
		     Kokkos::View<p[16]>& d2A44, int blockSize = 32);

template<typename p, int d>
struct multi_UBspline_base {
  // for bc_codes, 0 == periodic, 1 == deriv1,   2 == deriv2
  //               3 == flat    , 4 == natural,  5 == antiperiodic
  Kokkos::View<int[d]> left_bc_codes;
  Kokkos::View<int[d]> right_bc_codes;
  Kokkos::View<p[d]> left_values;
  Kokkos::View<p[d]> right_values;
  Kokkos::View<p[d]> gridStarts;
  Kokkos::View<p[d]> gridEnds;
  Kokkos::View<p[d]> deltas;
  Kokkos::View<p[d]> delta_invs;

protected:
  void initialize_base(std::vector<p> start_locs,
		       std::vector<p> end_locs,
		       std::vector<int> num_pts,
		       int boundary_condition_code) {
    assert(start_locs.size() == d);
    assert(end_locs.size() == d);
    
    left_bc_codes = Kokkos::View<int[d]>("left_bc_codes");
    right_bc_codes = Kokkos::View<int[d]>("right_bc_codes");
    left_values = Kokkos::View<p[d]>("left_values");
    right_values = Kokkos::View<p[d]>("right_values");
    gridStarts = Kokkos::View<p[d]>("gridStarts");
    gridEnds = Kokkos::View<p[d]>("gridEnds");
    deltas = Kokkos::View<p[d]>("deltas");
    delta_invs = Kokkos::View<p[d]>("delta_invs");
    
    auto lbcMirror = Kokkos::create_mirror_view(left_bc_codes);
    auto rbcMirror = Kokkos::create_mirror_view(right_bc_codes);
    auto gsMirror = Kokkos::create_mirror_view(gridStarts);
    auto geMirror = Kokkos::create_mirror_view(gridEnds);
    auto gdMirror = Kokkos::create_mirror_view(deltas);
    auto gdinvMirror = Kokkos::create_mirror_view(delta_invs);
    
    std::vector<int> nx;  
    for (int i = 0; i < d; i++) {
      lbcMirror(i) = boundary_condition_code;
      rbcMirror(i) = boundary_condition_code;
      gsMirror(i) = start_locs[i];
      geMirror(i) = end_locs[i];
      
      nx.push_back(0);
      // if periodic or antiperiodic
      if (boundary_condition_code == 0 || boundary_condition_code == 5) {
	nx[i] = num_pts[i] + 3;
      } else {
	nx[i] = num_pts[i] + 2;
      }
      p delta = (end_locs[i] - start_locs[i]) / static_cast<p>(nx[i]-3);
      gdMirror(i) = delta;
      gdinvMirror(i) = 1.0 / delta;
    }
    
    Kokkos::deep_copy(left_bc_codes, lbcMirror);
    Kokkos::deep_copy(right_bc_codes, rbcMirror);
    Kokkos::deep_copy(gridStarts, gsMirror);
    Kokkos::deep_copy(gridEnds, geMirror);
    Kokkos::deep_copy(deltas, gdMirror);
    Kokkos::deep_copy(delta_invs, gdinvMirror);
  }
};  


template<typename p, int blocksize, int d>
struct multi_UBspline : public multi_UBspline_base<p,d> {};

template<typename p, int blocksize>
struct multi_UBspline<p, blocksize, 1> : public multi_UBspline_base<p,1> {
  using layout = Kokkos::LayoutRight;
  using coef_t = Kokkos::View<p**, layout>; // x, splines
  coef_t coef;

  using single_coef_t = Kokkos::View<p*>;
  using single_coef_mirror_t = typename single_coef_t::HostMirror;
  single_coef_t single_coef;
  single_coef_mirror_t single_coef_mirror;

  void initialize(std::vector<int>& dvec, std::vector<p>& start,
		  std::vector<p>& end, int bcCode, int numSpo) {
    this->initialize_base(start, end, dvec, 0);
    initializeCoefs(dvec, numSpo);
  }
  
  void pushCoefToDevice(int i) {
    Kokkos::deep_copy(single_coef, single_coef_mirror);
    auto singleCoef = single_coef;
    auto multiCoefs = coef;	
    int spoNum = i;

    Kokkos::parallel_for("pushCoefToMulti", coef.extent(0),
	  		 KOKKOS_LAMBDA(const int& i0) {
                            multiCoefs(i0,spoNum) = singleCoef(i0);
                         }); 
  }
 private:
  void initializeCoefs(std::vector<int>& dvec, int numSpo) {
    assert(dvec.size() == 1);
    std::vector<int> nx(dvec.begin(), dvec.end());
    
    auto lbcMirror = Kokkos::create_mirror_view(this->left_bc_codes);
    Kokkos::deep_copy(lbcMirror, this->left_bc_codes);
    for (int i = 0; i < 1; i++) {
      if (lbcMirror(i) == 0 || lbcMirror(i) == 5) {
	nx[i] += 3;
      } else {
	nx[i] += 2;
      }
    }
    coef = coef_t("coefs", nx[0], numSpo);
    single_coef = single_coef_t("single_coef", nx[0]);
    single_coef_mirror = Kokkos::create_mirror_view(single_coef);
  }

};

template<typename p, int blocksize>
struct multi_UBspline<p, blocksize, 2> : public multi_UBspline_base<p,2> {
  using layout = Kokkos::LayoutRight;
  using coef_t = Kokkos::View<p***, layout>; // x, y, splines
  coef_t coef;

  using single_coef_t = Kokkos::View<p**>;
  using single_coef_mirror_t = typename single_coef_t::HostMirror;
  single_coef_t single_coef;
  single_coef_mirror_t single_coef_mirror;

  void initialize(std::vector<int>& dvec, std::vector<p>& start,
		  std::vector<p>& end, int bcCode, int numSpo) {
    this->initialize_base(start, end, dvec, 0);
    initializeCoefs(dvec, numSpo);
  }
    
  void pushCoefToDevice(int i) {
    Kokkos::deep_copy(single_coef, single_coef_mirror);
    auto singleCoef = single_coef;
    auto multiCoefs = coef;
    int spoNum = i;

    Kokkos::parallel_for("pushCoefToMulti", 
			 Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left> >({0,0}, {singleCoef.extent(0),singleCoef.extent(1)}),
	  	         KOKKOS_LAMBDA(const int& i0, const int& i1) {
			   multiCoefs(i0,i1,spoNum) = singleCoef(i0,i1);
                         }); 
  }
 private:
  void initializeCoefs(std::vector<int>& dvec, int numSpo) {
    assert(dvec.size() == 2);
    std::vector<int> nx(dvec.begin(), dvec.end());
    
    auto lbcMirror = Kokkos::create_mirror_view(this->left_bc_codes);
    Kokkos::deep_copy(lbcMirror, this->left_bc_codes);
    for (int i = 0; i < 2; i++) {
      if (lbcMirror(i) == 0 || lbcMirror(i) == 5) {
	nx[i] += 3;
      } else {
	nx[i] += 2;
      }
    }    
    coef = coef_t("coefs", nx[0], nx[1], numSpo);
    single_coef = single_coef_t("single_coef", nx[0], nx[1]);
    single_coef_mirror = Kokkos::create_mirror_view(single_coef);
  }

};

template<typename p, int blocksize>
struct multi_UBspline<p, blocksize, 3> : public multi_UBspline_base<p,3> {
  using layout = Kokkos::LayoutRight;
  using coef_t = Kokkos::View<p****, layout>; // x, y, z, splines
  coef_t coef;

  using single_coef_t = Kokkos::View<p***>;
  using single_coef_mirror_t = typename single_coef_t::HostMirror;
  single_coef_t single_coef;
  single_coef_mirror_t single_coef_mirror;

  Kokkos::View<p[16]> A44;
  Kokkos::View<p[16]> dA44;
  Kokkos::View<p[16]> d2A44;

  void initialize(std::vector<int>& dvec, std::vector<p>& start,
		  std::vector<p>& end, int bcCode, int numSpo) {
    this->initialize_base(start, end, dvec, 0);
    initializeCoefs(dvec, numSpo);
  }

  void pushCoefToDevice(int i) {
    Kokkos::deep_copy(single_coef, single_coef_mirror);
    auto singleCoef = single_coef;
    auto multiCoefs = coef;
    int spoNum = i;

    Kokkos::parallel_for("pushCoefToMulti", 
			 Kokkos::MDRangePolicy<Kokkos::Rank<3,Kokkos::Iterate::Left> >({0,0,0}, {singleCoef.extent(0),singleCoef.extent(1),singleCoef.extent(2)}),
	  	         KOKKOS_LAMBDA(const int& i0, const int& i1, const int& i2) {
			   multiCoefs(i0,i1,i2,spoNum) = singleCoef(i0,i1,i2);
                         }); 


  }

  template<typename valType>
  void evaluate_v(p x, p y, p z, valType& vals) {
    assert(vals.extent(0) == coef.extent(3));
    doEval_v(x, y, z, vals, coef, this->gridStarts, this->delta_invs, A44, blocksize);
  }
  template<typename multiPosType, typename multiValType>
  void multi_evaluate_v(multiPosType& pos, multiValType& vals) {
    doMultiEval_v(pos, vals, coef, this->gridStarts, this->delta_invs, A44, blocksize);
  }

  // the 2d here refers to the dimensionality of pos and vals
  // instead of just n positions and values, we have nwalkers x nknots
  template<typename multiPosType, typename multiValType>
  void multi_evaluate_v2d(multiPosType& pos, multiValType& vals) {
    doMultiEval_v2d(pos, vals, coef, this->gridStarts, this->delta_invs, A44, blocksize);
  }

  template<typename valType, typename gradType, typename hessType>
  void evaluate_vgh(p x, p y, p z, valType& vals, gradType& grad,
		    hessType& hess) {
    assert(vals.extent(0) == coef.extent(3));
    doEval_vgh(x, y, z, vals, grad, hess, coef, this->gridStarts, this->delta_invs, A44, dA44, d2A44, blocksize);
  }
  template<typename multiPosType, typename multiValType, typename multiGradType, typename multiHessType>
  void multi_evaluate_vgh(multiPosType& pos, multiValType& multiVals,
			  multiGradType& multiGrad, multiHessType& multiHess,
			  Kokkos::View<int*>& isValidMap, int numValid) {
    doMultiEval_vgh(pos, multiVals, multiGrad, multiHess, isValidMap, numValid, 
		    coef, this->gridStarts, this->delta_invs, A44, dA44, d2A44, blocksize);
  }

 private:
  void initializeCoefs(std::vector<int>& dvec, int numSpo) {
    assert(dvec.size() == 3);
    std::vector<int> nx(dvec.begin(), dvec.end());

    auto lbcMirror = Kokkos::create_mirror_view(this->left_bc_codes);
    Kokkos::deep_copy(lbcMirror, this->left_bc_codes);
    for (int i = 0; i < 3; i++) {
      if (lbcMirror(i) == 0 || lbcMirror(i) == 5) {
	nx[i] += 3;
      } else {
	nx[i] += 2;
      }
    }    
    coef = coef_t("coefs", nx[0], nx[1], nx[2], numSpo);
    single_coef = single_coef_t("single_coef", nx[0], nx[1], nx[2]);
    single_coef_mirror = Kokkos::create_mirror_view(single_coef);
    initializeAs();
  }

  void initializeAs() {
    A44 = Kokkos::View<p[16]>("A44");
    auto A44Mirror = Kokkos::create_mirror_view(A44);
    dA44 = Kokkos::View<p[16]>("dA44");
    auto dA44Mirror = Kokkos::create_mirror_view(dA44);
    d2A44 = Kokkos::View<p[16]>("d2A44");
    auto d2A44Mirror = Kokkos::create_mirror_view(d2A44);
    
    p ta[16] = {
      -1.0 / 6.0, 3.0 / 6.0, -3.0 / 6.0, 1.0 / 6.0, 3.0 / 6.0, -6.0 / 6.0,
      0.0 / 6.0,  4.0 / 6.0, -3.0 / 6.0, 3.0 / 6.0, 3.0 / 6.0, 1.0 / 6.0,
      1.0 / 6.0,  0.0 / 6.0, 0.0 / 6.0,  0.0 / 6.0};
    p tda[16] = {
      0.0, -0.5, 1.0, -0.5, 0.0, 1.5, -2.0, 0.0,
      0.0, -1.5, 1.0, 0.5,  0.0, 0.5, 0.0,  0.0};
    p td2a[16] = {
      0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 3.0, -2.0,
      0.0, 0.0, -3.0, 1.0, 0.0, 0.0, 1.0, 0.0};
    
    for (int i = 0; i < 16; i++) {
      A44Mirror(i) = ta[i];
      dA44Mirror(i) = tda[i];
      d2A44Mirror(i) = td2a[i];
    }
    Kokkos::deep_copy(A44, A44Mirror);
    Kokkos::deep_copy(dA44, dA44Mirror);
    Kokkos::deep_copy(d2A44, d2A44Mirror);
  }
};

template<typename viewType, typename T>
KOKKOS_INLINE_FUNCTION void compute_prefactors(Kokkos::Array<T,4>& a, T tx, viewType A44) {
  a[0] = ((A44(0) * tx + A44(1)) * tx + A44(2)) * tx + A44(3);
  a[1] = ((A44(4) * tx + A44(5)) * tx + A44(6)) * tx + A44(7);
  a[2] = ((A44(8) * tx + A44(9)) * tx + A44(10)) * tx + A44(11);
  a[3] = ((A44(12) * tx + A44(13)) * tx + A44(14)) * tx + A44(15);
}

// would rather this be a member, but not sure how to 
template<typename viewType, typename T>
KOKKOS_INLINE_FUNCTION void compute_prefactors(Kokkos::Array<T,4>& a, Kokkos::Array<T,4>& da,
					       Kokkos::Array<T,4>& d2a, T tx,
					       viewType A44, viewType dA44, viewType d2A44) {
  a[0]   = ((A44(0) * tx + A44(1)) * tx + A44(2)) * tx + A44(3);
  a[1]   = ((A44(4) * tx + A44(5)) * tx + A44(6)) * tx + A44(7);
  a[2]   = ((A44(8) * tx + A44(9)) * tx + A44(10)) * tx + A44(11);
  a[3]   = ((A44(12) * tx + A44(13)) * tx + A44(14)) * tx + A44(15);
  da[0]  = ((dA44(0) * tx + dA44(1)) * tx + dA44(2)) * tx + dA44(3);
  da[1]  = ((dA44(4) * tx + dA44(5)) * tx + dA44(6)) * tx + dA44(7);
  da[2]  = ((dA44(8) * tx + dA44(9)) * tx + dA44(10)) * tx + dA44(11);
  da[3]  = ((dA44(12) * tx + dA44(13)) * tx + dA44(14)) * tx + dA44(15);
  d2a[0] = ((d2A44(0) * tx + d2A44(1)) * tx + d2A44(2)) * tx + d2A44(3);
  d2a[1] = ((d2A44(4) * tx + d2A44(5)) * tx + d2A44(6)) * tx + d2A44(7);
  d2a[2] = ((d2A44(8) * tx + d2A44(9)) * tx + d2A44(10)) * tx + d2A44(11);
  d2a[3] = ((d2A44(12) * tx + d2A44(13)) * tx + d2A44(14)) * tx + d2A44(15);
}

#define MYMAX(a, b) (a < b ? b : a)
#define MYMIN(a, b) (a > b ? b : a)
template<typename T>
KOKKOS_INLINE_FUNCTION void get(T x, T& dx, int& ind, int ng) {
  T ipart;
  dx  = std::modf(x, &ipart);
  ind = MYMIN(MYMAX(int(0), static_cast<int>(ipart)), ng);
}
#undef MYMAX
#undef MYMIN

template<typename p, typename valType, typename coefType>
void doEval_v(p x, p y, p z, valType& vals, coefType& coefs,
	      Kokkos::View<p[3]>& gridStarts, Kokkos::View<p[3]>& delta_invs,
	      Kokkos::View<p[16]>& A44, int blockSize) {
  int numBlocks = coefs.extent(3) / blockSize;
  if (coefs.extent(3) % blockSize != 0) {
    numBlocks++;
  }
  
  Kokkos::TeamPolicy<> policy(numBlocks,1,32);
  Kokkos::parallel_for("KokkosMultiBspline-doEval_v",
		       policy, KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type member) {
      const int start = blockSize * member.league_rank();
      int end = start + blockSize;

      Kokkos::Array<p,3> loc;
      loc[0] = x;
      loc[1] = y;
      loc[2] = z;

      if (end > coefs.extent(3)) {
	end = coefs.extent(3);
      }
      const int num_splines = end-start;

      for (int i = 0; i < 3; i++) {
	loc[i] -= gridStarts(i);
      }
      Kokkos::Array<p,3> ts;
      Kokkos::Array<int,3> is;
      for (int i = 0; i < 3; i++) {
	get(loc[i] * delta_invs(i), ts[i], is[i], coefs.extent(i)-1);
      }

      Kokkos::Array<p,4> a,b,c;
      compute_prefactors(a, ts[0], A44);
      compute_prefactors(b, ts[1], A44);
      compute_prefactors(c, ts[2], A44);

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, num_splines),
			   [&](const int& i) { vals(start+i) = p(); });

      for (int i = 0; i < 4; i++) {
	for (int j = 0; j < 4; j++) {
	  const p pre00 = a[i] * b[j];
	  Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, num_splines), [&](const int& n) {
	      const int sponum = start+n;
	      vals(sponum) += pre00 * 
		(c[0] * coefs(is[0]+i,is[1]+j,is[2],sponum) +
		 c[1] * coefs(is[0]+i,is[1]+j,is[2]+1,sponum) +
		 c[2] * coefs(is[0]+i,is[1]+j,is[2]+2,sponum) +
		 c[3] * coefs(is[0]+i,is[1]+j,is[2]+3,sponum));
	    });
	}
      }
    });      
}

template<typename p, typename multiPosType, typename valType, typename coefType>
void doMultiEval_v2d(multiPosType& pos, valType& vals, coefType& coefs,
		     Kokkos::View<p[3]>& gridStarts, Kokkos::View<p[3]>& delta_invs,
		     Kokkos::View<p[16]>& A44, int blockSize) {
  int numWalkers = pos.extent(0);
  int numKnots = pos.extent(1);
  int numBlocks = coefs.extent(3) / blockSize;
  if (coefs.extent(3) % blockSize != 0) {
    numBlocks++;
  }
  Kokkos::TeamPolicy<> policy(numWalkers*numKnots,Kokkos::AUTO,32);
  Kokkos::parallel_for("KokkosMultiBspline-doMultiEval_v2d",
		       policy, KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type member) {
      int walkerNum = member.league_rank() / numKnots;
      int knotNum = member.league_rank() % numKnots;
      
      // wrap this so only a single thread per league does this
      Kokkos::single(Kokkos::PerTeam(member), [&]() {	 
	  for(int i = 0; i < 3; i++) {
	    pos(walkerNum, knotNum,i) -= gridStarts(i);
	  }
	});
      member.team_barrier();

      Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, numBlocks),
			   [&](int blockNum) {
			     const int start = blockSize * blockNum;
			     int end = start + blockSize;
			     if (end > coefs.extent(3)) {
			       end = coefs.extent(3);
			     }
			     const int num_splines = end-start;
			   
			     Kokkos::Array<p,3> ts;
			     Kokkos::Array<int,3> is;
			     for (int i = 0; i < 3; i++) {
			       get(pos(walkerNum,knotNum,i) * delta_invs(i), ts[i], is[i], coefs.extent(i)-1);
			     }
			     Kokkos::Array<p,4> a,b,c;
			     compute_prefactors(a, ts[0], A44);
			     compute_prefactors(b, ts[1], A44);
			     compute_prefactors(c, ts[2], A44);
			     
			     Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, num_splines),
						  [&](const int& i) { vals(walkerNum,knotNum,start+i) = p(); });

			     for (int i = 0; i < 4; i++) {
			       for (int j = 0; j < 4; j++) {
				 const p pre00 = a[i] * b[j];
				 Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, num_splines), 
						      [&](const int& n) {
							const int sponum = start+n;
							vals(walkerNum,knotNum,sponum) += pre00 * 
							  (c[0] * coefs(is[0]+i,is[1]+j,is[2],sponum) +
							   c[1] * coefs(is[0]+i,is[1]+j,is[2]+1,sponum) +
							   c[2] * coefs(is[0]+i,is[1]+j,is[2]+2,sponum) +
							   c[3] * coefs(is[0]+i,is[1]+j,is[2]+3,sponum));
						      });
			       }
			     }
			   });
    });
}

 
template<typename p, typename multiPosType, typename valType, typename coefType>
void doMultiEval_v(multiPosType& pos, valType& vals, coefType& coefs,
		   Kokkos::View<p[3]>& gridStarts, Kokkos::View<p[3]>& delta_invs,
		   Kokkos::View<p[16]>& A44, int blockSize) {
  int numWalkers = pos.extent(0);
  int numBlocks = coefs.extent(3) / blockSize;
  if (coefs.extent(3) % blockSize != 0) {
    numBlocks++;
  }
  Kokkos::TeamPolicy<> policy(numWalkers,numBlocks,32);
  Kokkos::parallel_for("KokkosMultiBspline-doMultiEval_v",
		       policy, KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type member) {
      const int walkerNum = member.league_rank();
      
      // wrap this so only a single thread per league does this
      Kokkos::single(Kokkos::PerTeam(member), [&]() {	  
	  for(int i = 0; i < 3; i++) {
	    pos(walkerNum,i) -= gridStarts(i);
	  }
	});

      Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, numBlocks),
			   [&](int blockNum) {
			     const int start = blockSize * blockNum;
			     int end = start + blockSize;
			     if (end > coefs.extent(3)) {
			       end = coefs.extent(3);
			     }
			     const int num_splines = end-start;
			   
			     Kokkos::Array<p,3> ts;
			     Kokkos::Array<int,3> is;
			     for (int i = 0; i < 3; i++) {
			       get(pos(walkerNum,i) * delta_invs(i), ts[i], is[i], coefs.extent(i)-1);
			     }
			     Kokkos::Array<p,4> a,b,c;
			     compute_prefactors(a, ts[0], A44);
			     compute_prefactors(b, ts[1], A44);
			     compute_prefactors(c, ts[2], A44);
			     
			     Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, num_splines),
						  [&](const int& i) { vals(walkerNum,start+i) = p(); });

			     for (int i = 0; i < 4; i++) {
			       for (int j = 0; j < 4; j++) {
				 const p pre00 = a[i] * b[j];
				 Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, num_splines), 
						      [&](const int& n) {
							const int sponum = start+n;
							vals(walkerNum,sponum) += pre00 * 
							  (c[0] * coefs(is[0]+i,is[1]+j,is[2],sponum) +
							   c[1] * coefs(is[0]+i,is[1]+j,is[2]+1,sponum) +
							   c[2] * coefs(is[0]+i,is[1]+j,is[2]+2,sponum) +
							   c[3] * coefs(is[0]+i,is[1]+j,is[2]+3,sponum));
						      });
			       }
			     }
			   });
    });
}

template<typename p, typename valType, typename gradType, typename hessType, typename coefType>
void doEval_vgh(p x, p y, p z, valType& vals, gradType& grad,
		hessType& hess, coefType& coefs,
		Kokkos::View<p[3]>& gridStarts, Kokkos::View<p[3]>& delta_invs,
		Kokkos::View<p[16]>& A44, Kokkos::View<p[16]>& dA44,
		Kokkos::View<p[16]>& d2A44, int blockSize) {
  int numBlocks = coefs.extent(3) / blockSize;
  if (coefs.extent(3) % blockSize != 0) {
    numBlocks++;
  }
    
  Kokkos::TeamPolicy<> policy(numBlocks,1,32);
  Kokkos::parallel_for("KokkosMultiBspline-doEval_vgh",
		       policy, KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type member) {
      const int start = blockSize * member.league_rank();
      int end = start + blockSize;
      if (end > coefs.extent(3)) {
	end = coefs.extent(3);
      }

      Kokkos::Array<p,3> loc;
      loc[0] = x;
      loc[1] = y;
      loc[2] = z;
      const int num_splines = end-start;

      for (int i = 0; i < 3; i++) {
	loc[i] -= gridStarts(i);
      }
      Kokkos::Array<p,3> ts;
      Kokkos::Array<int,3> is;
      for (int i = 0; i < 3; i++) {
	get(loc[i] * delta_invs[i], ts[i], is[i], coefs.extent(i)-1);
      }

      Kokkos::Array<p,4> a,b,c, da, db, dc, d2a, d2b, d2c;
      compute_prefactors(a, da, d2a, ts[0], A44, dA44, d2A44);
      compute_prefactors(b, db, d2b, ts[1], A44, dA44, d2A44);
      compute_prefactors(c, dc, d2c, ts[2], A44, dA44, d2A44);

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, num_splines), [&](const int& i) {
	  const int spl = start + i;
	  vals(spl) = p();
	  grad(spl,0) = p();
	  grad(spl,1) = p();
	  grad(spl,2) = p();
	  hess(spl,0) = p();
	  hess(spl,1) = p();
	  hess(spl,2) = p();
	  hess(spl,3) = p();
	  hess(spl,4) = p();
	  hess(spl,5) = p();
	});

      //std::cout << "Got here" << std::endl;
      for (int i = 0; i < 4; i++) {
	for (int j = 0; j < 4; j++) {
	  const p pre20 = d2a[i] * b[j];
	  const p pre10 = da[i] * b[j];
	  const p pre00 = a[i] * b[j];
	  const p pre11 = da[i] * db[j];
	  const p pre01 = a[i] * db[j];
	  const p pre02 = a[i] * d2b[j];

	  Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, num_splines), [&](const int& n) {
	      const int sponum = start+n;
	      const p sum0 = c[0] * coefs(is[0]+i, is[1]+j, is[2], sponum) +
		c[1] * coefs(is[0]+i, is[1]+j, is[2]+1, sponum) +
		c[2] * coefs(is[0]+i, is[1]+j, is[2]+2, sponum) +
		c[3] * coefs(is[0]+i, is[1]+j, is[2]+3, sponum);
	      const p sum1 = dc[0] * coefs(is[0]+i, is[1]+j, is[2], sponum) +
		dc[1] * coefs(is[0]+i, is[1]+j, is[2]+1, sponum) +
		dc[2] * coefs(is[0]+i, is[1]+j, is[2]+2, sponum) +
		dc[3] * coefs(is[0]+i, is[1]+j, is[2]+3, sponum);
	      const p sum2 = d2c[0] * coefs(is[0]+i, is[1]+j, is[2], sponum) +
		d2c[1] * coefs(is[0]+i, is[1]+j, is[2]+1, sponum) +
		d2c[2] * coefs(is[0]+i, is[1]+j, is[2]+2, sponum) +
		d2c[3] * coefs(is[0]+i, is[1]+j, is[2]+3, sponum);
	      
	      hess(sponum,0) += pre20 * sum0;
	      hess(sponum,1) += pre11 * sum0;
	      hess(sponum,2) += pre10 * sum1;
	      hess(sponum,3) += pre02 * sum0;
	      hess(sponum,4) += pre01 * sum1;
	      hess(sponum,5) += pre00 * sum2;
	      grad(sponum,0) += pre10 * sum0;
	      grad(sponum,1) += pre01 * sum0;
	      grad(sponum,2) += pre00 * sum1;
	      vals(sponum) += pre00 * sum0;
	    });
	}
      }

      const p dxx = delta_invs(0) * delta_invs(0);
      const p dyy = delta_invs(1) * delta_invs(1);
      const p dzz = delta_invs(2) * delta_invs(2);
      const p dxy = delta_invs(0) * delta_invs(1);
      const p dxz = delta_invs(0) * delta_invs(2);
      const p dyz = delta_invs(1) * delta_invs(2);
      
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, num_splines), [&](const int& n) {
	  const int sponum = start+n;
	  grad(sponum,0) *= delta_invs(0);
	  grad(sponum,1) *= delta_invs(1);
	  grad(sponum,2) *= delta_invs(2);
	  hess(sponum,0) *= dxx;
	  hess(sponum,1) *= dyy;
	  hess(sponum,2) *= dzz;
	  hess(sponum,3) *= dxy;
	  hess(sponum,4) *= dxz;
	  hess(sponum,5) *= dyz;
	});
    });
}
 

template<typename p, typename multiPosType, typename valType, typename gradType, typename hessType,typename coefType>
void doMultiEval_vgh(multiPosType& pos, valType& vals, gradType& grad,
		     hessType& hess, Kokkos::View<int*>& isValidMap, int numValid, coefType& coefs,
		     Kokkos::View<p[3]>& gridStarts, Kokkos::View<p[3]>& delta_invs,
		     Kokkos::View<p[16]>& A44, Kokkos::View<p[16]>& dA44,
		     Kokkos::View<p[16]>& d2A44, int blockSize) {
  
  int numWalkers = numValid;
  int numBlocks = coefs.extent(3) / blockSize;
  if (coefs.extent(3) % blockSize != 0) {
    numBlocks++;
  }
  Kokkos::TeamPolicy<> policy(numWalkers*numBlocks,1,32);
  Kokkos::parallel_for("KokkosMultiBspline-doMultiEval_vgh",
		       policy, KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type member) {
      const int packedWalkerIdx = member.league_rank() / numBlocks;
      const int walkerNum = isValidMap(packedWalkerIdx);
      const int blockNum = member.league_rank() % numBlocks;

      Kokkos::Array<p,3> locpos;
      for (int i = 0; i <3; i++) {
	locpos[i] = pos(walkerNum, i) - gridStarts(i);
      }

      const int start = blockSize * blockNum;
      int end = start + blockSize;
      if (end > coefs.extent(3)) {
	end = coefs.extent(3);
      }
      const int num_splines = end-start;
			     
      // could explore the use of scratch pad memory here
      // ts, is and all of the kokkos arrays will be the same
      // for every member of the team
      Kokkos::Array<p,3> ts;
      Kokkos::Array<int,3> is;
      for (int i = 0; i < 3; i++) {
	get(locpos[i] * delta_invs(i), ts[i], is[i], coefs.extent(i)-1);
      }

      Kokkos::Array<p,4> a,b,c, da, db, dc, d2a, d2b, d2c;
      compute_prefactors(a, da, d2a, ts[0], A44, dA44, d2A44);
      compute_prefactors(b, db, d2b, ts[1], A44, dA44, d2A44);
      compute_prefactors(c, dc, d2c, ts[2], A44, dA44, d2A44);
      
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, num_splines), 
			   [&](const int& i) {
			     const int spl = start + i;
			     vals(walkerNum)(spl) = p();
			     grad(walkerNum)(spl, 0) = p();
			     grad(walkerNum)(spl, 1) = p();
			     grad(walkerNum)(spl, 2) = p();
			     hess(walkerNum)(spl, 0) = p();
			     hess(walkerNum)(spl, 1) = p();
			     hess(walkerNum)(spl, 2) = p();
			     hess(walkerNum)(spl, 3) = p();
			     hess(walkerNum)(spl, 4) = p();
			     hess(walkerNum)(spl, 5) = p();
			   });

      // could explore using scratch pad memory again, all of the pre?? elements
      // are the same for every member of the team

      // could also imagine loading all of the necessary coefs into scratch pad memory
      // probably this would be worse for vectorization though
      for (int i = 0; i < 4; i++) {
	for (int j = 0; j < 4; j++) {
	  const p pre20 = d2a[i] * b[j];
	  const p pre10 = da[i] * b[j];
	  const p pre00 = a[i] * b[j];
	  const p pre11 = da[i] * db[j];
	  const p pre01 = a[i] * db[j];
	  const p pre02 = a[i] * d2b[j];

	  Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, num_splines), 
			       [&](const int& n) {
				 const int sponum = start+n;
				 const p sum0 = c[0] * coefs(is[0]+i, is[1]+j, is[2], sponum) +
				   c[1] * coefs(is[0]+i, is[1]+j, is[2]+1, sponum) +
				   c[2] * coefs(is[0]+i, is[1]+j, is[2]+2, sponum) +
				   c[3] * coefs(is[0]+i, is[1]+j, is[2]+3, sponum);
				 const p sum1 = dc[0] * coefs(is[0]+i, is[1]+j, is[2], sponum) +
				   dc[1] * coefs(is[0]+i, is[1]+j, is[2]+1, sponum) +
				   dc[2] * coefs(is[0]+i, is[1]+j, is[2]+2, sponum) +
				   dc[3] * coefs(is[0]+i, is[1]+j, is[2]+3, sponum);
				 const p sum2 = d2c[0] * coefs(is[0]+i, is[1]+j, is[2], sponum) +
				   d2c[1] * coefs(is[0]+i, is[1]+j, is[2]+1, sponum) +
				   d2c[2] * coefs(is[0]+i, is[1]+j, is[2]+2, sponum) +
				   d2c[3] * coefs(is[0]+i, is[1]+j, is[2]+3, sponum);
				 
				 hess(walkerNum)(sponum,0) += pre20 * sum0;
				 hess(walkerNum)(sponum,1) += pre11 * sum0;
				 hess(walkerNum)(sponum,2) += pre10 * sum1;
				 hess(walkerNum)(sponum,3) += pre02 * sum0;
				 hess(walkerNum)(sponum,4) += pre01 * sum1;
				 hess(walkerNum)(sponum,5) += pre00 * sum2;
				 grad(walkerNum)(sponum,0) += pre10 * sum0;
				 grad(walkerNum)(sponum,1) += pre01 * sum0;
				 grad(walkerNum)(sponum,2) += pre00 * sum1;
				 vals(walkerNum)(sponum) += pre00 * sum0;
			       });
	}
      }
      
      // again the d's are the same for every member of the team
      const p dxx = delta_invs(0) * delta_invs(0);
      const p dyy = delta_invs(1) * delta_invs(1);
      const p dzz = delta_invs(2) * delta_invs(2);
      const p dxy = delta_invs(0) * delta_invs(1);
      const p dxz = delta_invs(0) * delta_invs(2);
      const p dyz = delta_invs(1) * delta_invs(2);
      
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, num_splines), 
			   [&](const int& n) {
			     const int sponum = start+n;
			     grad(walkerNum)(sponum,0) *= delta_invs(0);
			     grad(walkerNum)(sponum,1) *= delta_invs(1);
			     grad(walkerNum)(sponum,2) *= delta_invs(2);
			     hess(walkerNum)(sponum,0) *= dxx;
			     hess(walkerNum)(sponum,1) *= dyy;
			     hess(walkerNum)(sponum,2) *= dzz;
			     hess(walkerNum)(sponum,3) *= dxy;
			     hess(walkerNum)(sponum,4) *= dxz;
			     hess(walkerNum)(sponum,5) *= dyz;
			   });
    });
}
			       
   
};    




#endif
  
  
