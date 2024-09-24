#pragma once

#include <Ponca/Fitting>
#include <Eigen/Eigenvalues>
#include "meanPlaneFit.h"
#include "covariancePlaneFit.h"

#include "./defines.h"

// TODO
// - inherit from plane primitive
// - inherit from curvature class
// - remove v0

namespace Ponca
{

template<class _Scalar>
class VarifoldWeightKernel
{
public:
    using Scalar = _Scalar;

    Scalar f(const Scalar& _x) const {
        const Scalar y = Scalar(1) / (1 - _x*_x);
        //
        // WARNING
        // rho prime is negative but the basket ignores negative weights
        // (see Basket::addNeighbor()), so the opposite is returned here
        // and Varifold::addLocalNeighbor() takes the opposite again
        // 
        // return - 2 * _x * y * y * std::exp(-y);
        return + 2 * _x * y * y * std::exp(-y);
    }
};

// ============================================================================


//! 
//! Implementation of 
//!     Weak and approximate curvatures of a measure: a varifold perspective
//!     B Buet, GP Leonardi, S Masnou
//!     
//! See Equation 7.2 of the article
//!
//! The _WFunctor must be DistWeightFunc<D,VarifoldWeightKernel<S>>.
//! Since rho' is negative and the basket ignores negative weights,
//! VarifoldWeightKernel::f() returns the opposite (positive) weight
//! and Varifold::addLocalNeighbor() takes the opposite back (see WARNING).
//! 
//! Remarks
//! - point weights (m_l) are set to 1
//! - only valid in 3D
//! - normals are required
//! 
template<class DataPoint, class _WFunctor, typename T>
class VarifoldsImpl : public T
{
    PONCA_FITTING_DECLARE_DEFAULT_TYPES
    PONCA_FITTING_DECLARE_MATRIX_TYPE

protected:

    enum
    {
        Check = Base::PROVIDES_PRIMITIVE_BASE &&
                Base::PROVIDES_LOCAL_FRAME
    };


    using Mat32 = Eigen::Matrix<Scalar,3,2>;
    using Mat22 = Eigen::Matrix<Scalar,2,2>;

    static_assert(DataPoint::Dim == 3); // only valid in 3D
    static_assert(std::is_same<_WFunctor, DistWeightFunc<DataPoint,VarifoldWeightKernel<Scalar>>>());

public:
    // input
    VectorType m_n_l0;
    MatrixType m_P_l0;

    // accumulation
    MatrixType m_nume;
    Scalar m_deno;

    // results
    Scalar m_k1, m_k2;
    VectorType m_dir1, m_dir2;

    // plane
    bool m_planeIsReady = false;
    VectorType m_sumN;      // sum of normals
    VectorType m_sumP;      // sum of points
    Scalar     m_sumWeight; // sum of weights


protected:
    Mat32 tangentPlane() const;

public:
    PONCA_EXPLICIT_CAST_OPERATORS(VarifoldsImpl,varifoldsImpl)
    // void init(const VectorType& x_l0, const VectorType& n_l0);
    void init(const VectorType& _evalPos);
    PONCA_FITTING_DECLARE_ADDNEIGHBOR
    PONCA_FITTING_DECLARE_FINALIZE

    PONCA_MULTIARCH inline Scalar f_smooth  (const Scalar& _x) const { Scalar v = _x*_x - Scalar(1.); return v*v; }
    // PONCA_MULTIARCH inline Scalar f_smooth  (const Scalar& _x) const { Scalar v = Scalar(1); return v; }

    // plane
    VectorType project(const VectorType& p) const;
    VectorType primitiveGradient (const VectorType& p) const;
    VectorType primitiveGradient () const { return primitiveGradient(Base::m_w.basisCenter()); }

    PONCA_MULTIARCH void reorientPlane();


    PONCA_MULTIARCH inline Scalar kmin () const { return m_k1; }
    PONCA_MULTIARCH inline Scalar kmax () const { return m_k2; }
    PONCA_MULTIARCH inline Scalar kMean () const { return (m_k1 + m_k2)/2; }
    PONCA_MULTIARCH inline Scalar GaussianCurvature () const { return m_k1 * m_k2; }
    
    
    PONCA_MULTIARCH inline VectorType kminDirection() const { return m_dir1; }
    PONCA_MULTIARCH inline VectorType kmaxDirection() const { return m_dir2; }

};

template<class D, class W, typename T>
//using Varifold = v1::Varifold<D,W,>;
 using VarifoldsCovPlane = VarifoldsImpl<D,W,
    Ponca::CovariancePlaneFitImpl<D, W,
        Ponca::CovarianceFitBase<D, W,
            Ponca::MeanPosition<D, W,
                Ponca::MeanNormal<D, W,
                    Ponca::LocalFrame<D, W,
                        Ponca::Plane<D, W, T>>>>>>>;

template<class D, class W, typename T>
//using Varifold = v1::Varifold<D,W,>;
 using VarifoldsMeanPlane = VarifoldsImpl<D,W,
    Ponca::MeanPlaneFitImpl<D, W,
            Ponca::MeanPosition<D, W,
                Ponca::MeanNormal<D, W,
                    Ponca::LocalFrame<D, W,
                        Ponca::Plane<D, W, T>>>>>>;

#include "varifolds.hpp"
} //namespace Ponca
