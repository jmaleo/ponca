
#include <Eigen/SVD>
#include <Eigen/Geometry>

template < class DataPoint, class _WFunctor, typename T>
void
NormalCovariance3D<DataPoint, _WFunctor, T>::init(const VectorType& _evalPos)
{
    Base::init(_evalPos);

    m_cov.setZero();
    m_normal_centroid.setZero();
    m_minDirIndex = 0;
    m_maxDirIndex = 0;

    m_P.setZero();
    m_W.setZero();
}

template < class DataPoint, class _WFunctor, typename T>
bool
NormalCovariance3D<DataPoint, _WFunctor, T>::addLocalNeighbor(Scalar w,
                                                      const VectorType &localQ,
                                                      const DataPoint &attributes)
{
    bool res = Base::addLocalNeighbor(w, localQ, attributes);
    if( ! res ){
        return res;
    }
    else {
        m_cov  += w * attributes.normal() * attributes.normal().transpose();
        m_normal_centroid += w * attributes.normal();
        return true;
    }
    return false;
}

template < class DataPoint, class _WFunctor, typename T>
FIT_RESULT
NormalCovariance3D<DataPoint, _WFunctor, T>::finalize ()
{
    FIT_RESULT res = Base::finalize();
    if ( ! res == FIT_RESULT::STABLE ) {
        return res;
    }
    else {
        m_normal_centroid /= Base::getWeightSum();
        m_cov = ( m_cov / Base::getWeightSum() ) - ( m_normal_centroid * m_normal_centroid.transpose() );
        // m_cov = (m_cov + m_cov.transpose()) / Scalar(2);
        setTangentPlane();
        m_W = m_P.transpose() * m_cov * m_P;
        m_solver.compute(m_W);

        if (m_solver.info() != Eigen::Success) {
            return Base::m_eCurrentState = UNDEFINED;
        }
        
        return Base::m_eCurrentState = STABLE;
    }
}

template < class DataPoint, class _WFunctor, typename T>
typename NormalCovariance3D<DataPoint, _WFunctor, T>::Scalar
NormalCovariance3D<DataPoint, _WFunctor, T>::kMean() const {
  return ( kmin() + kmax() ) / Scalar(2);
}

template < class DataPoint, class _WFunctor, typename T>
typename NormalCovariance3D<DataPoint, _WFunctor, T>::Scalar
NormalCovariance3D<DataPoint, _WFunctor, T>::GaussianCurvature() const {
    return kmin() * kmax();
}

template < class DataPoint, class _WFunctor, typename T>
typename NormalCovariance3D<DataPoint, _WFunctor, T>::Scalar
NormalCovariance3D<DataPoint, _WFunctor, T>::kmin() const {
    PONCA_MULTIARCH_STD_MATH(pow);
    PONCA_MULTIARCH_STD_MATH(sqrt);
    constexpr Scalar four = Scalar(4);
    constexpr Scalar PI = Scalar(M_PI);
    Scalar factor = pow ( Base::m_w.evalScale(), four ) * PI / four;
    return sqrt( m_solver.eigenvalues()(0) / factor );
}

template < class DataPoint, class _WFunctor, typename T>
typename NormalCovariance3D<DataPoint, _WFunctor, T>::Scalar
NormalCovariance3D<DataPoint, _WFunctor, T>::kmax() const {
    PONCA_MULTIARCH_STD_MATH(pow);
    PONCA_MULTIARCH_STD_MATH(sqrt);
    constexpr Scalar four = Scalar(4);
    constexpr Scalar PI = Scalar(M_PI);
    Scalar factor = pow ( Base::m_w.evalScale(), four ) * PI / four;
    return sqrt( m_solver.eigenvalues()(1) / factor );
}

template < class DataPoint, class _WFunctor, typename T>
typename NormalCovariance3D<DataPoint, _WFunctor, T>::VectorType
NormalCovariance3D<DataPoint, _WFunctor, T>::kminDirection() const {
    VectorType kMinDirection = m_P * m_solver.eigenvectors().col(0);
    return kMinDirection;
}

template < class DataPoint, class _WFunctor, typename T>
typename NormalCovariance3D<DataPoint, _WFunctor, T>::VectorType
NormalCovariance3D<DataPoint, _WFunctor, T>::kmaxDirection() const {
    VectorType kMaxDirection = m_P * m_solver.eigenvectors().col(1);
    return kMaxDirection;
}
