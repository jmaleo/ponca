template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::Scalar
ParabolicCylinder<DataPoint, _WFunctor, T>::potential( const VectorType &_q ) const
{
    VectorType x = Base::worldToLocalFrame(_q);
    return m_correctOrientation * eval_quadratic_function(*(x.data() +1 ), *(x.data() + 2)) - *(x.data());
}


template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::VectorType
ParabolicCylinder<DataPoint, _WFunctor, T>::project( const VectorType& _q ) const
{
    PONCA_MULTIARCH_STD_MATH(abs);
    constexpr Scalar epsilon = Eigen::NumTraits<Scalar>::dummy_precision();
    VectorType x = Base::worldToLocalFrame(_q);
    x(0) = eval_quadratic_function(x(1), x(2));
    return Base::localFrameToWorld(x);
}

template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::VectorType
ParabolicCylinder<DataPoint, _WFunctor, T>::primitiveGradient( const VectorType& _q ) const
{
    // Convexe = m_a >= 0    Concave = m_a <= 0

    VectorType proj = Base::worldToLocalFrame(_q);
    Vector2 temp {proj(1),  proj(2)};
    Vector2 df = m_ul + 2 * m_a * m_uq * temp;
    VectorType local_gradient { 1, -df(0) , -df(1) };
    local_gradient *= m_correctOrientation;

    VectorType world_gradient = Base::template localFrameToWorld<true>(local_gradient);

    return world_gradient;
}


template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::MatrixType
ParabolicCylinder<DataPoint, _WFunctor, T>::dNormal() const
{ 
    // Grab the normal (primitiv gradient)
    VectorType u = Base::kminDirection();
    VectorType v = Base::kmaxDirection();
    VectorType n = primitiveGradient();

    MatrixType B;
    B << n, u, v;

    Matrix2 dN_2D = (2 * m_a * m_uq);
    MatrixType dN = MatrixType::Zero();
    dN.block(1,1,2,2) = dN_2D;
    // put 1 on the first row and column
    dN(0,0) = 1; 

    VectorType ul {1, m_ul(0), m_ul(1)};
    
    return ( B * dN * B.transpose() ) / Base::template localFrameToWorld<true>(ul).norm(); 

}

template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::Scalar
ParabolicCylinder<DataPoint, _WFunctor, T>::kMean() const {
  PONCA_MULTIARCH_STD_MATH(pow);
  static const Scalar one (1);
  static const Scalar two (2);
  static const Scalar threeOverTwo (Scalar(3)/Scalar(2));
  return ( dh_uu() * ( one + pow( dh_v(), two ) ) + dh_vv() * ( one + pow( dh_u(), two ) ) - two * dh_uv() * dh_u() * dh_v() ) /
    ( two * pow( one + pow( dh_u(), two ) + pow( dh_v(), two ), threeOverTwo ) );
}

template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::Scalar
ParabolicCylinder<DataPoint, _WFunctor, T>::GaussianCurvature() const {
    PONCA_MULTIARCH_STD_MATH(pow);
    static const Scalar one (1);
    static const Scalar two (2);
    return ( dh_uu()* dh_vv() - pow(dh_uv(),two)) /
        pow(( pow(dh_u(),two) + pow(dh_v(),two) + one ), two);
}

template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::Scalar
ParabolicCylinder<DataPoint, _WFunctor, T>::kmin() const {
    Scalar mean = kMean();
    Scalar gauss = GaussianCurvature();
    return mean - sqrt(mean*mean - gauss);
}

template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::Scalar
ParabolicCylinder<DataPoint, _WFunctor, T>::kmax() const {
    Scalar mean = kMean();
    Scalar gauss = GaussianCurvature();
    return mean + sqrt(mean*mean - gauss);
}

template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::VectorType
ParabolicCylinder<DataPoint, _WFunctor, T>::kminDirection() const {
    Eigen::Matrix<Scalar, 2, 2> I, II, W;  
    I << dE(), dF(), dF(), dG();
    II << dL(), dM(), dM(), dN();
    W = I.inverse() * II;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 2, 2>> eigW(W);
    Eigen::Vector<Scalar, 2> dir = eigW.eigenvectors().col(0);

    VectorType v1 = VectorType(0, dir(0), dir(1));
    return Base::template localFrameToWorld<true>(v1);
}

template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::VectorType
ParabolicCylinder<DataPoint, _WFunctor, T>::kmaxDirection() const {

    Eigen::Matrix<Scalar, 2, 2> I, II, W;  
    I << dE(), dF(), dF(), dG();
    II << dL(), dM(), dM(), dN();
    W = I.inverse() * II;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 2, 2>> eigW(W);
    Eigen::Vector<Scalar, 2> dir = eigW.eigenvectors().col(1);

    VectorType v2 = VectorType(0, dir(0), dir(1));
    return Base::template localFrameToWorld<true>(v2);
}