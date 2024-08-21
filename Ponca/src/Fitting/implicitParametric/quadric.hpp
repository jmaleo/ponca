template < class DataPoint, class _WFunctor, typename T>
typename Quadric<DataPoint, _WFunctor, T>::Scalar
Quadric<DataPoint, _WFunctor, T>::potential( const VectorType &_q ) const
{
    // turn to centered basis
    const VectorType lq = Base::m_w.convertToLocalBasis(_q);
    const VectorA toParamters = convertToParameters(lq);
    return toParamters.transpose() * m_coefficients;
}

template < class DataPoint, class _WFunctor, typename T>
typename Quadric<DataPoint, _WFunctor, T>::VectorType
Quadric<DataPoint, _WFunctor, T>::project( const VectorType& _q ) const
{
    PONCA_MULTIARCH_STD_MATH(abs);

    constexpr Scalar epsilon = Eigen::NumTraits<Scalar>::dummy_precision();

    const VectorType grad = primitiveGradient( _q );
    const VectorA gradParameters = convertToParameters(grad);
    // const Scalar a = grad.transpose() * m_uq * grad;

    const Scalar a = gradParameters(0) * m_coefficients(0) +
                        gradParameters(1) * m_coefficients(1) +
                        gradParameters(2) * m_coefficients(2) +
                        gradParameters(3) * m_coefficients(3) +
                        gradParameters(4) * m_coefficients(4) +
                        gradParameters(5) * m_coefficients(5);
    
    const Scalar b = grad.squaredNorm();
    const Scalar c = potential(_q);

    // solve a t^2 + b t + c = 0
    Scalar t = 0;

    if(abs(a) < epsilon)
    {
        if(abs(b) < epsilon)
        {
            t = 0;
        }
        else
        {
            t = - c / b;
        }
    }
    else
    {
        const Scalar delta = b*b - 4*a*c;

        if(delta >= 0)
        {
            t = (- b + sqrt(delta)) / (2 * a);
        }
        else
        {
            t = 0; // no solution so no projection
        }
    }

    return _q + t * grad;
}


template < class DataPoint, class _WFunctor, typename T>
typename Quadric<DataPoint, _WFunctor, T>::VectorType
Quadric<DataPoint, _WFunctor, T>::primitiveGradient( const VectorType &_q ) const
{
        // turn to centered basis
        const VectorType lq = Base::m_w.convertToLocalBasis(_q);

        return VectorType( f_x(lq), f_y(lq), f_z(lq) );
}
