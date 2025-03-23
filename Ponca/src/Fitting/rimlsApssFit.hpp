/*
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


template<class DataPoint, class _WFunctor, typename T>
void RimlsApssFitImpl<DataPoint, _WFunctor, T>::init(const VectorType& _evalPos)
{
    Base::init( _evalPos );

    // Setup internal values
    m_minConvergence = Scalar(1e-4);
    m_maxIteration = 20;
    m_sigmaN = Scalar( 0.5 );

    iteration = 0;
    convergence = Scalar(1);
    f = Scalar(0.);
    gradF = VectorType::Zero();
    prevGrad = VectorType::Zero();
    alpha = Scalar(1.);
    convergence = Scalar(1);

    // Temporary variables
    sumF = Scalar(0);
    sumW = Scalar(0);

    m_mean = Scalar(0);
}

template<class DataPoint, class _WFunctor, typename T>
bool RimlsApssFitImpl<DataPoint, _WFunctor, T>::addLocalNeighbor(Scalar w,
                                                                    const VectorType &localQ,
                                                                    const DataPoint &attributes)
{

    VectorType px = Base::m_w.evalPos() - attributes.pos();
    Scalar fx = Base::potential( attributes.pos() );

    // use gaussian weights
    if( iteration > 0 ) {
        alpha = gaussian( ( attributes.normal() - gradF ).squaredNorm(), m_sigmaN );
        alpha *= gaussian( (fx - f) * (fx - f), 0.5 * Base::m_w.evalScale() );
    }
    Scalar w2 = alpha * w;

    return Base::addLocalNeighbor( w2, localQ, attributes );
}

template<class DataPoint, class _WFunctor, typename T>
FIT_RESULT RimlsApssFitImpl<DataPoint, _WFunctor, T>::finalize()
{

    if( Base::getWeightSum() == 0 ){
        return Base::m_eCurrentState = UNDEFINED;
    }

    if( iteration < m_maxIteration && convergence > m_minConvergence ) {
        Base::finalize();
        prevGrad = gradF;
        f = Base::potential();
        gradF = Base::primitiveGradient();
        convergence = ( prevGrad - gradF ).squaredNorm();
        ++iteration;

        Base::m_sumDotPN = Scalar(0);
        Base::m_sumDotPP = Scalar(0);
        Base::m_sumN = VectorType::Zero();
        Base::m_sumP = VectorType::Zero();
        Base::startNewPass();

        return Base::m_eCurrentState = NEED_OTHER_PASS;
    }
    return Base::m_eCurrentState = STABLE;
}
