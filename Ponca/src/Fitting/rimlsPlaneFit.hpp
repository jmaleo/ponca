/*
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


template<class DataPoint, class _WFunctor, typename T>
void RimlsPlaneFitImpl<DataPoint, _WFunctor, T>::init(const VectorType& _evalPos)
{
    Base::init( _evalPos );

    // Setup internal values
    m_minConvergence = Scalar(1e-3);
    m_maxIteration = 10;
    m_sigmaN = Scalar( 0.5 );

    iteration = 0;
    convergence = Scalar(1);
    f = Scalar(0.);
    gradF = VectorType::Zero();
    prevGrad = VectorType::Zero();
    alpha = Scalar(1.);
    convergence = Scalar(1);

    // Temporary variables
    sumGW = VectorType::Zero();
    sumW = Scalar(0);
    sumF = Scalar(0);
    sumGF = VectorType::Zero();
    sumN = VectorType::Zero();
}

template<class DataPoint, class _WFunctor, typename T>
bool RimlsPlaneFitImpl<DataPoint, _WFunctor, T>::addLocalNeighbor(Scalar w,
                                                                    const VectorType &localQ,
                                                                    const DataPoint &attributes)
{

    //if valid, the neighbor is added to the neighbors vector
    if( Base::addLocalNeighbor( w, localQ, attributes ) ){

        VectorType px = Base::m_w.evalPos() - attributes.pos();
        Scalar fx = px.dot( attributes.normal() );

        // use gaussian weights
        if( iteration > 0 ) {
            alpha = gaussian( ( attributes.normal() - gradF ).squaredNorm(), m_sigmaN );
            alpha *= gaussian( (fx - f) * (fx - f), 0.5 * Base::m_w.evalScale() );
        }

        Scalar w = alpha * Base::m_w.w( attributes.pos(), attributes ).first;
        VectorType gradW = alpha * 2 * px * Base::m_w.scaledw(px, attributes);

        sumW += w;
        sumGW += gradW;
        sumF += w * fx;
        sumGF += gradW * fx;
        sumN += w * attributes.normal();

        return true;
    }
    return false;
}

template<class DataPoint, class _WFunctor, typename T>
FIT_RESULT RimlsPlaneFitImpl<DataPoint, _WFunctor, T>::finalize()
{

    // handle UNDEFINED cases
    if( Base::finalize() != STABLE ) {
        return Base::m_eCurrentState = UNDEFINED;
    }

    // handle conflict error
    if( Base::plane().isValid() ){
        return Base::m_eCurrentState = CONFLICT_ERROR_FOUND;
    }

    if( sumW == 0 ){
        return Base::m_eCurrentState = UNDEFINED;
    }


    VectorType px;
    Scalar fx;
    Scalar w;
    VectorType gradW;

    if( iteration < m_maxIteration && convergence > m_minConvergence ) {
        prevGrad = gradF;

        // update the actual potential and gradient
        f = sumF / sumW;
        gradF = ( sumGF - f * sumGW + sumN ) / sumW;
        convergence = ( prevGrad - gradF ).squaredNorm();
        ++iteration;

        sumGW = VectorType::Zero();
        sumW = Scalar(0);
        sumF = Scalar(0);
        sumGF = VectorType::Zero();
        sumN = VectorType::Zero();

        return Base::m_eCurrentState = NEED_OTHER_PASS;
    }
    Base::setPlane( -gradF, -f * gradF );
    return Base::m_eCurrentState = STABLE;
}
