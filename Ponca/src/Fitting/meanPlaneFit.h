/*
 Copyright (C) 2018 Nicolas Mellado <nmellado0@gmail.com>

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include "./plane.h"
#include "./mean.h"
#include "./planeFrame.h"

namespace Ponca
{

/*!
    \brief Plane fitting procedure computing the frame plane using mean position and normal

    \inherit Concept::FittingProcedureConcept

    \see Plane
    \see PlaneFrame
*/
template < class DataPoint, class _WFunctor, typename T >
class MeanPlaneFitImpl : public T
{
PONCA_FITTING_DECLARE_DEFAULT_TYPES
PONCA_FITTING_DECLARE_MATRIX_TYPE

protected:
    enum { Check = Base::PROVIDES_PLANE_FRAME and Base::PROVIDES_PLANE };


public:
    PONCA_EXPLICIT_CAST_OPERATORS(MeanPlaneFitImpl,meanPlaneFit)

    PONCA_FITTING_APIDOC_FINALIZE
    PONCA_MULTIARCH inline FIT_RESULT finalize()
    {
        // handle specific configurations
        if(Base::finalize() == STABLE)
        {
            if (Base::plane().isValid()) Base::m_eCurrentState = CONFLICT_ERROR_FOUND;
            Base::setPlane(Base::m_sumN / Base::m_sumW, Base::barycenter());
            VectorType norm = Base::plane().normal();
            VectorType a;
            if (std::abs(norm.x()) > std::abs(norm.z())) {
                a = VectorType(-norm.y(), norm.x(), 0);
            } else {
                a = VectorType(0, -norm.z(), norm.y());
            }
            a.normalize();
            // use cross product to generate a orthogonal basis
            Base::m_u = norm.cross(a);
            Base::m_u.normalize();
            Base::m_v = norm.cross(Base::m_u);
            Base::m_v.normalize();
        }
        return Base::m_eCurrentState;
    }

}; //class MeanPlaneFitImpl

/// \brief Helper alias for Plane fitting on points using MeanPlaneFitImpl
//! [MeanPlaneFit Definition]
    template < class DataPoint, class _WFunctor, typename T>
    using MeanPlaneFit =
    MeanPlaneFitImpl<DataPoint, _WFunctor,
        MeanNormal<DataPoint, _WFunctor,
            MeanPosition<DataPoint, _WFunctor,
                PlaneFrame<DataPoint, _WFunctor,
                    Plane<DataPoint, _WFunctor,T>>>>>;
//! [MeanPlaneFit Definition]

} //namespace Ponca
