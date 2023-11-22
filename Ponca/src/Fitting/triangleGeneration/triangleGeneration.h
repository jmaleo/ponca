/*
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#pragma once

#include "../defines.h"

#include PONCA_MULTIARCH_INCLUDE_STD(cmath)
#include PONCA_MULTIARCH_INCLUDE_STD(limits)

#include "./CNCEigen.h"

// triangle storing indices of points
template < class DataPoint >
struct Triangle {

    public : 

        typedef typename DataPoint::Scalar Scalar;
        typedef typename DataPoint::VectorType VectorType;        
        typedef typename DataPoint::MatrixType MatrixType;
        
        std::array < VectorType, 3 > points;
        std::array < VectorType, 3 > normals;
        // Maybe need to store the normal too

    public :

        Triangle(const std::array < VectorType, 3 >& points, const std::array < VectorType, 3 >& normals) {
            this->points = points;
            this->normals = normals;
        }

        inline bool operator==(const Triangle& other) const {
            return (points[0] == other.points[0]) && (points[1] == other.points[1]) && (points[2] == other.points[2]);
        }

        inline bool operator!=(const Triangle& other) const {
            return !((*this) == other);
        }

        inline Scalar mu0InterpolatedU      () { return CNCEigen::mu0InterpolatedU  (points[0], points[1], points[2], normals[0], normals[1], normals[2]); }

        inline Scalar mu1InterpolatedU      () { return CNCEigen::mu1InterpolatedU  (points[0], points[1], points[2], normals[0], normals[1], normals[2]); }

        inline Scalar mu2InterpolatedU      () { return CNCEigen::mu2InterpolatedU  (points[0], points[1], points[2], normals[0], normals[1], normals[2]); }

        inline MatrixType muXYInterpolatedU () { return CNCEigen::muXYInterpolatedU (points[0], points[1], points[2], normals[0], normals[1], normals[2]); }

};

namespace Ponca
{

/*!
    \brief CNC generation of triangles from a set of points

    \cite coucou 

    \todo the code
*/


template < class DataPoint, class _WFunctor, typename T>
class TriangleGeneration : public T
{
    PONCA_FITTING_DECLARE_DEFAULT_TYPES
    PONCA_FITTING_DECLARE_MATRIX_TYPE
    typedef Eigen::VectorXd  DenseVector;
    typedef Eigen::MatrixXd  DenseMatrix;

protected:
    enum
    {
        PROVIDES_TRIANGLE_BASE                      /*!< \brief Provides Generated Triangle basis */
    };

    enum class Method {
        UniformGeneration, 
        IndependentGeneration, 
        HexagramGeneration, 
        AvgHexagramGeneration
    };

protected:
    //! \brief protected variables
    VectorType _normale;

    std::array < Scalar, 6 > _cos;
    std::array < Scalar, 6 > _sin;

    int _nb_vt {0}; // Number of valide triangles
    std::vector < Triangle < DataPoint > > _triangles;
    Scalar _A {0}; // Area
    Scalar _H {0}; // Mean Curvatures
    Scalar _G {0}; // Gaussian Curvatures
    Scalar _T11 {0}; // T11
    Scalar _T12 {0}; // T12
    Scalar _T13 {0}; // T13
    Scalar _T22 {0}; // T22
    Scalar _T23 {0}; // T23
    Scalar _T33 {0}; // T33

    Scalar k1 {0};
    Scalar k2 {0};

    VectorType v1;
    VectorType v2;


    // Hexagram
    std::array< Scalar    ,    6 > _distance2;
    std::array< VectorType,    6 > _targets;

// results
public:
    /*!< \brief Parameters of the triangles */
    int _maxtriangles {100};
    Scalar _avgnormals {Scalar(0.5)};
    Method _method {Method::UniformGeneration};

public:
    PONCA_EXPLICIT_CAST_OPERATORS(TriangleGeneration,triangleGeneration)
    PONCA_FITTING_DECLARE_FINALIZE

    /*! \brief Set the scalar field values to 0 and reset the isNormalized() status

    */
    PONCA_MULTIARCH inline void init(const VectorType& _basisCenter)
    {
        Base::init(_basisCenter);

        _normale = VectorType::Zero();

        k1 = Scalar(0);
        k2 = Scalar(0);

        v1 = VectorType::Zero();
        v2 = VectorType::Zero();

        // Instantiate the parameters
        _maxtriangles = 100;
        _method = Method::UniformGeneration;
        for ( int j = 0; j < 6; j++ )
        {
            const Scalar a = j * M_PI / 3.0;
            _cos[ j ] = std::cos( a );
            _sin[ j ] = std::sin( a );
        }

    }

    PONCA_MULTIARCH inline bool computeNeighbors(const DataPoint &evalPoint, const std::vector<VectorType>& _attribNeigs, const std::vector<VectorType>& _normNeigs);

    PONCA_MULTIARCH inline void chooseUniformGeneration( int maxtriangles = 100 ) {
        _method = Method::UniformGeneration;
        _maxtriangles = maxtriangles;
    }

    PONCA_MULTIARCH inline void chooseIndependentGeneration( int maxtriangles = 100 ) {
        _method = Method::IndependentGeneration;
        _maxtriangles = maxtriangles;
    }

    PONCA_MULTIARCH inline void chooseHexagramGeneration( Scalar average_normals_weight = 0.5 ) {
        _method = Method::HexagramGeneration;
        _avgnormals = average_normals_weight;
    }

    PONCA_MULTIARCH inline void chooseAvgHexagramGeneration( Scalar average_normals_weight = 0.5 ) {
        _method = Method::AvgHexagramGeneration;
        _avgnormals = average_normals_weight;
    }

    PONCA_MULTIARCH inline int getNumTriangles () {
        return _nb_vt;
    }

    PONCA_MULTIARCH inline void getTriangles(std::vector<std::array<Scalar, 3>>& triangles) {

        for (int i = 0; i < _triangles.size(); i++) {
            std::array <Scalar, 3> point0 = {_triangles[i].points[0][0], _triangles[i].points[0][1], _triangles[i].points[0][2]};
            std::array <Scalar, 3> point1 = {_triangles[i].points[1][0], _triangles[i].points[1][1], _triangles[i].points[1][2]};
            std::array <Scalar, 3> point2 = {_triangles[i].points[2][0], _triangles[i].points[2][1], _triangles[i].points[2][2]};
            triangles.push_back(point0);
            triangles.push_back(point1);
            triangles.push_back(point2);        
        }

    }

    PONCA_MULTIARCH inline Scalar kmin() { return k1; }

    PONCA_MULTIARCH inline Scalar kmax() { return k2; }

    PONCA_MULTIARCH inline VectorType kminDirection() { return v1; }

    PONCA_MULTIARCH inline VectorType kmaxDirection() { return v2; }

    PONCA_MULTIARCH inline Scalar kMean() { return _H; }

    PONCA_MULTIARCH inline Scalar kgauss() { return _G; }

private: 

    PONCA_MULTIARCH inline void construct_hexa(const DataPoint &evalPoint, const std::vector<VectorType>& _attribNeigs, const std::vector<VectorType>& _normNeigs);

    PONCA_MULTIARCH inline void construct_avgHexa(const DataPoint &evalPoint, const std::vector<VectorType>& _attribNeigs, const std::vector<VectorType>& _normNeigs);



}; //class triangleGeneration

#include "triangleGeneration.hpp"
} //namespace Ponca
