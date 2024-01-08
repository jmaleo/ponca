/*
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include "../../indexSquaredDistance.h"
#include "../../../Common/Containers/stack.h"

namespace Ponca {
template <typename Traits> class KdTreeBase;

template <typename Traits>
class KdTreeQuery
{
public:
    using DataPoint  = typename Traits::DataPoint;
    using IndexType  = typename Traits::IndexType;
    using Scalar     = typename DataPoint::Scalar;
    using VectorType = typename DataPoint::VectorType;

    explicit inline KdTreeQuery(const KdTreeBase<Traits>* kdtree) : m_kdtree( kdtree ), m_stack() {}

protected:
    /// \brief Init stack for a new search
    inline void reset() {
        m_stack.clear();
        m_stack.push({0,0});
    }

    /// [KdTreeQuery kdtree type]
    const KdTreeBase<Traits>* m_kdtree { nullptr };
    /// [KdTreeQuery kdtree type]
    Stack<IndexSquaredDistance<IndexType, Scalar>, 2 * Traits::MAX_DEPTH> m_stack;

    template<typename LeafPreparationFunctor,
            typename DescentDistanceThresholdFunctor,
            typename SkipIndexFunctor,
            typename ProcessNeighborFunctor>
    bool search_internal(const VectorType& point,
                         LeafPreparationFunctor prepareLeafTraversal,
                         DescentDistanceThresholdFunctor descentDistanceThreshold,
                         SkipIndexFunctor skipFunctor,
                         ProcessNeighborFunctor processNeighborFunctor
                         )
    {
        const auto& nodes  = m_kdtree->nodes();
        const auto& points = m_kdtree->points();

        if (nodes.empty() || points.empty() || m_kdtree->sample_count() == 0)
            throw std::invalid_argument("Empty KdTree");

        while(!m_stack.empty())
        {
            auto& qnode = m_stack.top();
            const auto& node = nodes[qnode.index];

            if(qnode.squared_distance < descentDistanceThreshold())
            {
                if(node.is_leaf())
                {
                    m_stack.pop();
                    IndexType start = node.leaf_start();
                    IndexType end = node.leaf_start() + node.leaf_size();
                    prepareLeafTraversal(start, end);
                    for(IndexType i=start; i<end; ++i)
                    {
                        IndexType idx = m_kdtree->pointFromSample(i);
                        if(skipFunctor(idx)) continue;

                        Scalar d = (point - points[idx].pos()).squaredNorm();

                        if(d < descentDistanceThreshold())
                        {
                            if( processNeighborFunctor( idx, i, d )) return false;
                        }
                    }
                }
                else
                {
                    // replace the stack top by the farthest and push the closest
                    Scalar newOff = point[node.inner_split_dim()] - node.inner_split_value();
                    m_stack.push();
                    if(newOff < 0)
                    {
                        m_stack.top().index = node.inner_first_child_id();
                        qnode.index         = node.inner_first_child_id()+1;
                    }
                    else
                    {
                        m_stack.top().index = node.inner_first_child_id()+1;
                        qnode.index         = node.inner_first_child_id();
                    }
                    m_stack.top().squared_distance = qnode.squared_distance;
                    qnode.squared_distance         = newOff*newOff;
                }
            }
            else
            {
                m_stack.pop();
            }
        }
        return true;
    }
};
} // namespace Ponca
