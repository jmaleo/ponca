
template < class DataPoint, class _WFunctor, typename T>
bool
WaveJets<DataPoint, _WFunctor, T>::addLocalNeighbor(Scalar w,
                                                    const VectorType &localQ,
                                                    const DataPoint &attributes)
{
    auto res = Base::addLocalNeighbor(w, localQ, attributes);
    if(! m_planeIsReady)
    {   
        return res; // To change into the IF if the master is modified
    }
    else // base plane is ready, we can now fit the demi ellipsoid / Cylinder
    {
        if (! m_normalIsCorrect){
            // express neighbor in local coordinate frame
            VectorType localPos = Base::worldToLocalFrame(attributes.pos());
            // Vector2 planePos = Vector2 ( *(localPos.data()+1), *(localPos.data()+2) );

            return m_local_jet_pass(localPos);
        }
        else {
            // express neighbor in corrected local coordinate frame
            VectorType localPos = m_P.transpose() * (Base::m_w.convertToLocalBasis(attributes.pos()));
            // Vector2 planePos = Vector2 ( *(localPos.data()+1), *(localPos.data()+2) );
            return m_local_jet_pass(localPos);
        }
    }
    return res;
}

template < class DataPoint, class _WFunctor, typename T>
FIT_RESULT
WaveJets<DataPoint, _WFunctor, T>::finalize () {
    PONCA_MULTIARCH_STD_MATH(abs);
    if (! m_planeIsReady) {

        FIT_RESULT res = Base::finalize();

        if (res == STABLE) {
            m_planeIsReady = true;
            m_b.resize(Base::getNumNeighbors());
            m_M.resize(Base::getNumNeighbors(), m_ncolM);
            m_b.setZero();
            m_M.setZero();
            return Base::m_eCurrentState = NEED_OTHER_PASS;
        }
        return res;
    }
    else {

        if (! m_normalIsCorrect){
            m_normalIsCorrect = true;
            FIT_RESULT res = m_correct_normal_process();
            return res;
        }
        else {
            FIT_RESULT res = m_jet_process();
            return res;
        }
    }
    return Base::m_eCurrentState = STABLE;
}

template < class DataPoint, class _WFunctor, typename T>
bool
WaveJets<DataPoint, _WFunctor, T>::m_local_jet_pass(const VectorType& localPos){
    
    const auto I     = std::complex<Scalar>(0,1); // complex number 0+i

    Vector2 planePos = Vector2 ( *(localPos.data()+1), *(localPos.data()+2) );
    // Switch to polar coordinates
    Scalar r = planePos.norm();
    Scalar theta = atan2(*(localPos.data()+2), *(localPos.data()+1)); 
    Scalar z = *(localPos.data());

    // TODO : check if this is correct, or if I need an extra parameter m_radius (instead of Base::m_w.evalScale()
    Scalar normalized_r = r / Base::m_w.evalScale(); 
    z = z / Base::m_w.evalScale(); // TODO : Same check here

    Scalar w = std::exp(-normalized_r * normalized_r/18.0);

    // Compute the coefficients of the polynomial
    int idx = 0;
    for (int k = 0; k <= m_order; ++k){
        Scalar rk = std::pow(normalized_r, k);
        for(int n = -k; n <= k; n += 2){
            m_M(m_idx_j,idx) = rk * std::exp(I * Scalar(n) * theta) * w;
            ++idx;
        }
    }
    m_b[m_idx_j] = z * w;
    ++m_idx_j;

    return (Base::getNumNeighbors() >= 3); // TODO CHECK IF ITS OK
}

template < class DataPoint, class _WFunctor, typename T>
FIT_RESULT
WaveJets<DataPoint, _WFunctor, T>::m_correct_normal_process(){

    Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, 1> Phi = m_M.colPivHouseholderQr().solve(m_b);

    VectorType t1 = Base::getFrameU();
    VectorType t2 = Base::getFrameV();
    VectorType normal = Base::primitiveGradient();

    VectorType u = - t1 * std::imag(Phi(1) - Phi(2)) + t2 * std::real(Phi(2) + Phi(1)); // % axis
    u.normalize();
    Scalar gamma = - std::atan( 2 * std::abs(Phi(2))); // % angle

    VectorType nc = std::cos(gamma) * normal + (1 - std::cos(gamma)) * (normal.dot(u)) * u + std::sin(gamma) * u.cross(normal);
    nc.normalize();
    VectorType t1c = t1 - (t1.transpose() * nc) * nc;
    t1c.normalize();
    VectorType t2c = nc.cross(t1c);
    t2c.normalize();

    // reset m_M and m_b to compute with new normal
    m_M.setZero();
    m_b.setZero();
    m_idx_j = 0;

    MatrixType B;
    B << nc, t1c, t2c;
    m_P = B;

    return FIT_RESULT::NEED_OTHER_PASS;
}

template < class DataPoint, class _WFunctor, typename T>
FIT_RESULT
WaveJets<DataPoint, _WFunctor, T>::m_jet_process(){
    
    Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, 1> Phicorr = m_M.colPivHouseholderQr().solve(m_b);

    // % compute real(a0) and |an| n>=1
    //    idx = 1;
    //    an=zeros(order+1,1);
    //    for k=0:order
    //        for n = -k:2:k
    //          if n>=0
    //           an(n+1) = an(n+1) + Phicorr(idx)/(k+2);
    //          end
    //          idx=idx+1;
    //        end
    //    end
    //    an(1)=real(an(1));%imaginary part is 0
    //    an(2:end)=abs(an(2:end));


    // std::complex<Scalar> phi_0_0  = Phicorr(0);
    // std::complex<Scalar> phi_1_m1 = Phicorr(1);
    // std::complex<Scalar> phi_1_p1 = Phicorr(2);
    std::complex<Scalar> phi_2_m2 = Scalar(-1) * Phicorr(3);
    std::complex<Scalar> phi_2_0  = Scalar(-1) * Phicorr(4);
    std::complex<Scalar> phi_2_p2 = Scalar(-1) * Phicorr(5);

    // corrected normal !
    VectorType N = m_P.col(0);

    // m_k1 = Scalar(2) * Scalar(std::real(phi_2_0 + phi_2_p2 + phi_2_m2));
    // m_k2 = Scalar(2) * Scalar(std::real(phi_2_0 - phi_2_p2 - phi_2_m2));

    Scalar twoPhi_2_0 = Scalar(2) * std::real(phi_2_0);
    Scalar fourSqrtPhi2_m2Phi2_p2 = Scalar(4) * std::sqrt(std::real(phi_2_m2 * phi_2_p2));

    m_k1 = twoPhi_2_0 - fourSqrtPhi2_m2Phi2_p2;
    m_k2 = twoPhi_2_0 + fourSqrtPhi2_m2Phi2_p2;

    // if(m_k2 < m_k1) std::swap(m_k1, m_k2);

    // reset to original scale TODO : check if it's correct with Base::m_w.evalScale()
    m_k1 /= Base::m_w.evalScale();
    m_k2 /= Base::m_w.evalScale();

    // The signal contains a constant component (phi_2_0) and a component that oscillates two times and whose maximum is aligned with the first principal curvature direction (corresponding to the phase of phi_2_p2).
    Scalar theta = std::arg(phi_2_p2);
    m_v1 = std::cos(theta) * m_P.col(1) + std::sin(theta) * m_P.col(2);
    m_v1 = m_v1.normalized();
    m_v2 = N.cross(m_v1).normalized();

    return Base::m_eCurrentState = STABLE;
}
