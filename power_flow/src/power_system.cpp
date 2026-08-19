#include "power_system.h"

#include <cmath>
#include <stdexcept>

namespace powerflow {

void PowerSystem::buildAdmittance(SparseMatrix& Yg, SparseMatrix& Yb) {
    const int n = static_cast<int>(buses.size());
    Yg.resize(n, n);
    Yb.resize(n, n);

    // Reserve roughly 4 off-diagonal + diagonal entries per branch.
    const size_t est = branches.size() * 6 + buses.size() * 2;
    Yg.reserve(est);
    Yb.reserve(est);

    for (const auto& br : branches) {
        const int f = br.from;
        const int t = br.to;
        if (f < 0 || f >= n || t < 0 || t >= n) {
            throw std::out_of_range("PowerSystem::buildAdmittance branch index");
        }

        const double r = br.r;
        const double x = br.x;
        const double z2 = r * r + x * x;
        if (z2 == 0.0) {
            throw std::invalid_argument("PowerSystem: zero-impedance branch");
        }
        // Series admittance y = 1/(r+jx) = (r - jx)/|z|^2.
        const double g_ser =  r / z2;
        const double b_ser = -x / z2;
        const double b_half = 0.5 * br.b;

        const double a = br.tap;
        if (a == 0.0) throw std::invalid_argument("PowerSystem: zero tap ratio");
        const double inv_a = 1.0 / a;
        const double inv_a2 = 1.0 / (a * a);
        const double cphi = std::cos(br.phi);
        const double sphi = std::sin(br.phi);

        // factor = (cos phi + j sin phi) / a  == 1 / conj(tap)
        // Y_ft = -y_s * factor ;  Y_tf = -y_s * conj(factor)
        // Y_ff = y_s / a^2 + j b/2 ; Y_tt = y_s + j b/2
        const double fr_re =  cphi * inv_a;
        const double fr_im =  sphi * inv_a;

        // Y_ft = -(g_ser + j b_ser)*(fr_re + j fr_im)
        double g_ft = -(g_ser * fr_re - b_ser * fr_im);
        double b_ft = -(g_ser * fr_im + b_ser * fr_re);
        // Y_tf = -(g_ser + j b_ser)*(fr_re - j fr_im)
        double g_tf = -(g_ser * fr_re + b_ser * fr_im);
        double b_tf = -(b_ser * fr_re - g_ser * fr_im);

        // Y_ff diagonal
        const double g_ff = g_ser * inv_a2;
        const double b_ff = b_ser * inv_a2 + b_half;
        // Y_tt diagonal
        const double g_tt = g_ser;
        const double b_tt = b_ser + b_half;

        Yg.add(f, f, g_ff); Yb.add(f, f, b_ff);
        Yg.add(t, t, g_tt); Yb.add(t, t, b_tt);
        Yg.add(f, t, g_ft); Yb.add(f, t, b_ft);
        Yg.add(t, f, g_tf); Yb.add(t, f, b_tf);
    }

    // Bus shunts into the diagonal.
    for (int i = 0; i < n; ++i) {
        if (buses[i].g_shunt != 0.0 || buses[i].b_shunt != 0.0) {
            Yg.add(i, i, buses[i].g_shunt);
            Yb.add(i, i, buses[i].b_shunt);
        }
    }

    Yg.buildCSR();
    Yb.buildCSR();
}

int PowerSystem::countByType(BusType t) const {
    int c = 0;
    for (const auto& b : buses) if (b.type == t) ++c;
    return c;
}

} // namespace powerflow
