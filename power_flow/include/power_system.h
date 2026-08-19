#ifndef POWER_FLOW_POWER_SYSTEM_H
#define POWER_FLOW_POWER_SYSTEM_H

#include <string>
#include <vector>
#include "sparse_matrix.h"

namespace powerflow {

enum class BusType {
    PQ     = 0,
    PV     = 1,
    SLACK  = 3
};

// Network bus (per-unit quantities).
struct Bus {
    int      id      = 0;     // external bus id
    BusType  type    = BusType::PQ;
    int      index   = 0;    // internal 0-based index
    double   v       = 1.0;  // voltage magnitude (pu)
    double   theta   = 0.0;  // voltage angle (rad)
    double   p_gen   = 0.0;  // scheduled generation (MW)
    double   q_gen   = 0.0;  // scheduled generation (MVAr)
    double   p_load  = 0.0;  // load (MW)
    double   q_load  = 0.0;  // load (MVAr)
    double   g_shunt = 0.0;  // shunt conductance (pu)
    double   b_shunt = 0.0;  // shunt susceptance (pu)
    double   v_spec  = 1.0;  // specified voltage for PV / slack (pu)
    double   v_min   = 0.95;
    double   v_max   = 1.05;
};

// Branch modeled as a pi-equivalent with optional off-nominal / phase-shifting
// tap on the "from" side.
struct Branch {
    int    from = 0;  // internal bus index
    int    to   = 0;
    double r    = 0.0;   // series resistance (pu)
    double x    = 0.0;   // series reactance (pu)
    double b    = 0.0;   // total line charging susceptance (pu)
    double tap  = 1.0;   // tap ratio magnitude
    double phi  = 0.0;   // phase shift angle (rad)
};

// Container for the network and admittance-matrix assembly.
class PowerSystem {
public:
    std::vector<Bus>     buses;
    std::vector<Branch>  branches;
    double baseMVA = 100.0;

    size_t nBus() const { return buses.size(); }

    // Build the bus admittance matrix Y = G + jB into two SparseMatrices that
    // share an identical sparsity pattern (so G.values[k] and B.values[k] refer
    // to the same (row,col) entry).
    void buildAdmittance(SparseMatrix& Yg, SparseMatrix& Yb);

    // Quick statistics / sanity helpers.
    int countByType(BusType t) const;
    int slackCount() const { return countByType(BusType::SLACK); }
};

} // namespace powerflow

#endif // POWER_FLOW_POWER_SYSTEM_H
