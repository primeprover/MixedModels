#pragma once

#include "model_structs.hpp"

inline bool random_effects_compute(
    const GroupData& group,
    const MatrixXd& G_inv,
    GroupWorkspace& ws)
{
    ws.ZtRinvZ.noalias() = group.Z.transpose() * ws.RinvZ;
    ws.ZtRinvX.noalias() = group.Z.transpose() * ws.RinvX;
    ws.ZtRinvy.noalias() = group.Z.transpose() * ws.Rinvy;

    ws.K = G_inv;
    ws.K.noalias() += ws.ZtRinvZ;

    ws.llt.compute(ws.K);
    if (ws.llt.info() != Eigen::Success)
        return false;

    ws.logdetK = 2.0 * ws.llt.matrixLLT().diagonal().array().log().sum();
    ws.KinvZtRinvX = ws.llt.solve(ws.ZtRinvX);
    ws.KinvZtRinvy = ws.llt.solve(ws.ZtRinvy);

    ws.VinvX.noalias() = ws.RinvX - ws.RinvZ * ws.KinvZtRinvX;
    ws.Vinvy.noalias() = ws.Rinvy - ws.RinvZ * ws.KinvZtRinvy;

    return true;
}
