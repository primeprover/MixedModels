#pragma once

#include <memory>
#include "response_base.hpp"
#include "homo_normal_response.hpp"
#include "hetero_normal_response.hpp"

inline std::unique_ptr<ResponseModel> make_response(const Rcpp::List& prepared)
{
    const bool heteroscedastic = prepared["heteroscedastic"];
    if (heteroscedastic) {
        return std::make_unique<HeteroNormalResponse>(prepared);
    }
    return std::make_unique<HomoNormalResponse>(prepared);
}
