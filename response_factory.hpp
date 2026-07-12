#pragma once

#include <memory>
#include <string>
#include "response_base.hpp"
#include "homo_normal_response.hpp"
#include "hetero_normal_response.hpp"

inline std::string response_type_from_prepared(const Rcpp::List& prepared)
{
    if (prepared.containsElementNamed("response_spec")) {
        const Rcpp::List response_spec = prepared["response_spec"];
        if (response_spec.containsElementNamed("type")) {
            const std::string type = Rcpp::as<std::string>(response_spec["type"]);
            if (type == "hetero_normal") return type;
            if (type == "homo_normal") return type;
        }
    }

    const bool heteroscedastic = prepared.containsElementNamed("heteroscedastic")
        ? Rcpp::as<bool>(prepared["heteroscedastic"])
        : false;
    return heteroscedastic ? "hetero_normal" : "homo_normal";
}

inline std::unique_ptr<ResponseModel> make_response(const Rcpp::List& prepared)
{
    const std::string response_type = response_type_from_prepared(prepared);
    if (response_type == "hetero_normal") {
        return std::make_unique<HeteroNormalResponse>(prepared);
    }
    return std::make_unique<HomoNormalResponse>(prepared);
}
