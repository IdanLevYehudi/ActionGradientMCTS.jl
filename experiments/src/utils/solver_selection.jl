const SOLVER_CHOICES = ("dpw", "ag-dpw", "vpw", "ag-vpw", "pomcpow", "vomcpow", "all")
const DEFAULT_SOLVERS = ["ag-dpw"]
const MDP_SOLVERS = ["dpw", "ag-dpw", "vpw", "ag-vpw"]
const POMDP_SOLVERS = ["dpw", "ag-dpw", "vpw", "ag-vpw", "pomcpow", "vomcpow"]

solver_uses_voo(solver::AbstractString) = solver in ("vpw", "ag-vpw", "vomcpow")
solver_is_dpw(solver::AbstractString) = solver in ("dpw", "vpw")
solver_is_agmcts(solver::AbstractString) = solver in ("ag-dpw", "ag-vpw")
solver_is_pomcpow(solver::AbstractString) = solver in ("pomcpow", "vomcpow")

function normalize_solver_selection(solvers; mdp::Bool)
    selected = solvers isa AbstractString ? [String(solvers)] : String.(collect(solvers))
    isempty(selected) && return copy(DEFAULT_SOLVERS)

    if "all" in selected
        length(selected) == 1 || throw(ArgumentError("--solver all cannot be combined with other solver names."))
        return mdp ? copy(MDP_SOLVERS) : copy(POMDP_SOLVERS)
    end

    if mdp
        invalid = [solver for solver in selected if solver_is_pomcpow(solver)]
        isempty(invalid) || throw(ArgumentError("$(join(invalid, ", ")) are only valid for POMDP domains; omit --mdp or choose dpw, ag-dpw, vpw, ag-vpw, or all."))
    end

    normalized = String[]
    for solver in selected
        solver in normalized || push!(normalized, solver)
    end
    return normalized
end

function canonical_solver_name(solver::AbstractString; mdp::Bool)
    if solver == "dpw"
        return mdp ? "DPW" : "PFT-DPW"
    elseif solver == "ag-dpw"
        return mdp ? "AG-DPW" : "AG-PFT-DPW"
    elseif solver == "vpw"
        return mdp ? "VPW" : "PFT-VPW"
    elseif solver == "ag-vpw"
        return mdp ? "AG-VPW" : "AG-PFT-VPW"
    elseif solver == "pomcpow"
        mdp && throw(ArgumentError("pomcpow is only valid for POMDP domains."))
        return "POMCPOW"
    elseif solver == "vomcpow"
        mdp && throw(ArgumentError("vomcpow is only valid for POMDP domains."))
        return "VOMCPOW"
    end
    throw(ArgumentError("unknown solver '$solver'. Expected one of: $(join(SOLVER_CHOICES[1:end-1], ", "))"))
end
