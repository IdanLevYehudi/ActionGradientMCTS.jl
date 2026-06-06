struct ValueUpdaterSNMISMC <: AbstractValueUpdater
    ## Values stores for each posterior node
    value::Vector{Float64}  # The value at the posterior node b
    prev_value::Vector{Float64}  # The previous values at the posterior node b - the recursive value update for the weighted mean estimator requires the previous value.
    prev_total_n::Vector{Int}  # The previous visitation counter at the posterior node b. This is used for updating the value estimate.
    p_log_likelihoods::Vector{Float64}  # The transition log likelihood for the posterior node given its parent and current action label
    q_log_likelihoods::Vector{Float64}  # The proposal log likelihood for the posterior node given its parent and proposal action label
    w_sampling::Vector{Float64}  # The relative weight (due to differing actions) for each posterior node. Used for sampling nodes at random w.r.t. action weights.

    ## Values stored for each action node
    r_est::Vector{Float64}  # If V(b)=E_{bp}[r(b,a,bp)+γ⋅Vp(bp)], this estimates the expected r (immediate reward) at the (b,a) action node. r_est is maintained as a simple running mean for each (b,a).
    future_v_est::Vector{Float64}  # If V(b)=E_{bp}[r(b,a,bp)+γ⋅Vp(bp)], this estimates the expected Vp (future value) at the (b,a) action node. future_v_est is maintained as a weighted mean of values for the node (b,a).
    sn_log_denom::Vector{Float64}  # The self-normalized log of denominator for each action node. Eta in the equations
    sn_log_w_norm::Vector{Float64}  # The denominator that was used for normalizing w_sampling. Lambda in the equations.

    function ValueUpdaterSNMISMC(sz::Int=1000)
        sz = min(sz, 100_000)
        new(sizehint!(Float64[], sz),  # value
            sizehint!(Float64[], sz),  # prev_value
            sizehint!(Int[], sz),  # prev_total_n
            sizehint!(Float64[], sz),  # p_log_likelihoods
            sizehint!(Float64[], sz),  # q_log_likelihoods
            sizehint!(Float64[], sz),  # w_sampling
            sizehint!(Float64[], sz),  # r_est
            sizehint!(Float64[], sz),  # future_v_est
            sizehint!(Float64[], sz),  # sn_log_denom
            sizehint!(Float64[], sz)  # sn_log_w_norm
        )
    end
end

function insert_state_node!(up::ValueUpdaterSNMISMC, snode::Int, p_log_likelihood::Float64=0.0, q_log_likelihood::Float64=0.0)
    push!(up.value, 0.0)
    push!(up.prev_value, 0.0)
    push!(up.prev_total_n, -1)
    push!(up.p_log_likelihoods, p_log_likelihood)
    push!(up.q_log_likelihoods, q_log_likelihood)
    push!(up.w_sampling, 0.0)
end

function insert_action_node!(up::ValueUpdaterSNMISMC, snode::Int, sanode::Int)
    push!(up.r_est, 0.0)
    push!(up.future_v_est, 0.0)
    push!(up.sn_log_denom, -Inf)
    push!(up.sn_log_w_norm, -Inf)
end

value(up::ValueUpdaterSNMISMC, index::Int) = up.value[index]

function logsumexp_factor(log_vec::T, scale_vec::U) where {T<:AbstractVector{Float64},U<:AbstractVector}
    # Computes the safe logsumexp of log(sum(scale_vec * exp(log_vec))). Supports negative scaling values, and returns the abs of the logsumexp and the sign of the original result.
    max_log = maximum(log_vec)
    sum_exp = zero(first(scale_vec))
    for i in 1:length(log_vec)
        sum_exp += scale_vec[i] .* exp(log_vec[i] - max_log)
    end
    sgn = sign.(sum_exp)
    return log.(abs.(sum_exp)) .+ max_log, sgn
end

function logsumexp_and_softmax(log_vec::T) where {T<:AbstractVector{Float64}}
    # Returns the log of the sum of exponentials of the input vector, and the softmax of the input vector
    # Returned as a tuple (log_sum_exp, softmax)
    max_log = maximum(log_vec)
    sum_exp = 0.0
    softmax = zero(log_vec)
    for i in 1:length(log_vec)
        softmax[i] = exp(log_vec[i] - max_log)
        sum_exp += softmax[i]
    end
    log_sum_exp = log(sum_exp) + max_log
    softmax /= sum_exp
    return log_sum_exp, softmax
end
