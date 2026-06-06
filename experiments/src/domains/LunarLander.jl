# Taken from:
# https://gist.githubusercontent.com/zsunberg/b42a7665f61dadc1dcd0cdb3a216e17e/raw/c04dde26a862aff0b844d57a4ddeefa188a1e6a6/lunar_lander.jl

module LunarLander


using Parameters
using Random
using Distributions
using Compose
import Cairo
import Plots
import Base: rand
using StaticArrays
using Statistics
using LinearAlgebra
using PDMats
using Enzyme

using POMDPs
using POMDPTools
import POMDPTools.ModelTools: UnderlyingMDP
using POMDPGifs
using MCTS
using ParticleFilters
import ParticleFilters: obs_weight, ParticleCollection

const Vec6 = SVector{6,Float64}
const Vec3 = SVector{3,Float64}

export LunarLanderMDP,
    LunarLanderPOMDP,
    LunarLanderProblem,
    LanderActionSpace,
    LanderPolicy,
    Vec6,
    Vec3,
    transition_mean_state

struct LunarLanderMDP <: MDP{Vec6,Vec3}
    # Transition parameters
    dt::Float64
    m::Float64
    I::Float64
    Q::Vec6
    jacobian_logdet::Float64
    ϵ_dist::AbstractMvNormal
    # Action parameters
    min_lateral::Float64
    max_lateral::Float64
    max_thrust::Float64
    max_offset::Float64
    # Reward parameters
    discount::Float64
    max_horizontal_offset::Float64
    max_angle::Float64
    landed_height::Float64
    step_penalty::Float64
    success_reward::Float64
    failure_penalty::Float64
    speed_step_penalty::Float64
    offset_step_penalty::Float64
    angle_step_penalty::Float64
end

function LunarLanderMDP(;
    # Transition parameters
    dt::Float64=0.1,
    m::Float64=1.0,
    I::Float64=10.0,
    Q::Vec6=Vec6([0.0, 0.0, 0.0, 0.1, 0.1, 0.01]),
    # Action parameters
    min_lateral::Float64=-5.0,
    max_lateral::Float64=5.0,
    max_thrust::Float64=15.0,
    max_offset::Float64=1.0,
    # Reward parameters
    discount::Float64=0.99,
    max_horizontal_offset::Float64=15.0,
    max_angle::Float64=0.5,
    landed_height::Float64=1.0,
    step_penalty::Float64=-1.0,
    success_reward::Float64=100.0,
    failure_penalty::Float64=-1000.0,
    speed_step_penalty::Float64=0.0,
    offset_step_penalty::Float64=0.0,
    angle_step_penalty::Float64=0.0
)
    return LunarLanderMDP(dt,
        m,
        I,
        Q,
        _j_logdet(Q),
        _noise_dist(Q),
        min_lateral,
        max_lateral,
        max_thrust,
        max_offset,
        discount,
        max_horizontal_offset,
        max_angle,
        landed_height,
        step_penalty,
        success_reward,
        failure_penalty,
        speed_step_penalty,
        offset_step_penalty,
        angle_step_penalty)
end

struct LunarLanderPOMDP <: POMDP{Vec6,Vec3,Vec3}
    mdp::LunarLanderMDP
    R::Vec3
    obs_dist::AbstractMvNormal
end

function LunarLanderPOMDP(;
    mdp::LunarLanderMDP=LunarLanderMDP(),
    R::Vec3=Vec3([1.0, 0.01, 0.1])
)
    obs_dist = MvNormal(Vec3(zeros(3)), diagm(R))
    return LunarLanderPOMDP(mdp, R, obs_dist)
end

const LunarLanderProblem = Union{LunarLanderMDP,LunarLanderPOMDP}
UnderlyingMDP(p::LunarLanderMDP) = p
UnderlyingMDP(p::LunarLanderPOMDP) = p.mdp


"""
This represents the deterministic transition function given the noise parameter ϵ:
s' = f(s, a, ϵ).
"""
function transition_f(m::LunarLanderMDP, s::Vec6, a::Vec3, ϵ::Vec3)
    x = s[1]
    z = s[2]
    θ = s[3]
    vx = s[4]
    vz = s[5]
    ω = s[6]

    f_lateral = a[1]
    thrust = a[2]
    δ = a[3]

    fx = cos(θ) * f_lateral - sin(θ) * thrust
    fz = cos(θ) * thrust + sin(θ) * f_lateral
    torque = -δ * f_lateral

    ax = fx / m.m
    az = fz / m.m
    ωdot = torque / m.I

    vxp = vx + ax * m.dt + ϵ[1] * m.Q[4]
    vzp = vz + (az - 9.0) * m.dt + ϵ[2] * m.Q[5]
    ωp = ω + ωdot * m.dt + ϵ[3] * m.Q[6]

    xp = x + vx * m.dt
    zp = z + vz * m.dt
    θp = θ + ω * m.dt

    sp = Vec6(xp, zp, θp, vxp, vzp, ωp)
    return sp
end

function POMDPs.transition(pp::LunarLanderProblem, s::Vec6, a::Vec3)
    ImplicitDistribution(pp, s, a) do pp, s, a, rng
        p = UnderlyingMDP(pp)
        ϵ = Vec3(randn(rng, 3))
        sp = transition_f(p, s, a, ϵ)
        return sp
    end
end

function transition_mean_state(pp::LunarLanderProblem, s::Vec6, a::Vec3)
    # This is not the real mean state, rather just a representative state assuming ML noise parameter.
    return transition_f(UnderlyingMDP(pp), s, a, Vec3(zeros(3)))
end

"""
Find ϵ that solves s' = f(s, a, ϵ) for given s, a, s', assuming that the solution exists and that it is unique.
"""
function inverse_transition_f(m::LunarLanderMDP, s::Vec6, a::Vec3, sp::Vec6)
    θ = s[3]
    vx = s[4]
    vz = s[5]
    ω = s[6]

    f_lateral = a[1]
    thrust = a[2]
    δ = a[3]

    fx = cos(θ) * f_lateral - sin(θ) * thrust
    fz = cos(θ) * thrust + sin(θ) * f_lateral
    torque = -δ * f_lateral

    ax = fx / m.m
    az = fz / m.m
    ωdot = torque / m.I

    vxp = sp[4]
    vzp = sp[5]
    ωp = sp[6]

    ϵ1 = (vxp - vx - ax * m.dt) / m.Q[4]
    ϵ2 = (vzp - vz - (az - 9.0) * m.dt) / m.Q[5]
    ϵ3 = (ωp - ω - ωdot * m.dt) / m.Q[6]

    return Vec3(ϵ1, ϵ2, ϵ3)
end

function Da_inverse_transition_f(m::LunarLanderMDP, s::Vec6, a::Vec3, sp::Vec6)
    θ = s[3]
    f_lateral = a[1]
    δ = a[3]

    ∂ϵ_∂a_mat = SMatrix{3,3,Float64}(
        [cos(θ)*(-m.dt) / (m.m * m.Q[4]) -sin(θ)*(-m.dt) / (m.m * m.Q[4]) 0.0;
            sin(θ)*(-m.dt) / (m.m * m.Q[5]) cos(θ)*(-m.dt) / (m.m * m.Q[5]) 0.0;
            -δ*(-m.dt) / (m.I * m.Q[6]) 0.0 -f_lateral*(-m.dt) / (m.I * m.Q[6])])

    return ∂ϵ_∂a_mat
end

_noise_dist(Q::Vec6) = MvNormal(Vec3(zeros(3)), ScalMat(3, 1.0))

"""
Returns the distribution of the noise parameter p_w(⋅ | s, a), given the current state and action.
"""
noise_dist(p::LunarLanderMDP, s::Vec6, a::Vec3) = p.ϵ_dist

function _j_logdet(Q::Vec6)
    return reduce(+, map((x) -> 2 * log(x), Q[4:6]))
end

j_logdet(p::LunarLanderProblem, s::Vec6, a::Vec3, ϵ::Vec3) = UnderlyingMDP(p).jacobian_logdet

function transition_logpdf(pp::LunarLanderProblem, s::Vec6, a::Vec3, sp::Vec6)
    p = UnderlyingMDP(pp)
    ϵ = inverse_transition_f(p, s, a, sp)
    # The following calculation is a result of the Area Formula Theorem - Need to factor by inverse of generalized Jacobian.
    d_noise = noise_dist(p, s, a)
    return logpdf(d_noise, ϵ) - 0.5 * j_logdet(p, s, a, ϵ)
end

"""
Gradient of transition_logdf w.r.t. a.
"""
function transition_gradlogpdf(pp::LunarLanderProblem, s::Vec6, a::Vec3, sp::Vec6)
    p = UnderlyingMDP(pp)
    ϵ = inverse_transition_f(p, s, a, sp)
    # The following calculation is a result of the Area Formula Theorem - Need to factor by inverse of generalized Jacobian.
    # Also - using the fact that
    # grad(f(g(x))) = (Df(g(x)))' * grad(g(x))
    d_noise = noise_dist(p, s, a)
    # The Jacobian of the transition function with respect to ϵ.
    # The enzyme jacobian turned out faster than the one written by hand!!
    # And this was considerably faster than autodiff'ing the transition_logpdf function.
    inverse_for_a_func = (a) -> inverse_transition_f(p, s, a, sp)
    j_mat = first(jacobian(Forward, inverse_for_a_func, a))
    grad_vec = gradlogpdf(d_noise, ϵ)
    return j_mat' * grad_vec
end

function POMDPs.reward(pp::LunarLanderProblem, s::Vec6, a::Vec3, sp::Vec6)
    p = UnderlyingMDP(pp)
    δ = abs(sp[1])
    z = sp[2]
    θ = abs(sp[3])
    vz = abs(sp[5])

    if δ >= p.max_horizontal_offset || θ >= p.max_angle
        return p.failure_penalty
    elseif z <= p.landed_height
        return -(δ + vz^2) + p.success_reward
    else
        return p.step_penalty + p.speed_step_penalty * abs(a[1]) + p.offset_step_penalty * abs(a[2]) + p.angle_step_penalty * abs(a[3])
    end
end

struct LunarLanderObsDist
    pp::LunarLanderPOMDP
    mean::Vec3
end

function rand(rng::AbstractRNG, d::LunarLanderObsDist)
    x_out = zero(MVector{3,Float64})
    rand!(rng, d.pp.obs_dist, x_out)
    return d.mean + Vec3(x_out)
end

function POMDPs.pdf(d::LunarLanderObsDist, o::Vec3)
    return pdf(d.pp.obs_dist, o - d.mean)
end

function Distributions.mean(d::LunarLanderObsDist)
    return d.mean
end

function POMDPs.observation(p::LunarLanderPOMDP, sp::Vec6)
    z = sp[2]
    θ = sp[3]
    xdot = sp[4]
    ω = sp[6]
    agl = z / cos(θ)  # This is the rangefinder sensor's measurement, depending on the height and the vehicle orientation
    return LunarLanderObsDist(p, Vec3([agl, ω, xdot]))
end

POMDPs.discount(pp::LunarLanderProblem) = UnderlyingMDP(pp).discount

function POMDPs.isterminal(pp::LunarLanderProblem, s::Vec6)
    p = UnderlyingMDP(pp)
    δ = abs(s[1])
    z = s[2]
    θ = abs(s[3])
    if δ >= p.max_horizontal_offset || θ >= p.max_angle || z <= p.landed_height
        return true
    else
        return false
    end
end

struct LanderInitState
    dist::MvNormal
end

function POMDPs.initialstate(pp::LunarLanderProblem)
    μ = Vec6([0.0, 50.0, 0.0, 0.0, -10.0, 0.0])
    σ = Vec6([0.1, 0.1, 0.01, 0.1, 0.1, 0.01])
    σ = diagm(σ)
    return LanderInitState(MvNormal(μ, σ))
end

function Base.rand(rng::AbstractRNG, s::LanderInitState)
    return Vec6(rand(rng, s.dist))
end

## Action Sampling

function _project_action(p::LunarLanderMDP, a::Vec3)
    # Project the action to the action space
    f_x = clamp(a[1], p.min_lateral, p.max_lateral)
    f_z = clamp(a[2], 0.0, p.max_thrust)
    offset = clamp(a[3], -p.max_offset, p.max_offset)
    return Vec3(f_x, f_z, offset)
end

struct LanderActionSpace
    pp::LunarLanderProblem
end

function Base.rand(rng::AbstractRNG, as::LanderActionSpace)
    p = UnderlyingMDP(as.pp)
    lateral_range = p.max_lateral - p.min_lateral
    f_x = rand(rng) * lateral_range + p.min_lateral
    f_z = rand(rng) * p.max_thrust
    offset = (rand() - 0.5) * 2.0 * p.max_offset
    return Vec3(f_x, f_z, offset)
end

POMDPs.actions(pp::LunarLanderProblem) = LanderActionSpace(pp)

## Heuristics

struct LanderPolicy <: Policy
    m::LunarLanderProblem
end

POMDPs.updater(p::LanderPolicy) = EKFUpdater(UnderlyingMDP(p).m, UnderlyingMDP(p).m.Q .^ 2, UnderlyingMDP(p).m.R .^ 2)

function POMDPs.action(p::LanderPolicy, s::Vec6)
    return Vec3(-0.1 * s[4], -0.1 * s[5], 0.0)
end

POMDPs.action(p::LanderPolicy, b::Union{AbstractMvNormal, AbstractParticleBelief{Vec6}}) = action(p, mean(b))

# For EKF Belief Updater

function gen_A(pp::LunarLanderProblem, s::Vec6, a::Vec3)
    m = UnderlyingMDP(pp)
    θ = s[3]
    f_l = a[1]
    thrust = a[2]
    A = zeros(Float64, 6, 6)
    A[1, 1] = 1.0
    A[1, 4] = m.dt
    A[2, 2] = 1.0
    A[2, 5] = m.dt
    A[3, 3] = 1.0
    A[3, 6] = m.dt

    A[4, 3] = (-sin(θ) * f_l - cos(θ) * thrust) * m.dt / m.m
    A[4, 4] = 1.0
    A[5, 3] = (-sin(θ) * thrust + cos(θ) * f_l) * m.dt / m.m
    A[5, 5] = 1.0
    A[6, 6] = 1.0
    return SMatrix{6,6,Float64}(A)
end

function gen_C(m::LunarLanderPOMDP, s::Vec6)
    z = s[2]
    θ = s[3]
    C = zeros(Float64, 3, 6)
    C[1, 2] = 1 / (cos(θ) + eps())
    C[1, 3] = z * sin(θ) / (cos(θ)^2 + eps())
    C[2, 6] = 1.0
    C[3, 4] = 1.0
    return SMatrix{3,6,Float64}(C)
end

end  # module
