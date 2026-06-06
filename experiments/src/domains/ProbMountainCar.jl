module ProbMountainCar


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
using SpecialFunctions

using OrdinaryDiffEq
using Enzyme
using SciMLSensitivity
using ForwardDiff

using POMDPs
using POMDPTools
import POMDPTools.ModelTools: UnderlyingMDP
using POMDPGifs
using MCTS
using ParticleFilters
import ParticleFilters: obs_weight, ParticleCollection

##################################
### Common structs and helpers ###
##################################

function clipped_gaussian_logprob(w::T, a::T; σ::Float64, a_min::Float64=-1.0, a_max::Float64=1.0) where {T<:Real}
    L = a_min - a
    R = a_max - a
    N = Normal(0, σ)
    if w ≤ L                      # left atom
        return logcdf(N, L)
    elseif w ≥ R                  # right atom
        return logccdf(N, R)
    else                          # interior density
        return logpdf(N, w)
    end
end

const Vec2 = SVector{2,Float64}
const CarState = Vec2

#########################################
### Simplified transition MountainCar ###
#########################################

function mountain(x::T) where {T<:AbstractFloat}
    0.45 * sin(3.0 * x) + 0.5
end

struct SimpleMountainCarMDP <: MDP{CarState,Float64}
    # General environment parameters
    x_min::Float64
    x_max::Float64
    v_min::Float64
    v_max::Float64
    action_min::Float64
    action_max::Float64
    action_std::Float64
    # Simplified transition parameters
    mountain_coeff::Float64
    a_to_v_coeff::Float64
    x_to_v_coeff::Float64
    # Reward parameters
    reward_goal::Float64
    velocity_goal_penalty::Float64
    distance_from_goal_penalty::Float64
    reward_step::Float64
    failure_penalty::Float64
    discount::Float64
    action_penalty_coeff::Float64
end

function SimpleMountainCarMDP(;
    x_min::Float64=-1.5,
    x_max::Float64=0.5,
    v_min::Float64=-0.05,
    v_max::Float64=0.05,
    action_min::Float64=-1.0,
    action_max::Float64=1.0,
    action_std::Float64=0.1,
    # Simplified transition parameters
    mountain_coeff::Float64=3.0,
    a_to_v_coeff::Float64=0.001,
    x_to_v_coeff::Float64=-0.0025,
    # Reward parameters
    reward_goal::Float64=100.0,
    velocity_goal_penalty::Float64=-200.0,  # Max speed at goal incurs -10 penalty
    distance_from_goal_penalty::Float64=-1.0,
    reward_step::Float64=-0.1,
    failure_penalty::Float64=-100.0,
    discount::Float64=0.99,
    action_penalty_coeff::Float64=-2.0
)
    return SimpleMountainCarMDP(
        x_min,
        x_max,
        v_min,
        v_max,
        action_min,
        action_max,
        action_std,
        mountain_coeff,
        a_to_v_coeff,
        x_to_v_coeff,
        reward_goal,
        velocity_goal_penalty,
        distance_from_goal_penalty,
        reward_step,
        failure_penalty,
        discount,
        action_penalty_coeff
    )
end

"""
This represents the deterministic transition function given the noise parameter w:
s' = f(s, a, w).
"""
function transition_f(p::SimpleMountainCarMDP, s::CarState, a::T, w::T) where {T<:AbstractFloat}
    a_noise = project_action(p, a + w)
    vp = s[2] + a_noise * p.a_to_v_coeff + cos(p.mountain_coeff * s[1]) * p.x_to_v_coeff
    xp = s[1] + vp
    return CarState(xp, vp)
end

function Dw_transition_f(p::SimpleMountainCarMDP, s::CarState, a::T, w::T) where {T<:AbstractFloat}
    return SMatrix{2,1,Float64}(p.a_to_v_coeff, p.a_to_v_coeff)
end

function inverse_transition_f(p::SimpleMountainCarMDP, s::CarState, a::T, sp::CarState) where {T<:AbstractFloat}
    a_noise = (sp[2] - s[2] - cos(p.mountain_coeff * s[1]) * p.x_to_v_coeff) / p.a_to_v_coeff
    return a_noise - a
end

function transition_logpdf(p::SimpleMountainCarMDP, s::CarState, a::T, sp::CarState) where {T<:AbstractFloat}
    w = inverse_transition_f(p, s, a, sp)
    in_bounds, a, w = check_w_bounds(p, a, w)
    if !in_bounds
        return -Inf
    end
    noise_logpdf = clipped_gaussian_logprob(w, a; σ=p.action_std, a_min=p.action_min, a_max=p.action_max)
    jacobian_matrix = Dw_transition_f(p, s, a, w)
    j_logdet = -0.5 * logabsdet(jacobian_matrix' * jacobian_matrix)[1]
    return noise_logpdf + j_logdet
end

function transition_gradlogpdf(p::SimpleMountainCarMDP, s::CarState, a::T, sp::CarState) where {T<:AbstractFloat}
    ret = Enzyme.autodiff(set_runtime_activity(Forward), Const(transition_logpdf), Const(p), Const(s), Duplicated(a, one(a)), Const(sp))
    return first(ret)
end

##################################
### ODE transition MountainCar ###
##################################

function hill(x::Real)
    if x < 0.0
        return x^2 + x
    else
        return x / sqrt(1.0 + 5.0 * x^2)
    end
end

function hill_prime(x::Real)
    if x < 0.0
        return 2.0 * x + 1.0
    else
        return 1.0 / (5.0 * x^2 + 1.0)^(3.0 / 2.0)
    end
end

function hill_double_prime(x::Real)
    if x < 0.0
        return 2.0
    else
        return -15.0 * x / (1.0 + 5.0 * x^2)^(5 / 2)
    end
end

"""
By DifferentialEquations.jl convention - u is the current state, p is the parameter vector, t is the current time. du/dt is returned.
"""
function car_dynamics!(du, u, p, t)
    a, m, g = p  # action, car mass, gravity
    x_car, v_car = u
    dp_car = v_car
    dv_car = (a / m - g * hill_prime(x_car) - v_car^2 * hill_prime(x_car) * hill_double_prime(x_car)) / (1 + hill_prime(x_car)^2)
    du[1] = dp_car
    du[2] = dv_car
    return nothing
    # return SA[dp_car, dv_car]
end

struct ODEMountainCarMDP <: MDP{CarState,Float64}
    # General environment parameters
    x_min::Float64
    x_max::Float64
    v_min::Float64
    v_max::Float64
    action_min::Float64
    action_max::Float64
    action_std::Float64
    # ODE transition parameters
    hill_car_mass::Float64
    hill_gravity::Float64
    hill_integration_dt::Float64
    hill_time_step::Float64
    # Reward parameters
    reward_goal::Float64
    velocity_goal_penalty::Float64
    distance_from_goal_penalty::Float64
    reward_step::Float64
    failure_penalty::Float64
    discount::Float64
    action_penalty_coeff::Float64
    transitions_logger::Dict{Tuple{CarState,CarState},Float64}
    ode_prob::ODEProblem
end

"""
A logging function to keep track of a_noise that generated the transitions (s,a_noise,sp).
"""
function logger_transition_fn(p::ODEMountainCarMDP, s::CarState, a::T, w::T, sp::CarState) where {T<:AbstractFloat}
    push!(p.transitions_logger, (s, sp) => project_action(p, a + w))
end

function ODEMountainCarMDP(;
    x_min::Float64=-1.0,
    x_max::Float64=1.0,
    v_min::Float64=-3.0,
    v_max::Float64=3.0,
    action_min::Float64=-4.0,
    action_max::Float64=4.0,
    action_std::Float64=0.4,
    # ODE transition - hill problem
    hill_car_mass::Float64=1.0,
    hill_gravity::Float64=9.81,
    hill_integration_dt::Float64=0.001,  # time interval for numerical integration
    hill_time_step::Float64=0.1,  # time interval between consecutive observations and actions
    # Reward parameters
    reward_goal::Float64=100.0,
    velocity_goal_penalty::Float64=-10.0 / 3.0,
    distance_from_goal_penalty::Float64=-1.0,
    reward_step::Float64=-0.1,
    failure_penalty::Float64=-100.0,
    discount::Float64=0.99,
    action_penalty_coeff::Float64=-0.25
)
    ode_prob = ODEProblem(car_dynamics!, [0.0, 0.0], (0.0, hill_time_step), [0.0, hill_car_mass, hill_gravity], save_everystep=false, save_start=false)

    return ODEMountainCarMDP(
        x_min,
        x_max,
        v_min,
        v_max,
        action_min,
        action_max,
        action_std,
        hill_car_mass,
        hill_gravity,
        hill_integration_dt,
        hill_time_step,
        reward_goal,
        velocity_goal_penalty,
        distance_from_goal_penalty,
        reward_step,
        failure_penalty,
        discount,
        action_penalty_coeff,
        Dict{Tuple{CarState,CarState},Float64}(),
        ode_prob
    )
end

"""
This represents the deterministic transition function given the noise parameter w:
s' = f(s, a, w).
"""
function transition_f(p::ODEMountainCarMDP, s::CarState, a::T, w::T) where {T<:AbstractFloat}
    a_noise = a + w
    prob = remake(p.ode_prob, u0=[s[1], s[2]], tspan=(0.0, p.hill_time_step), p=[a_noise, p.hill_car_mass, p.hill_gravity])
    ret = SciMLSensitivity.solve(prob, Tsit5(), dt=p.hill_integration_dt)
    sp = CarState(ret.u[end])
    logger_transition_fn(p, s, a, w, sp)
    return sp
end

function Dw_transition_f(p::ODEMountainCarMDP, s::CarState, a::T, w::T) where {T<:Real}
    function f_for_diff(w_val)
        a_noise = a + w_val
        prob = remake(p.ode_prob, u0=[s[1], s[2]], tspan=(0.0, p.hill_time_step), p=[a_noise, p.hill_car_mass, p.hill_gravity])
        sol = SciMLSensitivity.solve(prob, Tsit5(), dt=p.hill_integration_dt)
        return [sol.u[end][1], sol.u[end][2]]
    end
    jac = ForwardDiff.derivative(f_for_diff, w)
    return jac
end

function inverse_transition_f(p::ODEMountainCarMDP, s::CarState, a::T, sp::CarState) where {T<:AbstractFloat}
    a_noise = p.transitions_logger[(s, sp)]
    return a_noise - a
end

function transition_logpdf(p::ODEMountainCarMDP, s::CarState, a::T, sp::CarState) where {T<:AbstractFloat}
    w = inverse_transition_f(p, s, a, sp)
    in_bounds, a, w = check_w_bounds(p, a, w)
    if !in_bounds
        return -Inf
    end
    # The following calculation is a result of the Area Formula Theorem - Need to factor by inverse of generalized Jacobian.
    noise_logpdf = clipped_gaussian_logprob(w, a; σ=p.action_std, a_min=p.action_min, a_max=p.action_max)
    jacobian_matrix = Dw_transition_f(p, s, a, w)
    j_logdet = -0.5 * logabsdet(jacobian_matrix' * jacobian_matrix)[1]
    return noise_logpdf + j_logdet
end

function transition_gradlogpdf(p::ODEMountainCarMDP, s::CarState, a::T, sp::CarState) where {T<:AbstractFloat}
    w = inverse_transition_f(p, s, a, sp)
    in_bounds, a, w = check_w_bounds(p, a, w)
    if !in_bounds
        return 0.0
    end
    function f3_for_diff(vals)
        # The following calculation is a result of the Area Formula Theorem - Need to factor by inverse of generalized Jacobian.
        noise_logpdf = clipped_gaussian_logprob(vals[2], vals[1]; σ=p.action_std, a_min=p.action_min, a_max=p.action_max)
        jacobian_matrix = Dw_transition_f(p, s, vals[1], vals[2])
        j_logdet = -0.5 * logabsdet(jacobian_matrix' * jacobian_matrix)[1]
        return noise_logpdf + j_logdet
    end
    # Full derivative formula, and using the fact that ∂w/∂a = -1
    # df/da = ∂f/∂a + ∂f/∂w * ∂w/∂a
    grad = ForwardDiff.gradient(f3_for_diff, SA[a, w])
    return grad[1] - grad[2]
end

##########################################
### Shared function for Simplified/ODE ###
##########################################

const ProbMountainCarMDP = Union{SimpleMountainCarMDP,ODEMountainCarMDP}

@with_kw struct ProbMountainCarPOMDP <: POMDP{CarState,Float64,Float64}
    mdp::ProbMountainCarMDP = SimpleMountainCarMDP()
    meas_std::Float64 = 0.03
end

const ProbMountainCarProblem = Union{ProbMountainCarMDP,ProbMountainCarPOMDP}
UnderlyingMDP(p::ProbMountainCarMDP) = p
UnderlyingMDP(p::ProbMountainCarPOMDP) = p.mdp

function project_action(p::ProbMountainCarProblem, a::T) where {T<:AbstractFloat}
    p = UnderlyingMDP(p)
    return clamp(a, p.action_min, p.action_max)
end

function check_w_bounds(p::ProbMountainCarMDP, a::T, w::T) where {T<:AbstractFloat}
    a_noise = a + w
    a_noise_rounded = round(a_noise, digits=13)
    if (a + w) > p.action_max
        if a_noise_rounded > p.action_max
            return false, a, w
        else
            w = p.action_max - a
            return true, a, w
        end
    elseif (a + w) < p.action_min
        if a_noise_rounded < p.action_min
            return false, a, w
        else
            w = p.action_min - a
            return true, a, w
        end
    end
    return true, a, w
end

transition_f(pp::ProbMountainCarPOMDP, s::CarState, a::T, w::T) where {T<:AbstractFloat} = transition_f(UnderlyingMDP(pp), s, a, w)

"""
The Jacobian matrix of the transition function f(s, a, w) with respect to w.
Assuming that a + w is within the bounds of the action space.
"""
Dw_transition_f(pp::ProbMountainCarPOMDP, s::CarState, a::T, w::T) where {T<:AbstractFloat} = Dw_transition_f(UnderlyingMDP(pp), s, a, w)

function POMDPs.transition(pp::ProbMountainCarProblem, s::CarState, a::T) where {T<:AbstractFloat}
    ImplicitDistribution(pp, s, a) do pp, s, a, rng
        p = UnderlyingMDP(pp)
        w = p.action_std * randn(rng)
        sp = transition_f(p, s, a, w)
        return sp
    end
end

# This is not the real mean state, rather just a representative state assuming ML noise parameter.
transition_mean_state(pp::ProbMountainCarProblem, s::CarState, a::T) where {T<:AbstractFloat} = transition_f(UnderlyingMDP(pp), s, a, 0.0)

"""
Find w that solves s' = f(s, a, w) for given s, a, s', assuming that the solution exists and that it is unique.
"""
inverse_transition_f(pp::ProbMountainCarProblem, s::CarState, a::T, sp::CarState) where {T<:AbstractFloat} = inverse_transition_f(UnderlyingMDP(pp), s, a, sp)

"""
Returns the distribution of the noise parameter p_w(⋅ | s, a), given the current state and action.
"""
function noise_dist(p::ProbMountainCarMDP, s::CarState, a::T) where {T<:AbstractFloat}
    return truncated(Normal(0.0, p.action_std), p.action_min - a, p.action_max - a)
end

transition_logpdf(pp::ProbMountainCarProblem, s::CarState, a::T, sp::CarState) where {T<:AbstractFloat} = transition_logpdf(UnderlyingMDP(pp), s, a, sp)
transition_gradlogpdf(pp::ProbMountainCarProblem, s::CarState, a::T, sp::CarState) where {T<:AbstractFloat} = transition_gradlogpdf(UnderlyingMDP(pp), s, a, sp)

function POMDPs.reward(pp::ProbMountainCarProblem, s::CarState, a::T, sp::CarState) where {T<:AbstractFloat}
    p = UnderlyingMDP(pp)
    if sp[1] > p.x_max
        state_reward = p.reward_goal + p.velocity_goal_penalty * abs(sp[2])
    elseif sp[1] < p.x_min || sp[2] < p.v_min || sp[2] > p.v_max
        state_reward = p.failure_penalty
    else
        state_reward = p.reward_step + p.distance_from_goal_penalty * abs(p.x_max - sp[1])
    end
    action_reward = p.action_penalty_coeff * a^2
    return state_reward + action_reward
end

function POMDPs.observation(p::ProbMountainCarPOMDP, sp::CarState)
    return Normal(sp[1], p.meas_std)
end

POMDPs.discount(pp::ProbMountainCarProblem) = UnderlyingMDP(pp).discount
POMDPs.initialstate(::ProbMountainCarProblem) = ImplicitDistribution(rng -> CarState([-0.2 * rand(rng), 0.0]))
POMDPs.isterminal(pp::ProbMountainCarProblem, s::CarState) = s[1] > UnderlyingMDP(pp).x_max || s[1] < UnderlyingMDP(pp).x_min || s[2] < UnderlyingMDP(pp).v_min || s[2] > UnderlyingMDP(pp).v_max

## Action sampling

struct CarActionSpace end
rand(rng::AbstractRNG, ::CarActionSpace) = 2 * rand(rng) - 1.0  # Random action in [-1, 1]
POMDPs.actions(::ProbMountainCarMDP) = CarActionSpace()
POMDPs.actions(::ProbMountainCarPOMDP) = CarActionSpace()

## Rendering

function draw_actions(xy_center, actions::Vector{Float64}, mc="red", action_length=0.15, ma=0.7)
    point_array = [[xy_center, xy_center .+ (a * action_length, 0.0)] for a in actions]

    if length(actions) > 1
        # The blues start from light skyblue and end at deep navy blue
        # https://docs.juliaplots.org/latest/generated/colorschemes/
        cg = Plots.cgrad(:blues, range(0, 1, length=length(actions)))
    else
        # The default color is red, so if there was no optimization, we use a red arrow to indicate it
        cg = [Plots.coloralpha(Plots.color(mc), ma)]
    end
    colors = [Plots.RGBA(t.r, t.g, t.b, ma) for t in cg]
    return point_array, colors
end

function action_opt_trajectory(step)
    # Sorry for the ugly namespace trespassing
    if !hasproperty(step, :action_info) || step[:action_info] === nothing || !haskey(step.action_info, :tree) || !(step.action_info[:tree] isa Main.ActionGradientMCTS.ActionGradTree)
        return nothing
    end
    tree = step.action_info[:tree]
    sanode = Main.MCTS.best_sanode(tree, 1)
    return Main.ActionGradientMCTS.a_init(tree, sanode)
end

function render_step_actions(step, xy_center)
    actions = action_opt_trajectory(step)
    if actions !== nothing
        return draw_actions(xy_center, actions)
    elseif hasproperty(step, :a)
        return draw_actions(xy_center, [step.a])
    end
end

POMDPTools.render(pp::ProbMountainCarProblem, step::NamedTuple) = begin
    p = UnderlyingMDP(pp)
    track_func = p isa SimpleMountainCarMDP ? mountain : hill
    cx = step.s[1]
    cy = track_func(cx)
    car_radius = 0.035
    car_cy = cy + car_radius

    actions_point_array, actions_colors = render_step_actions(step, (cx, car_cy))

    car = (context(), Compose.circle(cx, car_cy, car_radius), fill("grey"),
        (context(), arrow(), fill(nothing), (context(), line(actions_point_array), stroke(actions_colors)))
    )

    track = (context(), line([(x, track_func(x)) for x in p.x_min:0.01:p.x_max]), Compose.stroke("black"))
    goal = (context(), star(p.x_max, 0.05 + track_func(p.x_max), -0.035, 5), fill("gold"), Compose.stroke("black"))
    bg = (context(), Compose.rectangle(), fill("white"))

    if p isa SimpleMountainCarMDP
        ctx = context(0.7, 0.05, 0.6, 0.9, mirror=Mirror(0, 0, 0.5))
    else
        ctx = context(0.485, 0.05, 0.505, 0.9, mirror=Mirror(0, 0, 0.35))
    end
    return compose(context(), (ctx, car, track, goal), bg)
end

export CarState,
    SimpleMountainCarMDP,
    ODEMountainCarMDP,
    ProbMountainCarMDP,
    ProbMountainCarPOMDP,
    ProbMountainCarProblem,
    UnderlyingMDP,
    transition_mean_state

end  # module
