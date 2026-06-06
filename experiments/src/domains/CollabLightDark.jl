module CollabLightDark

# Agents are deployed at a random ring in 2D around the origin
# First agent receives a very noisy observations of its location; The other agent's locations are distance to previous agent, cov decreases with distance to closest agent
# Reward is sum of square distance from origin for all agents
# Scenario terminates after 20 times steps or when all agents withing 0.1 of origin

using Parameters
using Random
using Distributions
using Plots
using Suppressor: @suppress, @suppress_err
import Cairo
import Base: rand
using StaticArrays
using Statistics
using LinearAlgebra: norm, Diagonal
using SpecialFunctions

using POMDPs
using POMDPTools
import POMDPTools.ModelTools: UnderlyingMDP
using POMDPGifs
using MCTS
using ParticleFilters
import ParticleFilters: obs_weight, ParticleCollection
using Enzyme
using Images


export CollabLightDarkMDP,
    CollabLightDarkPOMDP,
    CollabLightDarkProblem,
    VecN,
    d_slice,
    NStateDist,
    transition_mean_state,
    DKNStateObsDist,
    obs_pos_diff,
    CLDInitDist,
    CollabLightDarkActionSpace,
    _project_action,
    CLDStraightToGoalPolicy,
    CLDStraightToGoalSolver


VecN(N) = SVector{N,Float64}

struct CollabLightDarkMDP{D,K,N} <: MDP{SVector{N,Float64},SVector{N,Float64}}
    action_max_radius::Float64
    goal_position::SVector{D,Float64}
    goal_tolerance::Float64
    obstacles::Vector{SVector{D,Float64}}
    obstacles_radii::Vector{Float64}
    agent_transition_std::Float64
    transition_cov::Diagonal
    transition_dist::MvNormal
    b0_radius::Float64
    grad_log_transition_func::Function
    discount::Float64
    reward_penalty_goal_distsq::Float64
    reward_success::Float64
    reward_penalty_obstacles::Float64
    max_action_prob::Float64
end

function CollabLightDarkMDP{D,K}(;
    action_max_radius::Float64=1.5,
    goal_position::SVector{D,Float64}=SVector{D,Float64}(zeros(D)),
    goal_tolerance::Float64=0.1,
    obstacles::Vector{SVector{D,Float64}}=[SVector{D,Float64}(vcat(2.0, zeros(D - 1)))],
    obstacles_radii::Vector{Float64}=[1.0],
    agent_transition_std::Float64=0.025,
    b0_radius::Float64=0.5,
    discount::Float64=0.99,
    reward_penalty_goal_distsq::Float64=-0.1,
    reward_success::Float64=10.0,
    reward_penalty_obstacles::Float64=-5.0,
    max_action_prob::Float64=0.5,
) where {D,K}
    # Constructor ensures N = K * D
    N = D * K
    transition_cov = Diagonal(VecN(N)(repeat([agent_transition_std^2], N)))
    transition_dist = MvNormal(transition_cov)

    # This disgusting mess is a workaround for the fact that a local variable cannot be enclosed in a function signature
    # So we enclose N by passing a::SVector{N} to a function, that returns the actual function that we want
    # This is to make sure that the returns function actually has a signature of (VecN(N), VecN(N), VecN(N)) -> VecN(N)
    function create_f(a::SVector{N,Float64}) where {N}
        (s::VecN(N), a::VecN(N), sp::VecN(N)) -> begin
            ret = Enzyme.gradient(Forward, logpdf, Const(transition_dist), sp - s - a)
            return -ret[2]  # Need to multiply by the inner derivative of (sp - s - a) w.r.t. a - which is -1.
        end
    end
    grad_log_transition_func = create_f(SVector{N,Float64}(zeros(N)))

    return CollabLightDarkMDP{D,K,N}(
        action_max_radius,
        goal_position,
        goal_tolerance,
        obstacles,
        obstacles_radii,
        agent_transition_std,
        transition_cov,
        transition_dist,
        b0_radius,
        grad_log_transition_func,
        discount,
        reward_penalty_goal_distsq,
        reward_success,
        reward_penalty_obstacles,
        max_action_prob
    )
end

struct CollabLightDarkPOMDP{D,K,N} <: POMDP{SVector{N,Float64},SVector{N,Float64},SVector{N,Float64}}
    mdp::CollabLightDarkMDP{D,K,N}
    beacon_pos::SVector{D,Float64}
    meas_std_by_dist_k::Float64
    meas_std_by_dist_alpha::Float64
    max_obs_std::Float64

    function CollabLightDarkPOMDP{D,K}(;
        mdp::CollabLightDarkMDP=CollabLightDarkMDP{D,K}(),
        beacon_pos::SVector{D,Float64}=SVector{D,Float64}(vcat(2.5, zeros(D - 1))),
        meas_std_by_dist_k::Float64=0.25,
        meas_std_by_dist_alpha::Float64=2.0,
        max_obs_std::Float64=15.0,
    ) where {D,K}
        # Constructor ensures N = K * D
        N = D * K
        return new{D,K,N}(
            mdp,
            beacon_pos,
            meas_std_by_dist_k,
            meas_std_by_dist_alpha,
            max_obs_std
        )
    end
end

const CollabLightDarkProblem{D,K,N} = Union{CollabLightDarkMDP{D,K,N},CollabLightDarkPOMDP{D,K,N}}

UnderlyingMDP(p::CollabLightDarkMDP) = p
UnderlyingMDP(p::CollabLightDarkPOMDP) = p.mdp

d_slice(s::AbstractArray, D, i) = @view s[(i-1)*D+1:i*D]

## Transition

struct NStateDist{N}
    mean::SVector{N,Float64}
    dist::MvNormal
end

function rand(rng::AbstractRNG, d::NStateDist{N}) where {N}
    x_out = zero(MVector{N,Float64})
    rand!(rng, d.dist, x_out)
    return d.mean + VecN(N)(x_out)
end

POMDPs.pdf(d::NStateDist{N}, s::SVector{N,Float64}) where {N} = pdf(d.dist, s - d.mean)
Distributions.logpdf(d::NStateDist{N}, s::SVector{N,Float64}) where {N} = logpdf(d.dist, s - d.mean)
Distributions.mean(d::NStateDist{N}) where {N} = mean(d.dist) + d.mean
Distributions.cov(d::NStateDist{N}) where {N} = cov(d.dist)
# gradlogpdf is about 2.5x faster than the autodiff version
Distributions.gradlogpdf(d::NStateDist{N}, s::SVector{N,Float64}) where {N} = gradlogpdf(d.dist, s - d.mean)
# Computing the autodiff for a constant covariance matrix is 10x faster than holding m as const
_grad_log_transition(m::CollabLightDarkProblem{D,K,N}, s::SVector{N,Float64}, a::SVector{N,Float64}, sp::SVector{N,Float64}) where {D,K,N} = UnderlyingMDP(m).grad_log_transition_func(s, a, sp)

function _project_action(pp::CollabLightDarkProblem{D,K,N}, a::SVector{N,Float64}) where {D,K,N}
    p = UnderlyingMDP(pp)
    agent_actions = zero(MVector{N,Float64})
    for i in 1:K
        a_i = d_slice(a, D, i)
        n_a_i = norm(a_i)
        n_a_i_clamped = a_i
        if n_a_i > p.action_max_radius
            n_a_i_clamped = a_i * p.action_max_radius / n_a_i
        end
        d_slice(agent_actions, D, i) .= n_a_i_clamped
    end
    return VecN(N)(agent_actions)
end

function POMDPs.transition(pp::CollabLightDarkProblem{D,K,N}, s::SVector{N,Float64}, a::SVector{N,Float64}) where {D,K,N}
    p = UnderlyingMDP(pp)
    a_clamped = _project_action(p, a)
    next_state_mean = s + a_clamped
    return NStateDist{N}(next_state_mean, p.transition_dist)
end

function transition_mean_state(pp::CollabLightDarkProblem{D,K,N}, s::SVector{N,Float64}, a::SVector{N,Float64}) where {D,K,N}
    p = UnderlyingMDP(pp)
    a_clamped = _project_action(p, a)
    return s + a_clamped
end

## Reward

function _reward(pp::CollabLightDarkProblem{D,K,N}, sp::SVector{N,Float64}) where {D,K,N}
    p = UnderlyingMDP(pp)
    reward_sum = 0.0
    is_ter = isterminal(p, sp)
    for i in 1:K
        agent_pos = d_slice(sp, D, i)
        goal_distance = norm(agent_pos - p.goal_position)
        reward_sum += p.reward_penalty_goal_distsq * goal_distance^2
        T = p.goal_tolerance / 2
        reward_sum += -(p.reward_success / 5) * exp(-0.5 * ((goal_distance - 10 * T) / (2 * T))^2)
        for j in 1:length(p.obstacles)
            obstacle_pos = p.obstacles[j]
            obstacle_radius = p.obstacles_radii[j]
            dist_from_obstacle = norm(agent_pos - obstacle_pos)
            reward_sum += p.reward_penalty_obstacles / (1 + exp(50 * (dist_from_obstacle - (obstacle_radius - 0.05))))
        end
        if is_ter
            reward_sum += p.reward_success * exp(-0.5 * (goal_distance / T)^2)
        end
    end
    return reward_sum / K
end

function _grad_reward(pp::CollabLightDarkProblem{D,K,N}, s, a, sp) where {D,K,N}
    return zero(SVector{N,Float64})  # The reward is currently not a function of the action
end

POMDPs.reward(pp::CollabLightDarkProblem{D,K,N}, s::SVector{N,Float64}, a::SVector{N,Float64}, sp::SVector{N,Float64}) where {D,K,N} = _reward(pp, sp)
POMDPs.discount(pp::CollabLightDarkProblem) = UnderlyingMDP(pp).discount

## Observation

struct DKNStateObsDist{D,K,N}
    pp::CollabLightDarkPOMDP{D,K,N}
    mean::SVector{N,Float64}
    dist::MvNormal
end

function rand(rng::AbstractRNG, d::DKNStateObsDist{D,K,N}) where {D,K,N}
    x_out = zero(MVector{N,Float64})
    rand!(rng, d.dist, x_out)
    return d.mean + VecN(N)(x_out)
end

function obs_pos_diff(pp::CollabLightDarkPOMDP{D,K,N}, s::SVector{N,Float64}) where {D,K,N}
    diff = @views vcat(s[1:D] - pp.beacon_pos, s[D+1:end] - s[1:N-D])
    return diff
end

function POMDPs.pdf(d::DKNStateObsDist{D,K,N}, o::SVector{N,Float64}) where {D,K,N}
    return pdf(d.dist, o - d.mean)
end

function _obs_std(pp::CollabLightDarkPOMDP{D,K,N}, d) where {D,K,N}
    return min(pp.max_obs_std, pp.meas_std_by_dist_k * (d + d^pp.meas_std_by_dist_alpha))
end

function POMDPs.observation(pp::CollabLightDarkPOMDP{D,K,N}, sp::SVector{N,Float64}) where {D,K,N}
    # The observation variance for each agent is a diagonal matrix with the same σ^2
    # σ^2 for each agent is a function of the distance to the preceding agent. For the first agent it is the distance from the goal.
    # σ^2 = k * max(d, min_obs_d)^α
    obs_std = MVector{N,Float64}(undef)
    s_diff = obs_pos_diff(pp, sp)
    for i in 1:K
        d = norm(d_slice(s_diff, D, i))
        d_slice(obs_std, D, i) .= _obs_std(pp, d)
    end
    return DKNStateObsDist{D,K,N}(pp, s_diff, MvNormal(Diagonal(VecN(N)(obs_std .^ 2))))
end

## Initial and terminal conditions

# b0 is each agent distributed randomly on the unit D-Sphere at radius b0_radius
struct CLDInitDist{D,K,N}
    pp::CollabLightDarkProblem{D,K,N}
end

sampletype(::Type{CLDInitDist{D,K,N}}) where {D,K,N} = SVector{N,Float64}

function rand(rng::AbstractRNG, d::CLDInitDist{D,K,N}) where {D,K,N}
    p = UnderlyingMDP(d.pp)
    agents = Vector{SVector{D,Float64}}(undef, K)
    for i in 1:K
        agent = VecN(D)(randn(rng, D))
        agent = p.b0_radius * agent / norm(agent)
        agents[i] = agent
    end
    return SVector{N,Float64}(reduce(vcat, agents))
end

POMDPs.initialstate(pp::CollabLightDarkProblem{D,K,N}) where {D,K,N} = CLDInitDist{D,K,N}(pp)

function POMDPs.isterminal(pp::CollabLightDarkProblem{D,K,N}, s::SVector{N,Float64}) where {D,K,N}
    mdp = UnderlyingMDP(pp)
    return all(norm(d_slice(s, D, i) - mdp.goal_position) < mdp.goal_tolerance for i in 1:K)
end
POMDPs.isterminal(pp::CollabLightDarkProblem{D,K,N}, s::AbstractParticleBelief{SVector{N,Float64}}) where {D,K,N} = all(isterminal(pp, s_i) for s_i in particles(s))

## Visualization

function circle_shape(x, y, r, n=500)
    θ = LinRange(0, 2π, n)
    return x .+ r * cos.(θ), y .+ r * sin.(θ)
end

function make_background_image(pp::CollabLightDarkPOMDP{2,K,N}, c1, x, y) where {K,N}
    p = UnderlyingMDP(pp)
    max_r = p.reward_success
    min_r = -max_r / 4
    max_o = 3.0
    min_o = 0.0

    norm_r(r) = clamp((r - min_r) / (max_r - min_r), 0.0, 1.0)
    norm_o(o) = clamp((o - min_o) / (max_o - min_o), 0.0, 1.0)

    background_img_array = Array{RGB{Float64},2}(undef, length(y), length(x))
    for (j, yi) in enumerate(y)
        for (i, xi) in enumerate(x)
            r = _reward(pp, SVector{N,Float64}(repeat([xi, yi], outer=K)))
            o = _obs_std(pp, norm([xi, yi] - pp.beacon_pos))
            r_color = HSV(get(c1, norm_r(r)))

            # value_factor = min((1-0.25*(norm_o(o)^(1/8)))/0.93, 1.0)
            value_factor = 1 - 0.25(norm_o(o))
            ro_color = HSV(r_color.h, r_color.s, value_factor * r_color.v)
            background_img_array[j, i] = RGB(ro_color)
        end
    end
    background_img = colorview(RGB, background_img_array)
    return background_img
end

function draw_background(pp::CollabLightDarkProblem{2,K,N}, pl, x_lims=[-2, 4], y_lims=[-2, 4]) where {K,N}
    p = UnderlyingMDP(pp)

    x = range(x_lims[1], x_lims[2], length=300)
    y = range(y_lims[1], y_lims[2], length=300)

    # Plot the reward function as a heatmap
    c1 = cgrad([:blue, :white, :red], [0.19, 0.21])
    max_r = p.reward_success
    z = [_reward(pp, SVector{N,Float64}(repeat([xi, yi], outer=K))) for yi in y, xi in x] ./ K  # Compute rewards for the grid
    heatmap!(pl, x, y, z, color=c1, clim=(-max_r / 4, max_r), xticks=x_lims[1]:1:x_lims[2], yticks=y_lims[1]:1:y_lims[2])

    if pp isa CollabLightDarkPOMDP
        background_img = make_background_image(pp, c1, x, y)
        plot!(pl, x, y, background_img, yflip=false)
    end

    ## Draw initial region as red circle, radius b0_radius
    plot!(pl, circle_shape(0.0, 0.0, p.b0_radius), lw=1.0, c=:red, linecolor=:red, legend=false, fill_alpha=0.0, aspect_ratio=1)
    ## Draw goal region as blue circle
    plot!(pl, circle_shape(p.goal_position[1], p.goal_position[2], p.goal_tolerance), lw=1.0, c=:blue, linecolor=:blue, legend=false, fill_alpha=0.3, aspect_ratio=1)
end

function draw_agents!(pp::CollabLightDarkProblem{2,K,N}, pl, s::SVector{N,Float64}, mc=:black, ms=3.0, ma=1.0) where {K,N}
    xs = []
    ys = []
    for i in 1:K
        agent_pos = d_slice(s, 2, i)
        push!(xs, agent_pos[1])
        push!(ys, agent_pos[2])
    end
    scatter!(pl, xs, ys, mc=mc, ms=ms, markeralpha=ma, legend=false)
end

function rgba_list(color_name, alphas)
    return [Plots.coloralpha(Plots.color(color_name), a) for a in alphas]
end
is_iterable(obj) = Base.IteratorSize(typeof(obj)) != Base.HasShape{0}()

function draw_agents!(pp::CollabLightDarkProblem{2,K,N}, pl, states::Vector{SVector{N,Float64}}, mc="black", ms=3.0, ma=1.0) where {K,N}
    xs = []
    ys = []
    for s in states
        for i in 1:K
            agent_pos = d_slice(s, 2, i)
            push!(xs, agent_pos[1])
            push!(ys, agent_pos[2])
        end
    end
    if is_iterable(ma)
        color = repeat(rgba_list(mc, ma), inner=K)
    else
        color = Plots.color_alpha(mc, ma)
    end
    scatter!(pl, xs, ys, ms=ms, c=color, alpha=color, legend=false)
end

function draw_observation!(pp::CollabLightDarkPOMDP{2,K,N}, pl, o::SVector{N,Float64}, sp::SVector{N,Float64}, mc=:green, ms=3.0, ma=1.0) where {K,N}
    # Draw the observations of the agents
    for i in 1:K
        obs_pos = d_slice(o, 2, i)
        if i == 1
            agent_pos = pp.beacon_pos
        else
            agent_pos = d_slice(sp, 2, i - 1)
        end
        scatter!(pl, [obs_pos[1] + agent_pos[1]], [obs_pos[2] + agent_pos[2]], mc=mc, ms=ms, ma=ma, legend=false)
    end

    # Draw the beacon
    scatter!(pl, [pp.beacon_pos[1]], [pp.beacon_pos[2]], mc=mc, ms=10.0, ma=ma, markershape=:star5, legend=false)
end

function draw_belief!(pp::CollabLightDarkPOMDP{2,K,N}, pl, b::AbstractParticleBelief{SVector{N,Float64}}, mc="grey") where {K,N}
    alphas = (weights(b) ./ weight_sum(b)) .^ 0.25
    draw_agents!(pp, pl, particles(b), mc, 1.0, alphas)
end

function draw_actions!(pp::CollabLightDarkProblem{2,K,N}, pl, state::SVector{N,Float64}, actions::Vector{SVector{N,Float64}}, mc="red", ms=0.5, ma=0.7) where {K,N}
    xs = []
    ys = []
    us = []
    vs = []
    for a in actions
        for i in 1:K
            agent_i = d_slice(state, 2, i)
            action_i = d_slice(a, 2, i)
            push!(xs, agent_i[1])
            push!(ys, agent_i[2])
            push!(us, action_i[1])
            push!(vs, action_i[2])
        end
    end

    if length(actions) > 1
        # The blues start from light skyblue and end at deep navy blue
        # https://docs.juliaplots.org/latest/generated/colorschemes/
        cg = cgrad(:blues, range(0, 1, length=length(actions)))
    else
        # The default color is red, so if there was no optimization, we use a red arrow to indicate it
        cg = [Plots.coloralpha(Plots.color(mc), ma)]
    end
    colors = repeat([RGBA(t.r, t.g, t.b, ma) for t in cg], inner=K)
    @suppress begin
        quiver!(pl, xs, ys, quiver=(us, vs), ms=ms, c=colors, legend=false)
    end
end

function action_opt_trajectory(step)
    # Sorry for the ugly namespace trespassing
    if !hasproperty(step, :action_info) || !haskey(step.action_info, :tree) || !(step.action_info[:tree] isa Main.ActionGradientMCTS.ActionGradTree)
        return nothing
    end
    tree = step.action_info[:tree]
    sanode = Main.MCTS.best_sanode(tree, 1)
    return Main.ActionGradientMCTS.a_init(tree, sanode)
end

function render_step_actions!(pp, pl, step, s)
    actions = action_opt_trajectory(step)
    if actions !== nothing
        draw_actions!(pp, pl, s, actions)
    elseif hasproperty(step, :a)
        draw_actions!(pp, pl, s, [step.a])
    end
end

function POMDPTools.render(pp::CollabLightDarkPOMDP{2,K,N}, step::NamedTuple; dpi::Int=100) where {K,N}
    gr()
    x_lims = [-1, 3]
    y_lims = [-1, 3]
    pl = plot(bg=:white, x_lims, y_lims, color=:white, ylimits=y_lims, xlimits=x_lims, dpi=dpi, widen=false, tickdirection=:out, size=(x_lims[2] - x_lims[1] + 1.0, y_lims[2] - y_lims[1] + 0.0) .* 70)
    pl = draw_background(pp, pl, x_lims, y_lims)
    if hasproperty(step, :bp)
        draw_belief!(pp, pl, step.bp)
    end

    if hasproperty(step, :sp)
        if hasproperty(step, :o)
            draw_observation!(pp, pl, step.o, step.sp)
        end
        draw_agents!(pp, pl, step.sp)
    end

    if hasproperty(step, :b)
        draw_belief!(pp, pl, step.b, "orange")
        render_step_actions!(pp, pl, step, mean(step.b))
        if hasproperty(step, :s)
            draw_agents!(pp, pl, step.s, "red")
        end
    elseif hasproperty(step, :s)
        draw_agents!(pp, pl, step.s, "red")
        render_step_actions!(pp, pl, step, step.s)
    end

    return pl
end

## Action Space

struct CollabLightDarkActionSpace{D,K,N}
    pp::CollabLightDarkProblem{D,K,N}
end

function rand(rng::AbstractRNG, d::CollabLightDarkActionSpace{D,K,N}) where {D,K,N}
    # Sample for each agent a random action in the unit D-Sphere
    # The direction is sampled by normalizing a random vector of gaussian RVs
    # The radius is sampled from a uniform distribution that is scaled ^(1/D)
    # The final radius is multiplied by the action_max_radius
    p = UnderlyingMDP(d.pp)
    actions = Vector{SVector{D,Float64}}(undef, K)
    for i in 1:K
        action = VecN(D)(randn(rng, D))
        action = action / norm(action)
        # p.max_action_prob is the probability to sample the maximum action radius
        # Therefore the probability to sample a non-maximal radius is (1 - p.max_action_prob)
        # And since a ball's volume scales ∝ r^D, we need to scale the random radius by (1 / (1 - p.max_action_prob))^(1/D) so that with probability p.max_action_prob the the resulting radius is greater than 1.
        action_radius = p.action_max_radius * (rand(rng) / (1 - p.max_action_prob))^(1 / D)
        actions[i] = action_radius * action
    end
    return _project_action(d.pp, SVector{N,Float64}(reduce(vcat, actions)))
end

POMDPs.actions(pp::CollabLightDarkProblem{D,K,N}) where {D,K,N} = CollabLightDarkActionSpace{D,K,N}(pp)

## Heuristics

@with_kw mutable struct CLDStraightToGoalPolicy{D,K,N} <: Policy
    prob::CollabLightDarkProblem{D,K,N}
    rng::Union{TaskLocalRNG,AbstractRNG} = Random.GLOBAL_RNG
    sigma::Float64 = 0.0
end

function POMDPs.action(p::CLDStraightToGoalPolicy{D,K,N}, s::SVector{N,Float64}) where {D,K,N}
    m = UnderlyingMDP(p.prob)
    diff = SVector{N,Float64}(repeat(m.goal_position, K) .- s)
    if p.sigma > 0.0
        diff += randn(p.rng, N) * p.sigma
    end
    return _project_action(m, diff)
end
POMDPs.action(p::CLDStraightToGoalPolicy{D,K,N}, b::ParticleFilters.AbstractParticleBelief{SVector{N,Float64}}) where {D,K,N} = action(p, mean(b))  # Trying mean instead of a random particle

@with_kw mutable struct CLDStraightToGoalSolver <: POMDPs.Solver
    rng::Union{TaskLocalRNG,AbstractRNG} = Random.GLOBAL_RNG
    sigma::Float64 = 0.0
end
POMDPs.solve(s::CLDStraightToGoalSolver, pp::CollabLightDarkProblem{D,K,N}) where {D,K,N} = CLDStraightToGoalPolicy{D,K,N}(prob=UnderlyingMDP(pp), rng=s.rng, sigma=s.sigma)

end  # module
