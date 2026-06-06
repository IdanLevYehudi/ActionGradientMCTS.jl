using POMDPTools.ModelTools: GenerativeBeliefMDP, BackwardCompatibleTerminalBehavior, determine_gbmdp_state_type

import Lazy: @forward

export GenerativeBeliefPropMDP

"""
Slightly modifying this function from pomdps.jl in ParticleFilters.jl package to ignore terminal states
"""
function predict_ignore_terminal!(pm, m::POMDP, b, a, rng)
    for i in 1:n_particles(b)
        s = particle(b, i)
        sp = @gen(:sp)(m, s, a, rng)
        pm[i] = sp
    end
end

function reweight_with_terminal!(wm, m::POMDP, b, a, pm, o)
    for i in 1:n_particles(b)
        s = particle(b, i)
        sp = pm[i]
        if isterminal(m, s) || isterminal(m, sp)
            wm[i] = 0.0
        else
            wm[i] = obs_weight(m, s, a, sp, o)
        end
    end
end

function copy_bminus(b::ParticleCollection, pm)
    return ParticleCollection(copy(pm))
end

function copy_bminus(b::WeightedParticleBelief, pm)
    return WeightedParticleBelief(copy(pm), b.weights, b.weight_sum, nothing)
end

function update_prop_posterior(up::BasicParticleFilter, b::AbstractParticleBelief, a, o)
    pm = up._particle_memory
    wm = up._weight_memory
    resize!(pm, n_particles(b))
    predict_ignore_terminal!(pm, up.predict_model, b, a, up.rng)
    bminus = copy_bminus(b, pm)
    resize!(wm, n_particles(b))
    reweight_with_terminal!(wm, up.reweight_model, b, a, pm, o)
    bp = resample(up.resampler,
        WeightedParticleBelief(pm, wm, sum(wm), nothing),
        up.predict_model,
        up.reweight_model,
        b, a, o,
        up.rng)
    return bminus, bp
end

struct GenerativeBeliefPropMDP{P<:POMDP,U<:BasicParticleFilter,T,B,A} <: MDP{B,A}
    gmdp::GenerativeBeliefMDP{P,U,T,B,A}
end

function GenerativeBeliefPropMDP(pomdp, updater; terminal_behavior=BackwardCompatibleTerminalBehavior(pomdp, updater))
    gmdp = GenerativeBeliefMDP(pomdp, updater, terminal_behavior=terminal_behavior)
    B = determine_gbmdp_state_type(gmdp.pomdp, updater, terminal_behavior)
    GenerativeBeliefPropMDP{typeof(gmdp.pomdp),
        typeof(gmdp.updater),
        typeof(gmdp.terminal_behavior),
        B,
        actiontype(gmdp)
    }(gmdp)
end

function POMDPs.gen(bmdp::GenerativeBeliefPropMDP, b, a, rng::AbstractRNG)
    s = rand(rng, b)
    # We always want to have b-propagated even if bp is terminal.
    sp, o, r, gen_info = @gen(:sp, :o, :r, :info)(bmdp.gmdp.pomdp, s, a, rng)

    # TODO: Calculate reward as mean over all samples rather than single sample
    if gen_info === nothing
        gen_info = NamedTuple()
    end
    bminus, bp = update_prop_posterior(bmdp.gmdp.updater, b, a, o)
    info = merge(gen_info, (sminus=bminus, sampled_s_sp=(s, sp)))
    if isterminal(bmdp.gmdp.pomdp, s)
        r = 0.0
    end
    return (sp=bp, r=r, info=info)
end

POMDPs.isterminal(bmdp::GenerativeBeliefPropMDP, b::B) where {B<:AbstractParticleBelief} = all(isterminal(bmdp.gmdp.pomdp, s) for (s, w) in weighted_particles(b) if w > 0.0)
POMDPs.isterminal(bmdp::GenerativeBeliefPropMDP, t::TerminalState) = isterminal(bmdp.gmdp, t)

# Forwarded functions - will act on the gmdp field of the GenerativeBeliefPropMDP instance
@forward GenerativeBeliefPropMDP.gmdp POMDPs.initialstate, POMDPs.actions, POMDPs.isterminal, POMDPs.discount
