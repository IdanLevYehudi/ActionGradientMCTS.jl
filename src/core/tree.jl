mutable struct ActionGradTree{S,A}
    dpw_tree::DPWTree{S,A}
    sminus_labels::Vector{S}  # The proagated states for each state node (bminus for belief-MDPs, sp for MDPs)
    sampled_s_sps::Vector{Any}  # Tuples of the sampled state and resulting state in each episoe, saved for each posterior node. Since these are the MDP states rather than belief-states, we leave these as any for now. A more technical approach would be to add a type parameter to the ActionGradTree struct.
    a_proposals::Vector{Union{A,Nothing}}  # The action that was used to generate the propagated belief, saved at each potserior node.
    a_init::Vector{Vector{A}}  # For each action branch, the list of actions that were used to generate the propagated beliefs after each gradient update
    value_updater::AbstractValueUpdater

    function ActionGradTree{S,A}(sz::Int=1000, value_updater::Union{AbstractValueUpdater,Nothing}=nothing) where {S,A}
        sz = min(sz, 100_000)
        if value_updater === nothing
            value_updater = ValueUpdaterSNMISMC(sz)
        end
        new(DPWTree{S,A}(sz),
            sizehint!(S[], sz),  # sminus_labels
            sizehint!(Any[], sz),  # sampled_s_sps
            sizehint!(Union{A,Nothing}[], sz),  # a_proposals
            sizehint!(Vector{A}[], sz),  # a_init
            value_updater
        )
    end
end

# ActionGradTree Getter methods
total_n_visits(tree::ActionGradTree, index::Int) = tree.dpw_tree.total_n[index]
MCTS.children(tree::ActionGradTree, index::Int) = tree.dpw_tree.children[index]
s_labels(tree::ActionGradTree, index::Int) = tree.dpw_tree.s_labels[index]
s_lookup(tree::ActionGradTree{S,A}, s::S) where {S,A} = tree.dpw_tree.s_lookup[s]

has_state_index(tree::ActionGradTree, index::Int) = haskey(tree.dpw_tree.s_labels, index)
has_state(tree::ActionGradTree{S,A}, s::S) where {S,A} = haskey(tree.dpw_tree.s_lookup, s)

q_value(tree::ActionGradTree, sanode::Int) = tree.dpw_tree.q[sanode]
n_visits(tree::ActionGradTree, sanode::Int) = tree.dpw_tree.n[sanode]
transitions(tree::ActionGradTree, sanode::Int) = tree.dpw_tree.transitions[sanode]

a_labels(tree::ActionGradTree, sanode::Int) = tree.dpw_tree.a_labels[sanode]
a_lookup(tree::ActionGradTree{S,A}, snode::Int, a::A) where {S,A} = tree.dpw_tree.a_lookup[(snode, a)]
has_action(tree::ActionGradTree{S,A}, snode::Int, a::A) where {S,A} = haskey(tree.dpw_tree.a_lookup, (snode, a))

n_a_children(tree::ActionGradTree, sanode::Int) = tree.dpw_tree.n_a_children[sanode]
has_unique_transitions(tree::ActionGradTree, sanode::Int, spnode::Int) = (sanode, spnode) in tree.dpw_tree.unique_transitions

sminus_labels(tree::ActionGradTree, index::Int) = tree.sminus_labels[index]
sampled_s_sps_labels(tree::ActionGradTree, index::Int) = tree.sampled_s_sps[index]
a_proposal(tree::ActionGradTree, index::Int) = tree.a_proposals[index]
a_init(tree::ActionGradTree, index::Int) = tree.a_init[index]
value(tree::ActionGradTree, index::Int) = value(tree.value_updater, index)

## ActionGradTree setter/mutating methods
add_total_n_visits!(tree::ActionGradTree, index::Int, n::Int) = tree.dpw_tree.total_n[index] += n
add_n_visits!(tree::ActionGradTree, index::Int, n::Int) = tree.dpw_tree.n[index] += n
add_n_a_children!(tree::ActionGradTree, sanode::Int, n::Int) = tree.dpw_tree.n_a_children[sanode] += n

add_transition!(tree::ActionGradTree, sanode::Int, spnode::Int, r::Float64) = push!(tree.dpw_tree.transitions[sanode], (spnode, r))
add_unique_transition!(tree::ActionGradTree, sanode::Int, spnode::Int) = push!(tree.dpw_tree.unique_transitions, (sanode, spnode))

function set_a_label!(tree::ActionGradTree{S,A}, snode::Int, sanode::Int, a::A) where {S,A}
    current_action = a_labels(tree, sanode)
    if has_action(tree, snode, current_action)
        delete!(tree.dpw_tree.a_lookup, (snode, current_action))
        tree.dpw_tree.a_lookup[(snode, a)] = sanode
    end
    tree.dpw_tree.a_labels[sanode] = a
end

function MCTS.insert_state_node!(tree::ActionGradTree, s, maintain_s_lookup=true, sminus=nothing, s_sp=nothing, p_log_likelihood=0.0, q_log_likelihood=0.0, a_proposal=nothing)
    snode = insert_state_node!(tree.dpw_tree, s, maintain_s_lookup)
    if sminus === nothing
        sminus = s
    end
    push!(tree.sminus_labels, sminus)
    push!(tree.sampled_s_sps, s_sp)
    push!(tree.a_proposals, a_proposal)

    insert_state_node!(tree.value_updater, snode, p_log_likelihood, q_log_likelihood)
    return snode
end

function MCTS.insert_action_node!(tree::ActionGradTree, snode::Int, a, n0::Int, q0::Float64, maintain_a_lookup=true)
    sanode = insert_action_node!(tree.dpw_tree, snode, a, n0, q0, maintain_a_lookup)
    push!(tree.a_init, [a])
    insert_action_node!(tree.value_updater, snode, sanode)
    return sanode
end

# Forwarded functions - will act on the dpw_tree field of the ActionGradTree instance
@forward ActionGradTree.dpw_tree Base.isempty, MCTS.best_sanode, MCTS.best_sanode_UCB

struct ActionGradStateNode{S,A} <: AbstractStateNode
    tree::ActionGradTree{S,A}
    index::Int
end

# Base.convert(::Type{DPWStateNode{S,A}}, n::ActionGradStateNode{S,A}) = DPWStateNode{S,A}(n.tree.dpw_tree, n.index)
MCTS.children(n::ActionGradStateNode) = children(n.tree, n.index)
MCTS.n_children(n::ActionGradStateNode) = length(children(n))
MCTS.isroot(n::ActionGradStateNode) = n.index == 1
