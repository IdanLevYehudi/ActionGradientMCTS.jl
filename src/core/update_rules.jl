struct ActionUpdateAllTreeAfterSimulate <: AbstractActionUpdateRule end
struct NoActionUpdate <: AbstractActionUpdateRule end
struct ActionUpdateMinVisitations <: AbstractActionUpdateRule
    min_visits::Int
end

struct ActionUpdateMinEveryKVisits <: AbstractActionUpdateRule
    min_visits::Int
    k::Int
end

struct ActionUpdateMinEveryKMinADist <: AbstractActionUpdateRule
    min_visits::Int
    k::Int
    min_a_dist::Float64
end

struct ActionUpdateMinEveryKMaxADist <: AbstractActionUpdateRule
    min_visits::Int
    k::Int
    max_a_dist::Float64
end

struct ActionUpdateMinChildrenEveryKMinADist <: AbstractActionUpdateRule
    min_children::Int
    k::Int
    min_a_dist::Float64
end

struct ActionUpdateMinChildrenEveryKMaxADist <: AbstractActionUpdateRule
    min_children::Int
    k::Int
    max_a_dist::Float64
end

const ActionUpdateMinVisits = Union{ActionUpdateMinVisitations,ActionUpdateMinEveryKVisits,ActionUpdateMinEveryKMinADist,ActionUpdateMinEveryKMaxADist}
const ActionUpdateEveryK = Union{ActionUpdateMinEveryKVisits,ActionUpdateMinEveryKMinADist,ActionUpdateMinEveryKMaxADist,ActionUpdateMinChildrenEveryKMinADist,ActionUpdateMinChildrenEveryKMaxADist}
const ActionUpdateMinDist = Union{ActionUpdateMinEveryKMinADist,ActionUpdateMinChildrenEveryKMinADist}
const ActionUpdateMaxDist = Union{ActionUpdateMinEveryKMaxADist,ActionUpdateMinChildrenEveryKMaxADist}
const ActionUpdateMinChildren = Union{ActionUpdateMinChildrenEveryKMinADist,ActionUpdateMinChildrenEveryKMaxADist}
