using StatsBase
using CSV
using DataFrames

include("naive_benchmark.jl")

data_path = joinpath(@__DIR__, "..", "data", "combinatorial", "GB1")

file_path = joinpath(data_path, "ground_truth.csv")
df = CSV.read(file_path, DataFrame)
df.gap = get_mutational_gaps(df)
df.index = range(1, size(df)[1])

# Training set
train = difficulty_filter(df; percentile_range=(0.0, 0.3), min_gap=0)
set = zeros(Bool, size(df)[1])
set[train.index] .= 1
df.set = map(x -> x ? "train" : "test", set)

# Validation set
validation = Vector{Union{Bool, Missing}}(undef, size(df)[1])
map(i -> validation[i] = true, sample(train.index, Int(floor(0.1 * length(train.index))), replace = false))
df.validation = validation

# Sort & Rename columns
rename!(df, :score => :target)
select!(df, [:sequence, :target, :set, :validation, :gap])

# Save
CSV.write(joinpath(data_path, "flip_hard_gap0.csv"), df)
