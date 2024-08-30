using CSV
using DataFrames
using StatsBase
using Random

include("naive_benchmark.jl")

Random.seed!(42)

# ___ Choose dataset ___ AAV:
file_path = joinpath(@__DIR__, "..", "data", "AAV", "ground_truth.csv")
file_path = joinpath(@__DIR__, "..", "data", "preprocessed_data", "AAV", "AAV.csv")
file_path = joinpath(@__DIR__, "..", "data", "combinatorial", "TrpB", "ground_truth_non-zero.csv")
df = CSV.read(file_path, DataFrame)
rename!(df, :Combo => :sequence)
rename!(df, :fitness => :score)
select!(df, :sequence, :score)

df = filter(row -> row.score != 0.0, df)

CSV.write(joinpath(@__DIR__, "..", "data", "combinatorial", "TrpB", "ground_truth_non-zero.csv"), df)

# ___ Get avg_sequence ___
avg_sequence = map(i -> mode(map(s -> s[i], df.sequence)), 1:length(df.sequence[1]))
avg_sequence_string = String(avg_sequence)

# ___ Score avg_sequence ___
filtered_df = filter(row -> row.sequence == avg_sequence_string, df)
avg_score = sum(filtered_df.score) / (size(filtered_df)[1])
median(filtered_df.score)
median(filtered_df.score) |> normalize_score

# ___ Explore dataset ___
minimum(df.score)
maximum(df.score)
sequence_length = length(df[1, :].sequence)
filtered_df = filter(row -> length(row.sequence) == sequence_length, df)

# ___ Explore task difficulty ___
df.gap = get_mutational_gaps(df)
dff = difficulty_filter(df; percentile_range=(0.2, 0.4), min_gap=0) # medium
dff = difficulty_filter(df; percentile_range=(0.0, 0.3), min_gap=0) # hard
dff = difficulty_filter(df; percentile_range=(0.0, 0.63), min_gap=0) # custom
sum(dff.score .!= 0.0)
