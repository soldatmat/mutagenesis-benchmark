using CSV
using DataFrames
using StatsBase
using Random

Random.seed!(42)

# ___ Helper functions ___
normalize_score(score::Real) = (score - minimum(df.score)) / (maximum(df.score) - minimum(df.score))
denormalize_score(score::Real) = score * (maximum(df.score) - minimum(df.score)) + minimum(df.score)

# ___ Choose dataset ___ AAV:
file_path = joinpath(@__DIR__, "..", "data", "AAV", "ground_truth.csv")
file_path = joinpath(@__DIR__, "..", "data", "preprocessed_data", "AAV", "AAV.csv")
df = CSV.read(file_path, DataFrame)
rename!(df, :Combo => :sequence)
rename!(df, :fitness => :score)
select!(df, :sequence, :score)

CSV.write(joinpath(@__DIR__, "..", "data", "combinatorial", "PhoQ", "ground_truth.csv"), df)

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
