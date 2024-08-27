using CSV
using DataFrames

include("evaluation.jl")


datasets = ["GB1", "PhoQ", "TrpB"]
splits = ["medium_gap0", "hard_gap0"]

dataset = datasets[1]
split = splits[2]
model = "cnn"

n_mutants = 128

# Load data
predictions_path = joinpath(@__DIR__, "..", "results", "flip", dataset, model, split, "evaluate", "predictions.csv")
predictions = CSV.read(predictions_path, DataFrame)
data_path = joinpath(@__DIR__, "..", "data", "combinatorial", dataset, "flip_"*split*".csv")
df = CSV.read(data_path, DataFrame)

# Select mutants
sort!(predictions, [:prediction], rev=true)
mutants = map(mutant -> collect(mutant), df[1:n_mutants,:].sequence)

# Re-create df_train
rename!(df, :target => :score)
df_train = filter(row -> row.set == "train", df) # TODO ? exclude validation data

fitness, fitness_norm, diversity, novelty, n_replaced_scores, n_mutants_check = evaluate_mutants(mutants, df, df_train)
