using CSV
using DataFrames

include("evaluation.jl")


datasets = ["GB1", "PhoQ", "TrpB"]
splits = ["non-zero_medium_gap0", "non-zero_hard_gap0"]

model = "cnn"
n_mutants = 128

for dataset in datasets
    for split in splits
        # Load data
        predictions_path = joinpath(@__DIR__, "..", "results", "flip", dataset, model, split, "evaluate", "predictions.csv")
        #predictions_path = joinpath(@__DIR__, "..", "results", "flip", dataset, model, split, "predictions.csv")        
        predictions = CSV.read(predictions_path, DataFrame)
        data_path = joinpath(@__DIR__, "..", "data", "combinatorial", dataset, "flip_" * split * ".csv")
        df = CSV.read(data_path, DataFrame)

        # Select mutants
        sort!(predictions, [:prediction], rev=true)
        mutants = map(mutant -> collect(mutant), predictions[1:n_mutants, :].sequence)

        # Re-create df_train
        rename!(df, :target => :score)
        df_train = filter(row -> row.set == "train", df) # TODO ? exclude validation data

        fitness, fitness_norm, diversity, novelty, n_replaced_scores, n_mutants_check = evaluate_mutants(mutants, df, df_train)

        println("dataset = $dataset")
        println("split = $split")
        println("$fitness, $fitness_norm, $diversity, $novelty, $n_replaced_scores, $n_mutants_check\n")
    end
end
