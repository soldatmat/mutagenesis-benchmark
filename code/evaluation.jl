include("naive_benchmark.jl")

function prepare_data(dataset_name::String)
    file_path = joinpath(@__DIR__, "..", "data", "preprocessed_data", dataset_name, dataset_name * ".csv")
    df = CSV.read(file_path, DataFrame)
    rename!(df, :fitness => :score)
    df.gap = get_mutational_gaps(df)
    return df
end

function prepare_data_combinatorial(dataset_name::String; file_name::String="ground_truth.csv")
    file_path = joinpath(@__DIR__, "..", "data", "combinatorial", dataset_name, file_name)
    df = CSV.read(file_path, DataFrame)
    df.gap = get_mutational_gaps(df)
    return df
end

function evaluate_mutants(mutants::Vector{Vector{Char}}, df::DataFrame, df_train::DataFrame)
    # ___ Prepare Evaluation ___
    mutant_strings = map(i -> String(mutants[i]), eachindex(mutants))
    default_score = denormalize_score(0.0, df) # corresponds to minimum(df.score)
    scores = map(mutant -> filter(row -> row.sequence == mutant, df).score, mutant_strings)
    n_replaced_scores = mapreduce(s -> length(s) == 0, +, scores) # How many score will be replaced with default_score?
    scores = map(s -> length(s) == 0 ? default_score : mean(s), scores)
    n_replaced_scores = n_replaced_scores + mapreduce(s -> ismissing(s), +, scores) # How many score will be replaced with default_score?
    map(i -> scores[i] = ismissing(scores[i]) ? default_score : scores[i], eachindex(scores))

    # ___ Evaluate ___
    fitness = median(scores)
    fitness_norm = normalize_score(median(scores), df)
    diversity = median(pairwise(hamming, mutants))
    novelty = median(map(col -> minimum(col), eachcol(pairwise(hamming, df_train.sequence, mutants))))

    return (fitness, fitness_norm, diversity, novelty, n_replaced_scores, length(mutants))
end
