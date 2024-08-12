include("avg_seq.jl")

Random.seed!(42)

# ___ Load dataset ___
#= dataset_name = "GFP" # Choose from: "GFP", "AAV"
file_path = joinpath(@__DIR__, "..", "data", dataset_name, "ground_truth.csv") =#
dataset_name = "avGFP" # Choose from: "avGFP", "AAV", "TEM", "E4B", "AMIE", "LGK", "Pab1", "UBE2I"
file_path = joinpath(@__DIR__, "..", "data", "preprocessed_data", dataset_name, dataset_name * ".csv")
df = CSV.read(file_path, DataFrame)
rename!(df, :fitness => :score)
df.gap = get_mutational_gaps(df)

# ___ Define "training data" - Choose one ___
# (1) Following https://arxiv.org/pdf/2307.00494
df_train = difficulty_filter_medium(df)
df_train = difficulty_filter_hard(df)

# (2) Following https://arxiv.org/pdf/2405.18986
df_train = difficulty_filter_alt_medium(df)
df_train = difficulty_filter_alt_hard(df)

# ___ Create Mutants - Choose one ___
# (A) Most common mutation at each position (! positions with no mutations produce avg_sequence)
mutants = mutate_at_each_position(df_train; n_mutants=128)

# (B) Most common single mutations
mutants = common_single_mutants(df_train; n_mutants=128)

# (C) Double mutants from pairs of most common mutations
mutants = common_double_mutants(df_train; n_mutants=128)

# ___ Prepare Evaluation ___
mutant_strings = map(i -> String(mutants[i]), eachindex(mutants))
default_score = 0
scores = map(mutant -> filter(row -> row.sequence == mutant, df).score, mutant_strings)
mapreduce(s -> length(s) == 0, +, scores) # How many score will be replaced with default_score?
scores = map(s -> length(s) == 0 ? default_score : mean(s), scores)

# ___ Evaluate ___
median(scores)
normalize_score(median(scores), df)
median(pairwise(hamming, mutants)) # Diversity
median(map(col -> minimum(col), eachcol(pairwise(hamming, df_train.sequence, mutants)))) # Novelty
sum(map(mutant -> mutant in df_train.sequence, mutant_strings))
