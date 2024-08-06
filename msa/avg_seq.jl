using CSV
using DataFrames
using StatsBase
using Random
using Distances
using Combinatorics

Random.seed!(42)

# (some?) avGFP wild-type
wt_sequence = "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK" # GFP wild-type
filtered_df = filter(row -> row.sequence == wt_sequence, df)
avg_score = sum(filtered_df.score) / (size(filtered_df)[1])



# ___ Choose dataset ___
file_path = joinpath(@__DIR__, "..", "data", "AAV", "ground_truth.csv")
df = CSV.read(file_path, DataFrame)

# ___ Define "training data" ___
df_train = copy(df)

df_train = filter(row -> row.score <= 1.28, df)#[1:2, :]

df_sorted = sort(df, :score)
df_train = df_sorted[1:256+128,:]

percentile = [0.2, 0.4] # medium
percentile = [0.1, 0.3] # hard
n_sequences = size(df_sorted)[1]
df_sorted = df_sorted[Int(ceil(n_sequences*percentile[1])):Int(floor(n_sequences*percentile[2])), :]
n_train = 0 # n_train highest score variants from the percentile
n_oracle_calls = 256 + 128 # n_oracle_calls random variants from the percentile
df_train = df_sorted[size(df_sorted)[1]-n_train+1:size(df_sorted)[1],:]
df_oracle_calls = df_sorted[1:size(df_sorted)[1]-n_train,:]
df_oracle_calls = df_oracle_calls[shuffle(1:nrow(df_oracle_calls))[1:n_oracle_calls], :]
df_train = vcat(df_train, df_oracle_calls)


# ___ Get avg_sequence ___
avg_sequence = map(i -> mode(map(s -> s[i], df_train.sequence)), 1:length(df_train.sequence[1]))
avg_sequence = String(avg_sequence)

# Obtain gt score of avg_sequence
filtered_df = filter(row -> row.sequence == avg_sequence, df)
avg_score = sum(filtered_df.score)/(size(filtered_df)[1])
median(filtered_df.score)
(median(filtered_df.score) - minimum(df.score)) / maximum(df.score)



# ___ Get symbol distribution for each position ___
dists = map(i -> map(seq -> seq[i], df_train.sequence), 1:length(df_train[1, :].sequence))
dists = map(position -> sort(collect(countmap(position)), by=x -> x[2], rev=true), dists)
sum(map(position -> length(position), dists) .== 1) # How many positions have no mutations?

# ___ Create Mutants ___
avg_sequence = map(position -> position[1][1], dists)

# A) Most common mutation at each position (! positions with no mutations produce avg_sequence)
mutants = map(pos -> copy(avg_sequence), eachindex(dists))
map(pos -> mutants[pos][pos] = dists[pos][length(dists[pos]) > 1 ? 2 : 1][1], eachindex(dists))

# B) Most common single mutations
# TODO replace modus symbol (maximum()) with wt_sequence symbol
dist = reduce(vcat, map(pos -> map(pair -> (pair[1], pair[2] == maximum(map(symbol -> symbol[2], dists[pos])) ? -1 : pair[2], pos), dists[pos]), eachindex(dists)))
sort!(dist, by=x -> x[2], rev=true)
n_mutants = 128
mutants = map(pos -> copy(avg_sequence), 1:n_mutants)
map(m -> mutants[m][dist[m][3]] = dist[m][1], eachindex(mutants))

# C) Double mutants from pairs of most common mutations
dist = reduce(vcat, map(pos -> map(pair -> (pair[1], pair[2] == maximum(map(symbol -> symbol[2], dists[pos])) ? -1 : pair[2], pos), dists[pos]), eachindex(dists)))
sort!(dist, by=x -> x[2], rev=true)
mutation_pairs = collect(combinations(dist, 2))
mutation_pairs = mutation_pairs[map(pair -> pair[1][3] != pair[2][3], mutation_pairs)]
n_mutants = 128
mutants = map(pos -> copy(avg_sequence), 1:n_mutants)
map(m -> map(i -> mutants[m][mutation_pairs[m][i][3]] = mutation_pairs[m][i][1], 1:2), eachindex(mutants))

# Finish creating mutants
mutant_strings = map(i -> String(mutants[i]), eachindex(mutants))

default_score = 0 # TODO ? should be predicted
scores = map(mutant -> filter(row -> row.sequence == mutant, df).score, mutant_strings)
mapreduce(s -> length(s) == 0, +, scores) # How many score will be replaced with the default_score.
scores = map(s -> length(s) == 0 ? default_score : mean(s), scores)
median(scores)
(median(scores) - minimum(df.score)) / maximum(df.score)
median(pairwise(hamming, mutants))
mean(pairwise(hamming, mutants))



# ___ Explore dataset ___
minimum(df.score)
maximum(df.score)
minimum(df_train.score)
maximum(df_train.score)
sequence_length = length(df[1,:].sequence)
filtered_df = filter(row -> length(row.sequence) == sequence_length, df)

