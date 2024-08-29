using StatsBase
using CSV
using DataFrames
using IterTools

include("naive_benchmark.jl")

AMINO_ACIDS = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

_construct_sequence(variant::Vector{Char}, wt_string::String, mutation_positions::Vector{Int}) = collect(wt_string[1:mutation_positions[1]-1] * variant[1] * wt_string[mutation_positions[1]+1:mutation_positions[2]-1] * variant[2] * wt_string[mutation_positions[2]+1:mutation_positions[3]-1] * variant[3] * wt_string[mutation_positions[3]+1:mutation_positions[4]-1] * variant[4] * wt_string[mutation_positions[4]+1:end])
function complete_sequences!(df::DataFrame, mutation_positions::Vector{Int}, wt_string::String)
    df.sequence = map(variant -> _construct_sequence(collect(variant), wt_string, mutation_positions) |> String, df.sequence)
end

# GB1
data_path = joinpath(@__DIR__, "..", "data", "combinatorial", "GB1")
#wt_string = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
wt_string = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTELEVLFQGPLDPNSMATYEVLCEVARKLGTDDREVVLFLLNVFIPQPTLAQLIGALRALKEEGRLTFPLLAECLFRAGRRDLLRDLLHLDPRFLERHLAGTMSYFSPYQLTVLHVDGELCARDIRSLIFLSKDTIGSRSTPQTFLHWVYCMENLDLLGPTDVDALMSMLRSLSRVDLQRQVQTLMGLHLSGPSHSQHYRHTPLEHHHHHH"
mutation_positions = [39, 40, 41, 54] # ['V', 'D', 'G', 'V']

# PhoQ
data_path = joinpath(@__DIR__, "..", "data", "combinatorial", "PhoQ")
wt_string = "MKKLLRLFFPLSLRVRFLLATAAVVLVLSLAYGMVALIGYSVSFDKTTFRLLRGESNLFYTLAKWENNKLHVELPENIDKQSPTMTLIYDENGQLLWAQRDVPWLMKMIQPDWLKSNGFHEIEADVNDTSLLLSGDHSIQQQLQEVREDDDDAEMTHSVAVNVYPATSRMPKLTIVVVDTIPVELKSSYMVWSWFIYVLSANLLLVIPLLWVAAWWSLRPIEALAKEVRELEEHNRELLNPATTRELTSLVRNLNRLLKSERERYDKYRTTLTDLTHSLKTPLAVLQSTLRSLRSEKMSVSDAEPVMLEQISRISQQIGYYLHRASMRGGTLLSRELHPVAPLLDNLTSALNKVYQRKGVNISLDISPEISFVGEQNDFVEVMGNVLDNACKYCLEFVEISARQTDEHLYIVVEDDGPGIPLSKREVIFDRGQRVDTLRPGQGVGLAVAREITEQYEGKIVAGESMLGGARMEVIFGRQHSAPKDE"
mutation_positions = [284, 285, 288, 289]

# TrpB - E. coli, Tm9D8* variant
data_path = joinpath(@__DIR__, "..", "data", "combinatorial", "TrpB")
wt_string = "MKGYFGPYGGQYVPEILMGALEELEAAYEGIMKDESFWKEFNDLLRDYAGRPTPLYFARRLSEKYGARVYLKREDLLHTGAHKINNAIGQVLLAKLMGKTRIIAETGAGQHGVATATAAALFGMECVIYMGEEDTIRQKLNVERMKLLGAKVVPVKSGSRTLKDAIDEALRDWITNLQTTYYVFGSVVGPHPYPIIVRNFQKVIGEETKKQIPEKEGRLPDYIVACVSGGSNAAGIFYPFIDSGVKLIGVEAGGEGLETGKHAASLLKGKIGYLHGSKTFVLQDDWGQVQVSHSVSAGLDYSGVGPEHAYWRETGKVLYDAVTDEEALDAFIELSRLEGIIPALESSHALAYLKKINIKGKVVVVNLSGRGDKDLESVLNHPYVRERIRLEHHHHHH"
mutation_positions = [183, 184, 227, 228] # ['V', 'F', 'V', 'S']

# Difficulty
difficulty = "medium"
percentile_range = (0.2, 0.4) # hard: (0.0, 0.3), medium: (0.2, 0.4)
min_gap = 0

# Load data
file_path = joinpath(data_path, "ground_truth.csv")
df = CSV.read(file_path, DataFrame)

# Create DataFrame with missing variants
fitness_dict = Dict(df.sequence .=> df.score)
all_variants = map(v -> mapreduce(s -> s, *, v), Iterators.product(ntuple(_ -> AMINO_ACIDS, 4)...) |> collect |> vec)
all_scores = map(variant -> get(fitness_dict, variant, missing), all_variants)
df_all = DataFrame(sequence=all_variants, score=all_scores)
df_missing = filter(row -> row.score |> ismissing, df_all)

# Add extra columns
df.gap = get_mutational_gaps(df)
df_missing.gap = get_mutational_gaps(df_missing)

# Construct full sequences
#complete_sequences!(df, mutation_positions, wt_string)
#complete_sequences!(df_missing, mutation_positions, wt_string)

# Training set
df.index = range(1, size(df)[1])
df_non_zero = filter(row -> row.score != 0.0, df)
train = difficulty_filter(df_non_zero; percentile_range, min_gap)
set = zeros(Bool, size(df)[1])
set[train.index] .= 1
df.set = map(x -> x ? "train" : "test", set)

df_missing.set .= "test"

# Validation set
validation = Vector{Union{Bool,Missing}}(undef, size(df)[1])
map(i -> validation[i] = true, sample(train.index, Int(floor(0.1 * length(train.index))), replace=false))
df.validation = validation

df_missing.validation .= missing

# Sort columns
select!(df, [:sequence, :score, :set, :validation, :gap])
select!(df_missing, [:sequence, :score, :set, :validation, :gap])

# Add missing variants
df = vcat(df, df_missing)

# Rename score
rename!(df, :score => :target)

# Save
CSV.write(joinpath(data_path, "flip_non-zero_" * difficulty * "_gap0_short.csv"), df)
