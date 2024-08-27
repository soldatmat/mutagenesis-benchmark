using StatsBase
using CSV
using DataFrames

include("naive_benchmark.jl")

_construct_sequence(variant::Vector{Char}, wt_string::String, mutation_positions::Vector{Int}) = collect(wt_string[1:mutation_positions[1]-1] * variant[1] * wt_string[mutation_positions[1]+1:mutation_positions[2]-1] * variant[2] * wt_string[mutation_positions[2]+1:mutation_positions[3]-1] * variant[3] * wt_string[mutation_positions[3]+1:mutation_positions[4]-1] * variant[4] * wt_string[mutation_positions[4]+1:end])

# GB1
data_path = joinpath(@__DIR__, "..", "data", "combinatorial", "GB1")
wt_string = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
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
difficulty = "hard"
percentile_range = (0.0, 0.3) # hard: (0.0, 0.3), medium: (0.2, 0.4)
min_gap = 0

# Load data
file_path = joinpath(data_path, "ground_truth.csv")
df = CSV.read(file_path, DataFrame)
df.gap = get_mutational_gaps(df)
df.index = range(1, size(df)[1])

# Construct full sequences
df.sequence = map(variant -> _construct_sequence(collect(variant), wt_string, mutation_positions) |> String, df.sequence)

# Training set
train = difficulty_filter(df; percentile_range, min_gap)
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
CSV.write(joinpath(data_path, "flip_"*difficulty*"_gap0.csv"), df)
