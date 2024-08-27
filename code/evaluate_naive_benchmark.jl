include("utils.jl")
include("evaluation.jl")

Random.seed!(42)

"""
# Arguments
- `df::DataFrame`: Contains sequence variants (in a column named "sequence") and their fitness (in a column named "score")
                   and mutational gap (in a column named "gap").
- `construct_train_set::Function`:
- `select_mutants::Function`:
"""
function evaluate_method(df::DataFrame; n_mutants::Int)
    df_train = CONSTRUCT_TRAIN_SET(df)
    println("Size of training set = $(size(df_train)[1])")
    mutants = SELECT_MUTANTS(df_train; n_mutants=n_mutants)
    evaluate_mutants(mutants, df, df_train)
end

function evaluation_iteration(dataset_name::String; n_mutants::Int=128)
    println("Starting dataset $dataset_name ...")
    df = PREPARE_DATA(dataset_name)
    (fitness, fitness_norm, diversity, novelty, n_replaced_scores, n_mutants_real) = evaluate_method(df; n_mutants)
    println("Dataset $dataset_name finished:")
    println("fitness = $fitness")
    println("fitness_norm = $fitness_norm")
    println("diversity = $diversity")
    println("novelty = $novelty")
    println("n_replaced_scores = $n_replaced_scores")
    return (fitness, fitness_norm, diversity, novelty, n_replaced_scores, n_mutants_real)
end

# ___ Main ___
datasets = ["avGFP", "AAV", "TEM", "E4B", "AMIE", "LGK", "Pab1", "UBE2I"] # medium -> (3775, 2109, 0, 63, 0, 0, 0, 0)
PREPARE_DATA = prepare_data

datasets = ["GB1", "PhoQ", "TrpB"]
PREPARE_DATA = prepare_data_combinatorial

CONSTRUCT_TRAIN_SET = difficulty_filter_hard
CONSTRUCT_TRAIN_SET = (df::DataFrame) -> difficulty_filter(df; percentile_range=(0.2, 0.4), min_gap=0)

SELECT_MUTANTS = common_single_mutants
results = map(dataset_name -> evaluation_iteration(dataset_name), datasets)
