using Plots

mutations = 1:10
counts = [1114, 13010, 12683, 9759, 7215, 4643, 2783, 1526, 714, 352]

p = plot(bar(mutations, counts),
    xticks=mutations,
    xlabel="number of mutations",
    ylabel="number of variants",
    title="GFP dataset - reported data (wt length = 236)")

savefig(p, "reported_data.png")
