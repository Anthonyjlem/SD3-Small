import pstats

p = pstats.Stats('output.prof')
p.sort_stats('cumulative').print_stats(10) # Sort by cumulative time and print top 10