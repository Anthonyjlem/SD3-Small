import pstats

if __name__ == "__main__":
    p = pstats.Stats('output.prof')
    p.sort_stats('cumulative').print_stats()  # Sort by cumulative time and print top 10