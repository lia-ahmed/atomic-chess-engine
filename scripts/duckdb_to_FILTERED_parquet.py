import duckdb

con = duckdb.connect("lichess_atomic.duckdb")

# Select games with average elo > 1600
query = """
SELECT *
FROM games
WHERE (white_elo + black_elo) / 2 > 1600
"""

# Save to parquet
con.execute(f"COPY ({query}) TO 'filtered_games.parquet' (FORMAT 'parquet')")
