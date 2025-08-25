# check_duckdb.py - Teste 2
import duckdb

# Caminho com um padrão GLOB para encontrar todos os .parquet recursivamente
parquet_glob_path = "/home/lucas/UNIFEI/Vertis/vertis_research_agent/archive/partitioned_parquet/**/*.parquet"

print(f"Tentando escanear usando o padrão GLOB: {parquet_glob_path}")

try:
    con = duckdb.connect()
    # Passando o padrão glob para o parquet_scan
    result_df = con.execute(f"SELECT COUNT(*) FROM parquet_scan('{parquet_glob_path}')").fetchdf()

    print("\n--- SUCESSO! ---")
    print("Consegui ler os arquivos usando o padrão GLOB.")
    print("Contagem total de linhas encontradas:", result_df.iloc[0,0])

except Exception as e:
    print("\n--- ERRO! ---")
    print("Falhei em ler os arquivos mesmo com o padrão GLOB:")
    print(e)