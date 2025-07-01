import dask.dataframe as dd
from dask.distributed import Client
from pathlib import Path
import shutil
import logging

# --- Configuração do Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Função de Processamento com Dask ---

def process_large_csv_with_dask(csv_path: str, output_dir: Path):
    """
    Processa um arquivo CSV grande usando Dask, com as configurações finais
    para codificação, delimitador e tipos de dados.
    """
    logging.info(f"--- Iniciando processamento com Dask para o arquivo: {csv_path} ---")
    
    try:
        # 1. Ler o CSV com os parâmetros FINAIS E CORRETOS
        logging.info("Criando o grafo de tarefas do Dask com codificação 'utf-16', separador ',' e tipos de dados manuais.")
        ddf = dd.read_csv(
            csv_path,
            encoding='utf-16',
            sep=',',
            # A correção final sugerida pelo próprio Dask:
            dtype={
                'ocnr_tx_key': 'object',
                'ocnr_tx_key2': 'object'
            },
            blocksize='64MB'
        )

        # 2. Limpeza e verificação dos nomes das colunas
        ddf.columns = ddf.columns.str.strip()
        logging.info(f"Colunas encontradas e limpas: {list(ddf.columns)}")

        # 3. Definir as transformações
        logging.info("Definindo etapas de limpeza e transformação dos dados...")
        
        ddf['ocnr_dt_date'] = dd.to_datetime(ddf['ocnr_dt_date'])
        ddf['ocnr_nm_result'] = dd.to_numeric(ddf['ocnr_nm_result'], errors='coerce')
        ddf = ddf.dropna(subset=['ocnr_nm_result', 'ocnr_dt_date', 'ocnr_tx_namespace'])

        # 4. Executar o processamento e salvar o resultado
        logging.info("Iniciando a computação e escrita dos arquivos Parquet. Este processo pode levar um tempo considerável...")
        
        ddf.to_parquet(
            path=output_dir,
            engine='pyarrow',
            partition_on=['ocnr_tx_namespace']
        )
        
        logging.info("--- SUCESSO! Processamento com Dask concluído. ---")

    except KeyError as e:
        logging.error(f"ERRO DE CHAVE (KeyError): Não foi possível encontrar a coluna {e}. Verifique a lista de 'Colunas encontradas e limpas' no log para confirmar o nome exato.")
    except Exception as e:
        logging.error(f"ERRO CRÍTICO: Uma falha inesperada ocorreu durante o processamento com Dask. Detalhes: {e}")

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    client = Client()
    logging.info(f"Cliente Dask iniciado. Painel de diagnóstico disponível em: {client.dashboard_link}")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    INPUT_CSV_PATH = PROJECT_ROOT / "archive" / "00.csv"
    OUTPUT_PARQUET_DIR = PROJECT_ROOT / "archive" / "partitioned_parquet"

    if OUTPUT_PARQUET_DIR.exists():
        logging.warning(f"Diretório de saída antigo encontrado em '{OUTPUT_PARQUET_DIR}'. Removendo...")
        shutil.rmtree(OUTPUT_PARQUET_DIR)
    
    OUTPUT_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Diretório de saída '{OUTPUT_PARQUET_DIR}' está pronto.")

    process_large_csv_with_dask(str(INPUT_CSV_PATH), OUTPUT_PARQUET_DIR)

    client.close()
    logging.info("Cliente Dask finalizado.")