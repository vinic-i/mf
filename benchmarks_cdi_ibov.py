# ================================================================
# FUNÇÕES DE BENCHMARK: CDI + IBOV
# ================================================================
# Cole isto no início do notebook (após as importações),
# ou em uma célula nova antes da Célula 10 de execução.

def carregar_cdi_sintetico(data_inicial, data_final, taxa_anual_cdi=None):
    """
    Gera uma série sintética de retornos diários do CDI.

    Se taxa_anual_cdi for None, usa a média histórica do período
    (aproximadamente 10-12% a.a. para 2020-2026).

    Retorna um DataFrame com colunas 'Date' e 'retorno_cdi'.
    """
    import pandas as pd

    # Período médio do CDI 2020-2026: ~11.5% a.a.
    if taxa_anual_cdi is None:
        # Média aproximada do CDI no período 2020-2026
        # 2020: 2.0%, 2021: 4.4%, 2022: 13.75%, 2023: 11.75%, 2024: 10.5%, 2025: 11.0%
        taxa_anual_cdi = 0.115  # 11.5% média aproximada

    # Gera datas de pregão (dias úteis no Brasil)
    datas = pd.bdate_range(start=data_inicial, end=data_final)

    # Taxa diária equivalente (252 pregões/ano)
    taxa_diaria = (1 + taxa_anual_cdi) ** (1/252) - 1

    df_cdi = pd.DataFrame({
        'Date': datas,
        'retorno_cdi': taxa_diaria
    })

    return df_cdi


def carregar_ibov(data_inicial, data_final):
    """
    Baixa dados do IBOV (^BVSP) do Yahoo Finance.
    Retorna DataFrame com 'Date', 'Close', 'retorno_ibov'.

    NOTA: No Colab, certifique-se de que yfinance está instalado:
    !pip install yfinance -q
    """
    import yfinance as yf
    try:
        ibov = yf.download('^BVSP', start=data_inicial, end=data_final, 
                           auto_adjust=True, progress=False)
        ibov.columns = ibov.columns.get_level_values(0)
        ibov = ibov.reset_index()
        ibov['retorno_ibov'] = ibov['Close'].pct_change()
        return ibov[['Date', 'Close', 'retorno_ibov']].copy()
    except Exception as e:
        print(f"[ERRO] Falha ao baixar IBOV: {e}")
        return None


def calcular_retorno_buy_and_hold(data, coluna_close='Close'):
    """
    Calcula retorno acumulado do buy-and-hold para um ativo.
    """
    preco_inicial = data[coluna_close].iloc[0]
    preco_final = data[coluna_close].iloc[-1]
    return (preco_final / preco_inicial) - 1


def calcular_metricas_benchmark(strategy_returns, cdi_returns, ibov_returns, 
                               periodo_anos):
    """
    Compara a estratégia com CDI e IBOV.

    Parâmetros:
    -----------
    strategy_returns : pd.Series — retornos diários da estratégia (já com custos)
    cdi_returns : pd.Series — retornos diários do CDI
    ibov_returns : pd.Series — retornos diários do IBOV
    periodo_anos : int — número de anos do período analisado

    Retorna:
    --------
    dict com métricas comparativas
    """
    import pandas as pd
    import numpy as np

    # Alinha os índices (datas)
    df_comp = pd.DataFrame({
        'estrategia': strategy_returns,
        'cdi': cdi_returns.reindex(strategy_returns.index).fillna(0),
        'ibov': ibov_returns.reindex(strategy_returns.index).fillna(0)
    }).dropna()

    # Retornos acumulados
    ret_estrategia = (1 + df_comp['estrategia']).prod() - 1
    ret_cdi = (1 + df_comp['cdi']).prod() - 1
    ret_ibov = (1 + df_comp['ibov']).prod() - 1

    # CAGR
    cagr_estrategia = (1 + ret_estrategia) ** (1/periodo_anos) - 1
    cagr_cdi = (1 + ret_cdi) ** (1/periodo_anos) - 1
    cagr_ibov = (1 + ret_ibov) ** (1/periodo_anos) - 1

    # Excesso de retorno
    excesso_cdi = ret_estrategia - ret_cdi
    excesso_ibov = ret_estrategia - ret_ibov

    # Sharpe (usando CDI como taxa livre de risco aproximada)
    ret_excesso = df_comp['estrategia'] - df_comp['cdi']
    sharpe_excesso = ret_excesso.mean() / ret_excesso.std() * np.sqrt(252) if ret_excesso.std() != 0 else np.nan

    return {
        'retorno_estrategia': ret_estrategia,
        'retorno_cdi': ret_cdi,
        'retorno_ibov': ret_ibov,
        'cagr_estrategia': cagr_estrategia,
        'cagr_cdi': cagr_cdi,
        'cagr_ibov': cagr_ibov,
        'excesso_sobre_cdi': excesso_cdi,
        'excesso_sobre_ibov': excesso_ibov,
        'sharpe_excesso_cdi': sharpe_excesso,
        'volatilidade_estrategia': df_comp['estrategia'].std() * np.sqrt(252),
        'volatilidade_ibov': df_comp['ibov'].std() * np.sqrt(252),
    }


print("=" * 60)
print("FUNÇÕES DE BENCHMARK CARREGADAS COM SUCESSO")
print("=" * 60)
print("
Funções disponíveis:")
print("  - carregar_cdi_sintetico(data_inicial, data_final, taxa_anual_cdi)")
print("  - carregar_ibov(data_inicial, data_final)")
print("  - calcular_retorno_buy_and_hold(data)")
print("  - calcular_metricas_benchmark(strategy_returns, cdi_returns, ibov_returns, periodo_anos)")
