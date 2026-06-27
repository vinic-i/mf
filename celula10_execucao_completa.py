# ============================================================
# CÉLULA 10 — EXECUÇÃO FINAL COMPLETA (VERSÃO CORRIGIDA)
# ============================================================
# ✏️ Substitua a CÉLULA 10 original do notebook por este código completo
# 
# O QUE ESTE CÓDIGO FAZ:
#   ✓ Carrega benchmarks CDI (sintético) e IBOV (Yahoo Finance)
#   ✓ Usa grade expandida (980 combinações) para otimização
#   ✓ Implementa time stop para aumentar frequência de trades
#   ✓ Aplica stop loss percentual para proteção de capital
#   ✓ Compara estratégia com: buy-and-hold do ativo, CDI, IBOV
#   ✓ Salva todos os gráficos em alta resolução (300 DPI)
#   ✓ Exporta resumo comparativo em CSV
#   ✓ Gera tabela formatada pronta para copiar no relatório

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# --- Cria pasta de saída para os gráficos ---
os.makedirs('graficos_output', exist_ok=True)

# ============================================================
# PARTE 1: CARREGAMENTO DOS BENCHMARKS
# ============================================================
print("=" * 60)
print("CARREGANDO BENCHMARKS...")
print("=" * 60)

# CDI sintético (média ~11.5% a.a. para 2020-2026)
# Você pode ajustar a taxa se tiver dados reais do Banco Central
cdi_data = carregar_cdi_sintetico(Data_Inicial, Data_Final, taxa_anual_cdi=0.115)
cdi_data['Date'] = pd.to_datetime(cdi_data['Date'])
print(f"  CDI carregado: {len(cdi_data)} dias de pregão")

# IBOV (índice Bovespa)
ibov_data = carregar_ibov(Data_Inicial, Data_Final)
if ibov_data is not None:
    ibov_data['Date'] = pd.to_datetime(ibov_data['Date'])
    ibov_returns = ibov_data.set_index('Date')['retorno_ibov']
    print(f"  IBOV carregado: {len(ibov_data)} dias de pregão")
else:
    ibov_returns = None
    print("  [AVISO] IBOV não disponível, usando apenas CDI como benchmark")

# ============================================================
# PARTE 2: LOOP PRINCIPAL — WALK-FORWARD POR ATIVO
# ============================================================
resumo_geral = []      # métricas finais de todos os ativos
benchmarks_lista = []  # métricas de benchmark para tabela final

for ticker in TICKERS:

    print('\n' + '=' * 60)
    print(f'  ATIVO: {ticker}')
    print('=' * 60)

    # --- Baixa dados do ativo ---
    try:
        data = yf.download(ticker, start=Data_Inicial, end=Data_Final, 
                           auto_adjust=True, progress=False)
        data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        data['Year'] = data['Date'].dt.year
        data['Date'] = pd.to_datetime(data['Date'])
        years = sorted(data['Year'].unique())
    except Exception as e:
        print(f'  [ERRO] Falha ao baixar {ticker}: {e}')
        continue

    if len(data) < 500:
        print(f'  [AVISO] Dados insuficientes. Pulando.')
        continue

    # --- Buy-and-Hold do próprio ativo (para comparação) ---
    retorno_bh_ativo = calcular_retorno_buy_and_hold(data)
    print(f'  Buy-and-Hold do ativo: {retorno_bh_ativo:.2%}')

    # --- Walk-Forward com grade expandida ---
    try:
        records = rodar_walk_forward(
            data           = data,
            years          = years,
            train_years    = train_years,
            test_years     = test_years,
            gerar_grade    = gerar_grade_expandida,      # ← GRADE EXPANDIDA
            calcular_sinal = calcular_sinal_com_time_stop, # ← COM TIME STOP
            aplicar_stop   = aplicar_stop_modificado,     # ← COM STOP PERCENTUAL
            custo_trade    = CUSTO_TRADE
        )
    except Exception as e:
        print(f'  [ERRO] Walk-forward: {e}')
        continue

    if not records:
        print(f'  [AVISO] Nenhuma janela OOS gerada. Pulando.')
        continue

    # --- Calcula métricas da estratégia ---
    m = calcular_metricas(records)

    # --- Calcula benchmarks ---
    strategy_returns = m['todos_diarios']

    # CDI: alinha datas
    cdi_rets = cdi_data.set_index('Date')['retorno_cdi'].reindex(
        strategy_returns.index).fillna(0)

    # IBOV: alinha datas
    if ibov_returns is not None:
        ibov_rets = ibov_returns.reindex(
            strategy_returns.index).fillna(0)
    else:
        ibov_rets = pd.Series(0, index=strategy_returns.index)

    # Métricas comparativas
    bench_metrics = calcular_metricas_benchmark(
        strategy_returns, cdi_rets, ibov_rets, 
        periodo_anos=m['n_anos']
    )
    benchmarks_lista.append(bench_metrics)

    # --- Testes estatísticos de robustez ---
    print(f'  Rodando testes estatísticos...')
    res_t    = teste_t_student(m)
    res_boot = bootstrap_sharpe(m)
    res_mc   = monte_carlo(m)
    n_testes = len(gerar_grade_expandida())  # ← 980 combinações
    res_dsr  = deflated_sharpe(m, n_testes=n_testes)
    res_wrc  = whites_reality_check(
        data, years, train_years, test_years,
        gerar_grade_expandida, calcular_sinal_com_time_stop, CUSTO_TRADE
    )

    # --- Imprime resultados no console ---
    imprimir_metricas(m)
    imprimir_testes(res_t, res_boot, res_mc, res_dsr, res_wrc)

    print('\n--- COMPARAÇÃO COM BENCHMARKS ---')
    print(f"  Retorno Estratégia OOS: {bench_metrics['retorno_estrategia']:.2%}")
    print(f"  Retorno Buy-and-Hold  : {retorno_bh_ativo:.2%}")
    print(f"  Retorno CDI           : {bench_metrics['retorno_cdi']:.2%}")
    print(f"  Retorno IBOV          : {bench_metrics['retorno_ibov']:.2%}")
    print(f"  Excesso sobre CDI     : {bench_metrics['excesso_sobre_cdi']:.2%}")
    print(f"  Excesso sobre IBOV    : {bench_metrics['excesso_sobre_ibov']:.2%}")
    print(f"  Excesso sobre B&H     : {bench_metrics['retorno_estrategia'] - retorno_bh_ativo:.2%}")
    print(f"  CAGR Estratégia       : {bench_metrics['cagr_estrategia']:.2%}")
    print(f"  CAGR CDI              : {bench_metrics['cagr_cdi']:.2%}")
    print(f"  CAGR IBOV             : {bench_metrics['cagr_ibov']:.2%}")
    print(f"  Sharpe Excesso/CDI    : {bench_metrics['sharpe_excesso_cdi']:.3f}")

    # --- GERA E SALVA GRÁFICOS EM ALTA RESOLUÇÃO ---
    titulo_base = f'{NOME_DO_GRUPO} — {ticker} — {DESCRICAO}'

    # Gráficos dos testes estatísticos (6 gráficos: t-Student, Bootstrap, Monte Carlo, etc.)
    plotar_testes(res_t, res_boot, res_mc, res_dsr, res_wrc, titulo=titulo_base)
    plt.savefig(f'graficos_output/{ticker}_testes_estatisticos.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  [SALVO] graficos_output/{ticker}_testes_estatisticos.png")

    # Gráficos do walk-forward (9 gráficos: curva de capital, drawdown, parâmetros, etc.)
    plotar_resultados(m, titulo=titulo_base)
    plt.savefig(f'graficos_output/{ticker}_walkforward_OOS.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  [SALVO] graficos_output/{ticker}_walkforward_OOS.png")

    # --- Armazena resumo para comparação final ---
    resumo_geral.append({
        'Ativo': ticker,
        'Retorno_Estrategia': f"{bench_metrics['retorno_estrategia']:.2%}",
        'Retorno_BH': f"{retorno_bh_ativo:.2%}",
        'Retorno_CDI': f"{bench_metrics['retorno_cdi']:.2%}",
        'Retorno_IBOV': f"{bench_metrics['retorno_ibov']:.2%}",
        'Excesso_CDI': f"{bench_metrics['excesso_sobre_cdi']:.2%}",
        'Excesso_IBOV': f"{bench_metrics['excesso_sobre_ibov']:.2%}",
        'Excesso_BH': f"{bench_metrics['retorno_estrategia'] - retorno_bh_ativo:.2%}",
        'CAGR_Estrategia': f"{bench_metrics['cagr_estrategia']:.2%}",
        'Sharpe': f"{m['sharpe']:.3f}",
        'Sharpe_Excesso_CDI': f"{bench_metrics['sharpe_excesso_cdi']:.3f}",
        'Max_Drawdown': f"{m['max_drawdown']:.2%}",
        'Win_Rate': f"{m['win_rate']:.2%}" if m['win_rate'] else 'N/A',
        'N_Trades': m['num_trades'],
        'Exposicao_Mercado': f"{m['exposicao']:.2%}" if m['exposicao'] else 'N/A',
        'p_valor_ttest': f"{res_t['p_valor']:.4f}",
        'Significativo': 'Sim' if res_t['significativo'] else 'Não',
        'DSR_Aprovado': 'Sim' if res_dsr['aprovado'] else 'Não',
        'White_RC_Aprovado': 'Sim' if res_wrc['aprovado'] else 'Não',
    })

# ============================================================
# PARTE 3: RESUMO COMPARATIVO FINAL
# ============================================================
print('\n' + '=' * 80)
print('  RESUMO COMPARATIVO — SETUP 7 COM BENCHMARKS')
print('  Walk-Forward OOS | CDI | IBOV | Buy-and-Hold')
print('=' * 80)

df_resumo = pd.DataFrame(resumo_geral)
print(df_resumo.to_string(index=False))

# Salva em CSV para fácil importação no relatório
df_resumo.to_csv('graficos_output/resumo_comparativo.csv', 
                 index=False, encoding='utf-8')
print('\n[SALVO] graficos_output/resumo_comparativo.csv')

# ============================================================
# PARTE 4: TABELA FORMATADA PARA COPIAR NO RELATÓRIO
# ============================================================
print('\n' + '=' * 80)
print('  TABELA FORMATADA PARA O RELATÓRIO')
print('  Copie e cole no Word/LaTeX/Markdown')
print('=' * 80)
print('\n')

print('-' * 110)
print(f"{'Ativo':<10} {'Ret.Estr':<10} {'Ret.BH':<10} {'Ret.CDI':<10} {'Ret.IBOV':<10} {'Excess.CDI':<12} {'Excess.IBOV':<12} {'Sharpe':<8} {'Trades':<7}")
print('-' * 110)
for _, row in df_resumo.iterrows():
    print(f"{row['Ativo']:<10} {row['Retorno_Estrategia']:<10} {row['Retorno_BH']:<10} "
          f"{row['Retorno_CDI']:<10} {row['Retorno_IBOV']:<10} {row['Excesso_CDI']:<12} "
          f"{row['Excesso_IBOV']:<12} {row['Sharpe']:<8} {row['N_Trades']:<7}")
print('-' * 110)

print('\n')
print('=' * 80)
print('  ARQUIVOS GERADOS NA PASTA graficos_output/')
print('=' * 80)
print('  Para cada ativo:')
print('    - {TICKER}_testes_estatisticos.png  (6 gráficos de robustez)')
print('    - {TICKER}_walkforward_OOS.png      (9 gráficos de desempenho)')
print('  Geral:')
print('    - resumo_comparativo.csv            (tabela para Excel/Word)')
print('=' * 80)
