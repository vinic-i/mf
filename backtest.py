# =============================================================================
# BACKTEST DE ESTRATÉGIA TÉCNICA — AÇÕES DA B3
# =============================================================================
# Estratégia: Compra quando o preço de fechamento rompe abaixo da mínima
# dos últimos 7 candles E abaixo da banda inferior de Bollinger.
# Venda quando o preço fecha acima da máxima dos últimos 9 candles.
#
# Requisitos atendidos:
#   - Timeframe diário (D1)
#   - Mercado: ações B3 com liquidez adequada
#   - Ordens executadas no candle SEGUINTE ao sinal (shift de 1 período)
#   - Custos operacionais (0,30%) e slippage (0,10%) descontados
#   - Dados ajustados por dividendos e desdobramentos (auto_adjust=True)
#   - Bibliotecas: pandas, numpy, ta
#   - Métricas: retorno acumulado, drawdown máximo, payoff, profit factor,
#               Sharpe ratio e percentual de acerto
# =============================================================================

import os                                      # manipulação de diretórios
import yfinance as yf                          # download de dados históricos
import pandas as pd                            # manipulação de séries temporais
import numpy as np                             # cálculos numéricos
import matplotlib                              # backend sem janela gráfica
matplotlib.use('Agg')
import matplotlib.pyplot as plt                # geração dos gráficos
import matplotlib.backends.backend_pdf as pdf_backend  # exportação para PDF
import ta                                      # indicadores técnicos

# =============================================================================
# PARÂMETROS GLOBAIS
# =============================================================================

# Lista de ativos da B3 selecionados por liquidez
tickers = ['PETR4.SA', 'ITUB4.SA', 'VALE3.SA', 'ABEV3.SA', 'WEGE3.SA']

# Janela temporal do backtest
data_inicial = '2020-01-01'
data_final   = '2026-01-01'

# Timeframe utilizado: diário (D1)
intervalo = '1d'

# Custo por operação: 0,30% — cobre corretagem, emolumentos e taxas B3
custo_op = 0.003

# Slippage estimado: 0,10% — diferença entre o preço esperado e o executado
# Ocorre por falta de liquidez pontual ou delay na execução
slippage = 0.001


# =============================================================================
# FUNÇÃO PRINCIPAL DE BACKTEST
# =============================================================================

def run_backtest(ticker):
    """
    Executa o backtest completo para um ativo e retorna o DataFrame
    com colunas de indicadores, sinais e retornos.
    """

    print(f'\nBaixando dados de {ticker}...')

    # -------------------------------------------------------------------------
    # 1. DOWNLOAD DOS DADOS
    # -------------------------------------------------------------------------
    # yfinance busca dados OHLCV (abertura, máx, mín, fechamento, volume).
    # auto_adjust=True aplica automaticamente ajustes retroativos por
    # dividendos e desdobramentos, tornando os preços comparáveis ao longo
    # do tempo (padrão recomendado para backtests).
    # -------------------------------------------------------------------------
    df = yf.download(
        ticker,
        start=data_inicial,
        end=data_final,
        interval=intervalo,
        auto_adjust=True
    )

    # O yfinance retorna MultiIndex nas colunas; este comando achata para
    # nomes simples: Open, High, Low, Close, Volume
    df.columns = df.columns.get_level_values(0)

    # Interrompe se o ativo não retornou dados válidos
    if df.empty:
        print(f'  [AVISO] Nenhum dado retornado para {ticker}.')
        return None

    # -------------------------------------------------------------------------
    # 2. PARÂMETROS DO SETUP
    # -------------------------------------------------------------------------
    p_min  = 7      # janela para calcular a mínima de curto prazo (suporte)
    p_max  = 9      # janela para calcular a máxima de curto prazo (resistência)
    p_bb   = 13     # período da média móvel das Bandas de Bollinger
    std_bb = 1.61   # multiplicador do desvio padrão (banda ligeiramente estreita)

    # -------------------------------------------------------------------------
    # 3. INDICADORES TÉCNICOS
    # -------------------------------------------------------------------------

    # Mínima rolante de 7 períodos — serve como referência de suporte local.
    # Um fechamento abaixo desta mínima indica fraqueza de curto prazo.
    df['min7'] = df['Low'].rolling(p_min).min()

    # Máxima rolante de 9 períodos — serve como alvo/stop de saída.
    # Um fechamento acima desta máxima sinaliza recuperação do preço.
    df['max9'] = df['High'].rolling(p_max).max()

    # Bandas de Bollinger calculadas pela biblioteca `ta`.
    # A banda inferior = média - (std_bb × desvio padrão).
    # Fechamentos abaixo dela indicam que o preço está "esticado" para baixo.
    bb = ta.volatility.BollingerBands(
        close=df['Close'],
        window=p_bb,
        window_dev=std_bb
    )
    df['bb_lower'] = bb.bollinger_lband()

    # -------------------------------------------------------------------------
    # 4. CONDIÇÕES DE ENTRADA E SAÍDA
    # -------------------------------------------------------------------------

    # COMPRA: o fechamento rompe abaixo da mínima de 7 candles
    #         E também está abaixo da banda inferior de Bollinger.
    # Combina dois filtros de sobrevendido para aumentar a precisão do sinal.
    df['buy_trigger'] = (
        (df['Close'] <= df['min7']) &
        (df['Close'] <  df['bb_lower'])
    )

    # SAÍDA: o fechamento supera a máxima dos últimos 9 candles.
    # Indica que o ativo recuperou o nível de resistência recente.
    df['exit_trigger'] = df['Close'] >= df['max9']

    # -------------------------------------------------------------------------
    # 5. GERAÇÃO DOS SINAIS (controle de posição)
    # -------------------------------------------------------------------------
    # A coluna 'signal' assume:
    #   1 = posição comprada ativa
    #   0 = sem posição
    #
    # Regras de transição:
    #   - Entrada (0→1): buy_trigger ativo e não está posicionado
    #   - Saída  (1→0): exit_trigger ativo e está posicionado
    #   - Manutenção: permanece com o valor anterior
    # -------------------------------------------------------------------------
    df['signal'] = 0
    posicionado  = False

    for i in range(len(df)):
        if not posicionado and df['buy_trigger'].iloc[i]:
            # Registra sinal de entrada no candle atual
            df.iloc[i, df.columns.get_loc('signal')] = 1
            posicionado = True
        elif posicionado and df['exit_trigger'].iloc[i]:
            # Registra sinal de saída (posição zerada)
            df.iloc[i, df.columns.get_loc('signal')] = 0
            posicionado = False
        elif posicionado:
            # Mantém a posição ativa enquanto não há sinal de saída
            df.iloc[i, df.columns.get_loc('signal')] = 1

    # -------------------------------------------------------------------------
    # 6. CÁLCULO DOS RETORNOS
    # -------------------------------------------------------------------------

    # Retorno diário do ativo em percentual: (Close_t / Close_{t-1}) - 1
    df['returns'] = df['Close'].pct_change()

    # Retorno da estratégia: aplica shift(1) para simular a execução da
    # ordem no candle SEGUINTE ao sinal (abertura do próximo pregão).
    # Isso evita look-ahead bias (usar informação do futuro).
    df['strategy'] = df['signal'].shift(1) * df['returns']

    # Detecta mudanças de posição: abs(diff) == 1 quando há entrada ou saída.
    # O custo total (corretagem + slippage) é descontado nessas transições.
    mudancas = df['signal'].diff().abs()
    df['strategy_net'] = df['strategy'] - mudancas * (custo_op + slippage)

    # -------------------------------------------------------------------------
    # 7. CURVA DE CAPITAL (equity curve)
    # -------------------------------------------------------------------------
    # Produto acumulado de (1 + retorno diário líquido), partindo de 1,00.
    # Representa como R$ 1,00 inicial teria evoluído ao longo do período.
    df['equity'] = (1 + df['strategy_net'].fillna(0)).cumprod()

    return df


# =============================================================================
# CÁLCULO DE MÉTRICAS DE DESEMPENHO
# =============================================================================

def calcular_metricas(df, ticker):
    """
    Recebe o DataFrame com a curva de capital e retorna um dicionário
    com todas as métricas de avaliação do backtest.
    """

    # -------------------------------------------------------------------------
    # RETORNO ACUMULADO
    # -------------------------------------------------------------------------
    # Ganho total em relação ao capital inicial.
    # Exemplo: equity final = 3,11 → retorno = 211%
    retorno_total = df['equity'].iloc[-1] - 1

    # -------------------------------------------------------------------------
    # DRAWDOWN MÁXIMO
    # -------------------------------------------------------------------------
    # Drawdown = queda percentual em relação ao pico anterior da equity.
    # cummax() captura o maior valor até o momento t.
    # O drawdown máximo é a maior queda observada no período — mede o risco
    # de perda que o investidor teria suportado em pior cenário.
    rolling_max  = df['equity'].cummax()
    drawdown_ser = (df['equity'] - rolling_max) / rolling_max
    max_drawdown = drawdown_ser.min()

    # -------------------------------------------------------------------------
    # ÍNDICE DE SHARPE ANUALIZADO
    # -------------------------------------------------------------------------
    # Mede o retorno ajustado ao risco: quantos "desvios padrão de retorno"
    # são gerados por unidade de volatilidade.
    # Fórmula: (média diária / desvio padrão diário) × √252
    # 252 = número aproximado de pregões anuais na B3.
    # Sharpe > 1 é considerado bom; > 2 é muito bom.
    ret_diarios = df['strategy_net'].dropna()
    if ret_diarios.std() != 0:
        sharpe = (ret_diarios.mean() / ret_diarios.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # -------------------------------------------------------------------------
    # IDENTIFICAÇÃO DE TRADES INDIVIDUAIS
    # -------------------------------------------------------------------------
    # diff() detecta quando o sinal muda.
    #   +1 = nova entrada (0 → 1)
    #   -1 = saída       (1 → 0)
    signal_changes = df['signal'].diff()
    entradas = df.index[signal_changes ==  1].tolist()
    saidas   = df.index[signal_changes == -1].tolist()

    # Se o backtest terminar com posição aberta, usa o último dia como saída
    if len(entradas) > len(saidas):
        saidas.append(df.index[-1])

    # Calcula o retorno composto de cada trade individualmente
    n_pairs = min(len(entradas), len(saidas))
    trade_returns = []
    for i in range(n_pairs):
        s, e  = entradas[i], saidas[i]
        # Produto dos retornos diários dentro do trade (retorno composto)
        tr = (1 + df.loc[s:e, 'strategy_net'].fillna(0)).prod() - 1
        trade_returns.append(tr)

    trade_returns = np.array(trade_returns)

    # Separa trades vencedores (wins) e perdedores (losses)
    wins   = trade_returns[trade_returns >  0]
    losses = trade_returns[trade_returns <= 0]
    n_trades = len(trade_returns)

    # -------------------------------------------------------------------------
    # PERCENTUAL DE ACERTO (Win Rate)
    # -------------------------------------------------------------------------
    # Proporção de trades que terminaram com resultado positivo.
    pct_acerto = len(wins) / n_trades if n_trades > 0 else np.nan

    # -------------------------------------------------------------------------
    # PAYOFF
    # -------------------------------------------------------------------------
    # Relação entre o ganho médio dos trades vencedores e a perda média dos
    # perdedores. Payoff > 1 significa que os ganhos médios superam as perdas.
    payoff = abs(wins.mean() / losses.mean()) if len(wins) > 0 and len(losses) > 0 else np.nan

    # -------------------------------------------------------------------------
    # PROFIT FACTOR
    # -------------------------------------------------------------------------
    # Soma total dos ganhos dividida pela soma total das perdas (em módulo).
    # Profit Factor > 1 indica estratégia lucrativa no agregado.
    # PF > 1,5 é considerado robusto; PF > 2 é excelente.
    profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.nan

    return {
        'ticker':        ticker,
        'retorno_total': retorno_total,
        'max_drawdown':  max_drawdown,
        'sharpe':        sharpe,
        'n_trades':      n_trades,
        'pct_acerto':    pct_acerto,
        'payoff':        payoff,
        'profit_factor': profit_factor,
        'drawdown_ser':  drawdown_ser,
    }


# =============================================================================
# UTILITÁRIO: formata valor numérico ou retorna 'N/A'
# =============================================================================

def _fmt(v, fmt):
    """Formata `v` com `fmt` se for número válido; caso contrário retorna 'N/A'."""
    return fmt.format(v) if not (isinstance(v, float) and np.isnan(v)) else 'N/A'


# =============================================================================
# EXECUÇÃO DO BACKTEST PARA TODOS OS ATIVOS
# =============================================================================

os.makedirs('results', exist_ok=True)   # garante que a pasta de saída exista

all_metrics = []   # lista com dicionários de métricas de cada ativo
all_dfs     = {}   # dicionário {ticker: DataFrame} para uso nos gráficos

for ticker in tickers:

    df = run_backtest(ticker)
    if df is None:
        continue

    m = calcular_metricas(df, ticker)
    all_metrics.append(m)
    all_dfs[ticker] = df

    # Exibe resumo no terminal
    print('\n' + '=' * 40)
    print(f'ATIVO: {ticker}')
    print('=' * 40)
    print(f'Retorno acumulado : {m["retorno_total"]:.2%}')
    print(f'Drawdown máximo   : {m["max_drawdown"]:.2%}')
    print(f'Sharpe ratio      : {m["sharpe"]:.2f}')
    print(f'Nº de trades      : {m["n_trades"]}')
    print(f'% de acerto       : {_fmt(m["pct_acerto"],    "{:.2%}")}')
    print(f'Payoff            : {_fmt(m["payoff"],        "{:.2f}")}')
    print(f'Profit factor     : {_fmt(m["profit_factor"], "{:.2f}")}')


# =============================================================================
# GERAÇÃO DE IMAGENS INDIVIDUAIS (para uso no Overleaf / LaTeX)
# =============================================================================
# Cada imagem é salva como PNG de alta resolução (300 dpi) na pasta results/.
# Nomenclatura: tabela_resumo.png | equity_PETR4.png | drawdown_PETR4.png ...
# =============================================================================

# -----------------------------------------------------------------------------
# IMAGEM 1: Tabela resumo de métricas de todos os ativos
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 4))
ax.axis('off')

col_labels = [
    'Ativo', 'Retorno Acum.', 'Drawdown Máx.',
    'Sharpe', 'Nº Trades', '% Acerto', 'Payoff', 'Profit Factor'
]

rows = []
for m in all_metrics:
    rows.append([
        m['ticker'],
        f'{m["retorno_total"]:.2%}',
        f'{m["max_drawdown"]:.2%}',
        f'{m["sharpe"]:.2f}',
        str(m['n_trades']),
        _fmt(m['pct_acerto'],    '{:.2%}'),
        _fmt(m['payoff'],        '{:.2f}'),
        _fmt(m['profit_factor'], '{:.2f}'),
    ])

tbl = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.1, 2.4)

# Estilo visual: cabeçalho azul escuro, linhas alternadas
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#2C5F8A')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows) + 1):
    for j in range(len(col_labels)):
        tbl[i, j].set_facecolor('#EEF4FB' if i % 2 == 0 else 'white')

ax.set_title(
    'Backtest — Estratégia Bollinger + Min/Max (B3)\n'
    f'Período: {data_inicial} a {data_final}  |  '
    f'Custo op.: {custo_op:.1%}  |  Slippage: {slippage:.1%}  |  '
    'Dados ajustados por dividendos',
    fontsize=12, fontweight='bold', pad=18
)

plt.tight_layout()
plt.savefig('results/tabela_resumo.png', dpi=300, bbox_inches='tight')
plt.close()
print('\nSalvo: results/tabela_resumo.png')

# -----------------------------------------------------------------------------
# IMAGENS POR ATIVO: curva de capital + drawdown (separadas)
# -----------------------------------------------------------------------------
for m in all_metrics:
    ticker_limpo = m['ticker'].replace('.SA', '')  # ex: PETR4
    df           = all_dfs[m['ticker']]

    # --- Curva de Capital ---
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df.index, df['equity'], color='steelblue', linewidth=1.5)
    ax.axhline(1, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_title(f'Curva de Capital — {m["ticker"]}', fontsize=13, fontweight='bold')
    ax.set_ylabel('Equity (base = 1,00)')
    ax.set_xlabel('Data')
    ax.grid(True, alpha=0.25)

    # Caixa com as principais métricas dentro do gráfico
    info = (
        f'Retorno: {m["retorno_total"]:.2%}   '
        f'Drawdown Máx.: {m["max_drawdown"]:.2%}   '
        f'Sharpe: {m["sharpe"]:.2f}   '
        f'Trades: {m["n_trades"]}   '
        f'Acerto: {_fmt(m["pct_acerto"], "{:.2%}")}   '
        f'Payoff: {_fmt(m["payoff"], "{:.2f}")}   '
        f'Profit Factor: {_fmt(m["profit_factor"], "{:.2f}")}'
    )
    ax.text(
        0.01, 0.03, info,
        transform=ax.transAxes, fontsize=7.5, color='#333333',
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.85)
    )

    plt.tight_layout()
    path_eq = f'results/equity_{ticker_limpo}.png'
    plt.savefig(path_eq, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Salvo: {path_eq}')

    # --- Drawdown ---
    fig, ax = plt.subplots(figsize=(12, 3))

    ax.fill_between(df.index, m['drawdown_ser'] * 100, 0,
                    color='red', alpha=0.35)
    ax.plot(df.index, m['drawdown_ser'] * 100, color='darkred', linewidth=0.9)
    ax.set_title(f'Drawdown (%) — {m["ticker"]}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Drawdown (%)')
    ax.set_xlabel('Data')
    ax.grid(True, alpha=0.25)

    # Linha do drawdown máximo
    ax.axhline(
        m['max_drawdown'] * 100, color='darkred', linestyle='--',
        linewidth=0.8, alpha=0.6,
        label=f'Máx.: {m["max_drawdown"]:.2%}'
    )
    ax.legend(fontsize=9)

    plt.tight_layout()
    path_dd = f'results/drawdown_{ticker_limpo}.png'
    plt.savefig(path_dd, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Salvo: {path_dd}')

# -----------------------------------------------------------------------------
# IMAGEM EXTRA: Comparativo de todas as curvas de capital
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6))

colors = ['steelblue', 'darkorange', 'seagreen', 'crimson', 'mediumpurple']
for i, m in enumerate(all_metrics):
    df = all_dfs[m['ticker']]
    ax.plot(df.index, df['equity'],
            label=m['ticker'], linewidth=1.4, color=colors[i % len(colors)])

ax.axhline(1, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax.set_title('Comparativo — Curvas de Capital', fontsize=13, fontweight='bold')
ax.set_ylabel('Equity (base = 1,00)')
ax.set_xlabel('Data')
ax.legend()
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('results/comparativo_equity.png', dpi=300, bbox_inches='tight')
plt.close()
print('Salvo: results/comparativo_equity.png')


# =============================================================================
# GERAÇÃO DO PDF COMPLETO (todas as páginas em um único arquivo)
# =============================================================================

pdf_path = os.path.join('results', 'relatorio_backtest.pdf')

with pdf_backend.PdfPages(pdf_path) as pdf:

    # Página 1: tabela resumo
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.1, 2.2)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#2C5F8A')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor('#EEF4FB' if i % 2 == 0 else 'white')
    ax.set_title(
        'Backtest — Estratégia Bollinger + Min/Max (B3)\n'
        f'Período: {data_inicial} a {data_final}  |  '
        f'Custo op.: {custo_op:.1%}  |  Slippage: {slippage:.1%}  |  '
        'Dados ajustados por dividendos',
        fontsize=12, fontweight='bold', pad=20
    )
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Uma página por ativo (equity + drawdown)
    for m in all_metrics:
        df = all_dfs[m['ticker']]
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 9),
            gridspec_kw={'height_ratios': [3, 1]}, sharex=True
        )
        ax1.plot(df.index, df['equity'], color='steelblue', linewidth=1.4)
        ax1.axhline(1, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax1.set_title(f'Curva de Capital — {m["ticker"]}', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Equity (base = 1,00)')
        ax1.grid(True, alpha=0.25)
        info = (
            f'Retorno: {m["retorno_total"]:.2%}   '
            f'Drawdown Máx.: {m["max_drawdown"]:.2%}   '
            f'Sharpe: {m["sharpe"]:.2f}   '
            f'Trades: {m["n_trades"]}   '
            f'Acerto: {_fmt(m["pct_acerto"], "{:.2%}")}   '
            f'Payoff: {_fmt(m["payoff"], "{:.2f}")}   '
            f'Profit Factor: {_fmt(m["profit_factor"], "{:.2f}")}'
        )
        ax1.text(0.01, 0.03, info, transform=ax1.transAxes, fontsize=7.5,
                 verticalalignment='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        ax2.fill_between(df.index, m['drawdown_ser'] * 100, 0, color='red', alpha=0.35)
        ax2.plot(df.index, m['drawdown_ser'] * 100, color='darkred', linewidth=0.8)
        ax2.set_title('Drawdown (%)', fontsize=10)
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.25)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    # Última página: comparativo
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, m in enumerate(all_metrics):
        df = all_dfs[m['ticker']]
        ax.plot(df.index, df['equity'], label=m['ticker'],
                linewidth=1.4, color=colors[i % len(colors)])
    ax.axhline(1, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_title('Comparativo — Curvas de Capital', fontsize=13, fontweight='bold')
    ax.set_ylabel('Equity (base = 1,00)')
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print(f'\nRelatório PDF salvo em: {pdf_path}')
