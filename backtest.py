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
from scipy import stats                        # testes estatísticos
from scipy.stats import norm as sci_norm       # distribuição normal cumulativa

# =============================================================================
# PARÂMETROS GLOBAIS
# =============================================================================

# Lista de ativos da B3 selecionados por liquidez
tickers = ['PETR4.SA', 'ITUB4.SA', 'VALE3.SA', 'ABEV3.SA', 'WEGE3.SA']

# Janela temporal do backtest
data_inicial = '2015-01-01'
data_final   = '2025-01-01'

# Timeframe utilizado: diário (D1)
intervalo = '1d'

# Custo por operação: 0,30% — cobre corretagem, emolumentos e taxas B3
custo_op = 0.003

# Slippage estimado: 0,10% — diferença entre o preço esperado e o executado
# Ocorre por falta de liquidez pontual ou delay na execução
slippage = 0.001

# Constante de Euler-Mascheroni (usada no cálculo do DSR)
EULER_MASCHERONI = 0.5772156649

# Grade de 27 variantes de parâmetros para os testes WRC e DSR.
# Combina diferentes janelas de entrada (p_min), saída (p_max) e Bollinger (p_bb).
PARAM_GRID_WRC = [
    (pm, px, pb)
    for pm in [5, 7, 10]    # janela mínima de entrada
    for px in [7, 9, 12]    # janela máxima de saída
    for pb in [10, 13, 20]  # período das Bandas de Bollinger
]
N_TRIALS = len(PARAM_GRID_WRC)  # 27

# Janelas do walk-forward: 5 anos de treino (expansivo) + 1 ano de teste.
# data_inicial = '2015-01-01' → 5 janelas cobrindo 2020–2024.
WF_WINDOWS = [
    {
        'janela':       i + 1,
        'treino_inicio': data_inicial,
        'treino_fim':   f'{2019 + i}-12-31',
        'teste_inicio': f'{2020 + i}-01-01',
        'teste_fim':    f'{2020 + i}-12-31',
    }
    for i in range(5)
]


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
# VARIANTE DE ESTRATÉGIA (helper para WRC e DSR)
# =============================================================================

def _variant_returns(df, p_min, p_max, p_bb):
    """
    Executa a mesma estratégia com parâmetros alternativos.
    Retorna array numpy com os retornos diários líquidos da variante.
    """
    min_r = df['Low'].rolling(p_min).min()
    max_r = df['High'].rolling(p_max).max()
    bb    = ta.volatility.BollingerBands(close=df['Close'], window=p_bb, window_dev=1.61)
    bbl   = bb.bollinger_lband()

    buy   = (df['Close'] <= min_r) & (df['Close'] < bbl)
    exit_ = df['Close'] >= max_r

    sig = np.zeros(len(df))
    pos = False
    for i in range(len(df)):
        if not pos and buy.iloc[i]:
            sig[i] = 1; pos = True
        elif pos and exit_.iloc[i]:
            sig[i] = 0; pos = False
        elif pos:
            sig[i] = 1

    sig_s  = pd.Series(sig, index=df.index)
    rets   = df['Close'].pct_change()
    strat  = sig_s.shift(1) * rets
    mudanc = sig_s.diff().abs()
    net    = (strat - mudanc * (custo_op + slippage)).fillna(0)
    return net.values


# =============================================================================
# TESTES ESTATÍSTICOS DE VALIDADE DO BACKTEST
# =============================================================================

def calcular_testes(df, ticker, n_sim=5000, seed=42):
    """
    Executa os 5 testes estatísticos:
      a) Teste t de Student
      b) Bootstrap do Sharpe Ratio
      c) Monte Carlo por embaralhamento dos retornos
      d) Deflated Sharpe Ratio  (Bailey & López de Prado, 2014)
      e) White's Reality Check  (White, 2000)
    """
    rng = np.random.default_rng(seed)
    ret = df['strategy_net'].dropna().values
    T   = len(ret)

    def sharpe_ann(r):
        return (r.mean() / r.std()) * np.sqrt(252) if r.std() > 1e-12 else 0.0

    # -------------------------------------------------------------------------
    # a) TESTE t DE STUDENT
    # H0: retorno médio diário = 0 (estratégia não gera alfa)
    # -------------------------------------------------------------------------
    t_stat, p_ttest = stats.ttest_1samp(ret, 0)

    # -------------------------------------------------------------------------
    # b) BOOTSTRAP DO SHARPE RATIO
    # Reamostragem com reposição para construir a distribuição empírica do Sharpe.
    # p-valor: P(Sharpe bootstrap ≤ 0) — mede robustez do Sharpe observado.
    # IC 95%: intervalo de confiança pelo método dos percentis.
    # -------------------------------------------------------------------------
    real_sh = sharpe_ann(ret)
    boot_sh = np.array([
        sharpe_ann(rng.choice(ret, size=T, replace=True))
        for _ in range(n_sim)
    ])
    p_boot  = float((boot_sh <= 0).mean())
    ci_boot = np.percentile(boot_sh, [2.5, 97.5])

    # -------------------------------------------------------------------------
    # c) MONTE CARLO POR EMBARALHAMENTO
    # Permuta aleatória dos retornos — destrói qualquer estrutura de timing.
    # p-valor: P(Sharpe simulado ≥ Sharpe real) — estratégia vs. ruído puro.
    # -------------------------------------------------------------------------
    mc_sh = np.array([
        sharpe_ann(rng.permutation(ret))
        for _ in range(n_sim)
    ])
    p_mc = float((mc_sh >= real_sh).mean())

    # -------------------------------------------------------------------------
    # d) DEFLATED SHARPE RATIO (DSR)
    # Corrige o Sharpe por: (1) tamanho finito da amostra,
    # (2) não-normalidade (assimetria γ3 e curtose γ4),
    # (3) múltiplos testes (N variantes de parâmetros).
    #
    # Variância do estimador Sharpe (Bailey & López de Prado, eq. 3):
    #   σ²[SR] = (1 − γ3·SR + (γ4−1)/4·SR²) / (T−1)
    #
    # PSR(SR*) = Φ( (SR_hat − SR*) / σ[SR] )
    # SR*: benchmark esperado sob N testes (eq. 11 do mesmo artigo)
    # -------------------------------------------------------------------------
    sr_p  = ret.mean() / ret.std() if ret.std() > 1e-12 else 0.0  # Sharpe por período
    skew  = float(stats.skew(ret))
    kurt  = float(stats.kurtosis(ret, fisher=False))  # curtose regular (Normal = 3)
    var_sr = max((1 - skew * sr_p + (kurt - 1) / 4 * sr_p**2) / (T - 1), 1e-12)

    psr_0 = float(sci_norm.cdf(sr_p / np.sqrt(var_sr)))  # PSR(0): sem ajuste por múltiplos testes

    # Roda todas as variantes para estimar σ_SR (dispersão dos Sharpes entre parâmetros)
    print(f'  [DSR/WRC] Calculando {N_TRIALS} variantes para {ticker}...')
    variant_rets = []
    variant_sh   = []
    for (pm, px, pb) in PARAM_GRID_WRC:
        vr = _variant_returns(df, pm, px, pb)
        variant_rets.append(vr)
        variant_sh.append(sharpe_ann(vr))
    variant_sh = np.array(variant_sh)
    sigma_sr   = variant_sh.std() if variant_sh.std() > 1e-12 else 1.0

    # SR* esperado como máximo de N testes sob H0 (Bailey & López de Prado, 2014)
    sr_star = sigma_sr * (
        (1 - EULER_MASCHERONI) * sci_norm.ppf(1 - 1 / N_TRIALS) +
        EULER_MASCHERONI       * sci_norm.ppf(1 - 1 / (N_TRIALS * np.e))
    )
    sr_star_p = sr_star / np.sqrt(252)  # converte para unidade de período
    dsr = float(sci_norm.cdf((sr_p - sr_star_p) / np.sqrt(var_sr)))

    # -------------------------------------------------------------------------
    # e) WHITE'S REALITY CHECK (White, 2000)
    # Testa se a melhor estratégia de um conjunto supera benchmark por acaso.
    # H0: E[f_k] ≤ 0 para toda variante k (nenhuma gera retorno positivo).
    # Estatística observada: V_N = max_k( mean(f_k) )
    # Bootstrap centrado: V_boot_b = max_k( mean(f_k_b*) − mean(f_k) )
    # p-valor: P(V_boot ≥ V_N)
    # -------------------------------------------------------------------------
    T_w   = len(variant_rets[0])
    panel = np.vstack(variant_rets)   # shape: (N_TRIALS, T_w)
    f_bar = panel.mean(axis=1)        # média temporal por variante, shape: (N_TRIALS,)
    V_N   = float(f_bar.max())

    V_boot_wrc = np.empty(n_sim)
    for b in range(n_sim):
        idx              = rng.integers(0, T_w, size=T_w)
        boot_means       = panel[:, idx].mean(axis=1)
        V_boot_wrc[b]    = (boot_means - f_bar).max()  # centrado em H0

    p_wrc = float((V_boot_wrc >= V_N).mean())

    return {
        't_stat':     t_stat,    'p_ttest':    p_ttest,
        'real_sh':    real_sh,   'boot_sh':    boot_sh,
        'p_boot':     p_boot,    'ci_boot':    ci_boot,
        'mc_sh':      mc_sh,     'p_mc':       p_mc,
        'psr_0':      psr_0,     'dsr':        dsr,      'sr_star':    sr_star,
        'variant_sh': variant_sh,
        'V_N':        V_N,       'V_boot_wrc': V_boot_wrc, 'p_wrc': p_wrc,
    }


# =============================================================================
# HELPERS E FUNÇÕES ADICIONAIS
# =============================================================================

def _sharpe_ann(arr):
    """Sharpe anualizado a partir de um array numpy de retornos diários."""
    return (arr.mean() / arr.std()) * np.sqrt(252) if arr.std() > 1e-12 else 0.0


def walk_forward(df_raw, ticker):
    """
    Otimização walk-forward expansivo: treino crescente de 5 anos, teste de 1 ano.
    Para cada janela, seleciona os parâmetros com maior Sharpe no treino e
    aplica fora da amostra (teste).  Retorna lista de dicts por janela.
    """
    resultados = []
    print(f'  [Walk-Forward] Otimizando {len(WF_WINDOWS)} janelas para {ticker}...')
    for w in WF_WINDOWS:
        df_tr = df_raw.loc[w['treino_inicio'] : w['treino_fim']]
        df_te = df_raw.loc[w['teste_inicio']  : w['teste_fim']]

        if len(df_tr) < 200 or len(df_te) < 20:
            continue

        # Otimiza Sharpe no treino
        best_sh, best_p = -np.inf, PARAM_GRID_WRC[0]
        for (pm, px, pb) in PARAM_GRID_WRC:
            sh = _sharpe_ann(_variant_returns(df_tr, pm, px, pb))
            if sh > best_sh:
                best_sh, best_p = sh, (pm, px, pb)

        # Aplica os melhores parâmetros no período de teste
        pm, px, pb = best_p
        ret_te = _variant_returns(df_te, pm, px, pb)

        resultados.append({
            'janela':         w['janela'],
            'treino_ini':     w['treino_inicio'],
            'treino_fim':     w['treino_fim'],
            'teste_ini':      w['teste_inicio'],
            'teste_fim':      w['teste_fim'],
            'obs_treino':     len(df_tr),
            'obs_teste':      len(df_te),
            'p_min':          pm,
            'p_max':          px,
            'p_bb':           pb,
            'sharpe_treino':  round(best_sh, 3),
            'retorno_teste':  float((1 + ret_te).prod() - 1),
            'ano_teste':      int(w['teste_inicio'][:4]),
        })
    return resultados


def calcular_metricas_ext(df, ticker):
    """
    Wrapper sobre calcular_metricas() que adiciona CAGR, Sortino, Calmar,
    exposição ao mercado, retorno bruto e custo total estimado.
    """
    m   = calcular_metricas(df, ticker)
    ret = df['strategy_net'].dropna()
    n   = len(ret)

    years = n / 252
    cagr  = df['equity'].iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0

    neg     = ret[ret < 0]
    sortino = (ret.mean() / neg.std()) * np.sqrt(252) if len(neg) > 1 and neg.std() > 1e-12 else 0.0

    calmar = cagr / abs(m['max_drawdown']) if m['max_drawdown'] != 0 else np.nan

    n_posic    = int(df['signal'].sum())
    exposure   = n_posic / len(df)
    n_entradas = int((df['signal'].diff() == 1).sum())
    ret_bruto  = float((1 + df['strategy'].fillna(0)).cumprod().iloc[-1] - 1)
    custo_est  = n_entradas * 2 * (custo_op + slippage)

    m.update({
        'cagr':        cagr,
        'sortino':     sortino,
        'calmar':      calmar,
        'exposure':    exposure,
        'n_posic':     n_posic,
        'n_entradas':  n_entradas,
        'ret_bruto':   ret_bruto,
        'custo_est':   custo_est,
    })
    return m


def caracterizar_ativo(ticker, df):
    """Estatísticas descritivas da série de preços do ativo."""
    return {
        'ticker':         ticker,
        'periodo_ini':    df.index[0].strftime('%d/%m/%Y'),
        'periodo_fim':    df.index[-1].strftime('%d/%m/%Y'),
        'n_pregoes':      len(df),
        'preco_inicial':  float(df['Close'].iloc[0]),
        'preco_final':    float(df['Close'].iloc[-1]),
        'retorno_ativo':  float(df['Close'].iloc[-1] / df['Close'].iloc[0] - 1),
    }


def retornos_anuais(df):
    """Retorno composto da estratégia por ano-calendário."""
    ret  = df['strategy_net'].fillna(0)
    anos = {}
    for year in sorted(ret.index.year.unique()):
        mask      = ret.index.year == year
        anos[year] = float((1 + ret[mask]).prod() - 1)
    return anos


# =============================================================================
# EXECUÇÃO DO BACKTEST PARA TODOS OS ATIVOS
# =============================================================================

os.makedirs('results', exist_ok=True)   # garante que a pasta de saída exista

all_metrics = []   # lista com dicionários de métricas de cada ativo
all_dfs     = {}   # dicionário {ticker: DataFrame} para uso nos gráficos
all_testes  = {}   # resultados dos testes estatísticos por ativo
all_caract  = {}   # caracterização dos ativos
all_wf      = {}   # resultados walk-forward por ativo
all_anuais  = {}   # retornos anuais por ativo

for ticker in tickers:

    df = run_backtest(ticker)
    if df is None:
        continue

    m = calcular_metricas_ext(df, ticker)
    all_metrics.append(m)
    all_dfs[ticker] = df
    all_caract[ticker] = caracterizar_ativo(ticker, df)
    all_anuais[ticker] = retornos_anuais(df)

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

    t = calcular_testes(df, ticker)
    all_testes[ticker] = t

    wf = walk_forward(df, ticker)
    all_wf[ticker] = wf

    sig = lambda p: '✓ sig.' if p < 0.05 else '✗ n.s.'
    print(f'\n  --- Testes Estatísticos (α=0,05) ---')
    print(f'  a) Teste t        : t={t["t_stat"]:+.3f},  p={t["p_ttest"]:.4f}  {sig(t["p_ttest"])}')
    print(f'  b) Bootstrap SR   : p={t["p_boot"]:.4f}  {sig(t["p_boot"])},  '
          f'IC95=[{t["ci_boot"][0]:.2f}, {t["ci_boot"][1]:.2f}]')
    print(f'  c) Monte Carlo    : p={t["p_mc"]:.4f}  {sig(t["p_mc"])}')
    print(f'  d) PSR(0) / DSR   : {t["psr_0"]:.4f} / {t["dsr"]:.4f}  '
          f'(SR*={t["sr_star"]:.2f})')
    print(f'  e) White RC       : p={t["p_wrc"]:.4f}  {sig(t["p_wrc"])},  '
          f'V_N={t["V_N"]:.6f}')


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
# IMAGEM: TABELA RESUMO DOS TESTES ESTATÍSTICOS
# =============================================================================

fig, ax = plt.subplots(figsize=(18, 4))
ax.axis('off')

col_testes = [
    'Ativo',
    'a) t-stat', 'a) p-val',
    'b) Boot p', 'b) IC95 Sharpe',
    'c) MC p',
    'd) PSR(0)', 'd) DSR', 'd) SR*',
    'e) WRC p',
]

rows_testes = []
for tk, t in all_testes.items():
    rows_testes.append([
        tk,
        f'{t["t_stat"]:+.3f}',  f'{t["p_ttest"]:.4f}',
        f'{t["p_boot"]:.4f}',   f'[{t["ci_boot"][0]:.2f}, {t["ci_boot"][1]:.2f}]',
        f'{t["p_mc"]:.4f}',
        f'{t["psr_0"]:.4f}',    f'{t["dsr"]:.4f}',  f'{t["sr_star"]:.2f}',
        f'{t["p_wrc"]:.4f}',
    ])

tbl_t = ax.table(cellText=rows_testes, colLabels=col_testes, loc='center', cellLoc='center')
tbl_t.auto_set_font_size(False)
tbl_t.set_fontsize(9)
tbl_t.scale(1.0, 2.3)

for j in range(len(col_testes)):
    tbl_t[0, j].set_facecolor('#1A4A6B')
    tbl_t[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows_testes) + 1):
    for j in range(len(col_testes)):
        tbl_t[i, j].set_facecolor('#EEF4FB' if i % 2 == 0 else 'white')

ax.set_title(
    f'Testes Estatísticos — Validade do Backtest  '
    f'(N_TRIALS={N_TRIALS}  |  n_sim=5000  |  sig. = p<0,05)',
    fontsize=11, fontweight='bold', pad=18
)
plt.tight_layout()
plt.savefig('results/tabela_testes.png', dpi=300, bbox_inches='tight')
plt.close()
print('Salvo: results/tabela_testes.png')


# =============================================================================
# IMAGENS POR ATIVO — TESTES ESTATÍSTICOS
# =============================================================================

def _hist_safe(ax, arr, cap=60, **kwargs):
    """Histograma robusto com range explícito: funciona mesmo com valores quase idênticos."""
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi - lo < 1e-15:
        lo, hi = lo - 0.5, hi + 0.5
    for n in (cap, max(1, cap // 2), 20, 10, 5, 3, 1):
        try:
            ax.hist(arr, bins=n, range=(lo, hi), **kwargs)
            return
        except (ValueError, Exception):
            continue
    # Último recurso: linha vertical na média
    mean_val = float(np.mean(arr))
    ax.axvline(mean_val, color=kwargs.get('color', 'steelblue'),
               lw=3, alpha=kwargs.get('alpha', 0.7),
               label=kwargs.get('label', f'≈{mean_val:.4f}'))
    ax.text(0.5, 0.5, f'Valores constantes\n≈{mean_val:.6f}',
            transform=ax.transAxes, ha='center', va='center', fontsize=9,
            bbox=dict(facecolor='lightyellow', alpha=0.85))

for ticker, t in all_testes.items():
    ticker_limpo = ticker.replace('.SA', '')

    # --- Gráfico 1: Bootstrap Sharpe (b) + Monte Carlo (c) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    _hist_safe(ax1, t['boot_sh'], color='steelblue', alpha=0.70, density=True,
               label='Bootstrap Sharpe')
    ax1.axvline(t['real_sh'],    color='red',   lw=2,   label=f'Sharpe real: {t["real_sh"]:.2f}')
    ax1.axvline(0,               color='black', lw=1.2, linestyle='--', label='SR = 0')
    ax1.axvline(t['ci_boot'][0], color='green', lw=1,   linestyle=':',
                label=f'IC95: [{t["ci_boot"][0]:.2f}, {t["ci_boot"][1]:.2f}]')
    ax1.axvline(t['ci_boot'][1], color='green', lw=1,   linestyle=':')
    ax1.set_title(f'b) Bootstrap do Sharpe — {ticker}', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Sharpe Anualizado')
    ax1.set_ylabel('Densidade')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)
    ax1.text(0.03, 0.97, f'p-valor = {t["p_boot"]:.4f}', transform=ax1.transAxes,
             fontsize=9, va='top', bbox=dict(facecolor='lightyellow', alpha=0.85))

    _hist_safe(ax2, t['mc_sh'], color='darkorange', alpha=0.70, density=True,
               label='Sharpe (permutações)')
    ax2.axvline(t['real_sh'], color='red',   lw=2,   label=f'Sharpe real: {t["real_sh"]:.2f}')
    ax2.axvline(0,            color='black', lw=1.2, linestyle='--')
    ax2.set_title(f'c) Monte Carlo (embaralhamento) — {ticker}', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Sharpe Anualizado')
    ax2.set_ylabel('Densidade')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.text(0.03, 0.97, f'p-valor = {t["p_mc"]:.4f}', transform=ax2.transAxes,
             fontsize=9, va='top', bbox=dict(facecolor='lightyellow', alpha=0.85))

    plt.suptitle(f'Testes b) e c) — {ticker}', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path_bc = f'results/testes_bc_{ticker_limpo}.png'
    plt.savefig(path_bc, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Salvo: {path_bc}')

    # --- Gráfico 2: DSR (d) + White's Reality Check (e) ---
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

    ax3.hist(t['variant_sh'], bins=15, color='mediumpurple', alpha=0.75, density=True,
             label=f'Sharpe das {N_TRIALS} variantes')
    ax3.axvline(t['real_sh'], color='red',  lw=2,   label=f'Sharpe real: {t["real_sh"]:.2f}')
    ax3.axvline(t['sr_star'], color='navy', lw=1.5, linestyle='--',
                label=f'SR* (benchmark): {t["sr_star"]:.2f}')
    ax3.set_title(f'd) Deflated Sharpe Ratio — {ticker}', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Sharpe Anualizado')
    ax3.set_ylabel('Densidade')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.25)
    ax3.text(0.03, 0.97,
             f'PSR(0) = {t["psr_0"]:.4f}\nDSR    = {t["dsr"]:.4f}',
             transform=ax3.transAxes, fontsize=9, va='top',
             bbox=dict(facecolor='lightyellow', alpha=0.85))

    _hist_safe(ax4, t['V_boot_wrc'], color='seagreen', alpha=0.70, density=True,
               label='Distribuição nula (bootstrap)')
    ax4.axvline(t['V_N'], color='red',   lw=2,   label=f'V_N observado: {t["V_N"]:.5f}')
    ax4.axvline(0,        color='black', lw=1.2, linestyle='--')
    ax4.set_title(f"e) White's Reality Check — {ticker}", fontsize=11, fontweight='bold')
    ax4.set_xlabel('max( mean(fₖ) ) sob H₀')
    ax4.set_ylabel('Densidade')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.25)
    ax4.text(0.03, 0.97, f'p-valor = {t["p_wrc"]:.4f}', transform=ax4.transAxes,
             fontsize=9, va='top', bbox=dict(facecolor='lightyellow', alpha=0.85))

    plt.suptitle(f'Testes d) e e) — {ticker}', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path_de = f'results/testes_de_{ticker_limpo}.png'
    plt.savefig(path_de, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Salvo: {path_de}')


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

    # Página 2: tabela de testes estatísticos
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.axis('off')
    tbl_t2 = ax.table(cellText=rows_testes, colLabels=col_testes, loc='center', cellLoc='center')
    tbl_t2.auto_set_font_size(False)
    tbl_t2.set_fontsize(9)
    tbl_t2.scale(1.0, 2.2)
    for j in range(len(col_testes)):
        tbl_t2[0, j].set_facecolor('#1A4A6B')
        tbl_t2[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows_testes) + 1):
        for j in range(len(col_testes)):
            tbl_t2[i, j].set_facecolor('#EEF4FB' if i % 2 == 0 else 'white')
    ax.set_title(
        f'Testes Estatísticos — Validade do Backtest  '
        f'(N_TRIALS={N_TRIALS}  |  n_sim=5000  |  sig. = p<0,05)',
        fontsize=11, fontweight='bold', pad=20
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

        # Sub-página B: Bootstrap + Monte Carlo
        t = all_testes[m['ticker']]
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6))

        _hist_safe(axA, t['boot_sh'], color='steelblue', alpha=0.70, density=True)
        axA.axvline(t['real_sh'],    color='red',   lw=2,  label=f'Real: {t["real_sh"]:.2f}')
        axA.axvline(0,               color='black', lw=1.2, linestyle='--')
        axA.axvline(t['ci_boot'][0], color='green', lw=1,  linestyle=':',
                    label=f'IC95: [{t["ci_boot"][0]:.2f}, {t["ci_boot"][1]:.2f}]')
        axA.axvline(t['ci_boot'][1], color='green', lw=1,  linestyle=':')
        axA.set_title(f'b) Bootstrap Sharpe — {m["ticker"]}', fontweight='bold')
        axA.set_xlabel('Sharpe Anualizado')
        axA.legend(fontsize=8)
        axA.grid(True, alpha=0.25)
        axA.text(0.03, 0.97, f'p = {t["p_boot"]:.4f}', transform=axA.transAxes,
                 fontsize=9, va='top', bbox=dict(facecolor='lightyellow', alpha=0.85))

        _hist_safe(axB, t['mc_sh'], color='darkorange', alpha=0.70, density=True)
        axB.axvline(t['real_sh'], color='red',   lw=2,   label=f'Real: {t["real_sh"]:.2f}')
        axB.axvline(0,            color='black', lw=1.2, linestyle='--')
        axB.set_title(f'c) Monte Carlo — {m["ticker"]}', fontweight='bold')
        axB.set_xlabel('Sharpe Anualizado')
        axB.legend(fontsize=8)
        axB.grid(True, alpha=0.25)
        axB.text(0.03, 0.97, f'p = {t["p_mc"]:.4f}', transform=axB.transAxes,
                 fontsize=9, va='top', bbox=dict(facecolor='lightyellow', alpha=0.85))

        plt.suptitle(f'Testes b) e c) — {m["ticker"]}', fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Sub-página C: DSR + WRC
        fig, (axC, axD) = plt.subplots(1, 2, figsize=(14, 6))

        axC.hist(t['variant_sh'], bins=15, color='mediumpurple', alpha=0.75, density=True)
        axC.axvline(t['real_sh'], color='red',  lw=2,   label=f'Real: {t["real_sh"]:.2f}')
        axC.axvline(t['sr_star'], color='navy', lw=1.5, linestyle='--',
                    label=f'SR*: {t["sr_star"]:.2f}')
        axC.set_title(f'd) DSR — {m["ticker"]}', fontweight='bold')
        axC.set_xlabel('Sharpe das variantes')
        axC.legend(fontsize=8)
        axC.grid(True, alpha=0.25)
        axC.text(0.03, 0.97,
                 f'PSR(0)={t["psr_0"]:.4f}\nDSR={t["dsr"]:.4f}',
                 transform=axC.transAxes, fontsize=9, va='top',
                 bbox=dict(facecolor='lightyellow', alpha=0.85))

        _hist_safe(axD, t['V_boot_wrc'], color='seagreen', alpha=0.70, density=True)
        axD.axvline(t['V_N'], color='red',   lw=2,   label=f'V_N: {t["V_N"]:.5f}')
        axD.axvline(0,        color='black', lw=1.2, linestyle='--')
        axD.set_title(f"e) White's Reality Check — {m['ticker']}", fontweight='bold')
        axD.set_xlabel('max( mean(fₖ) ) sob H₀')
        axD.legend(fontsize=8)
        axD.grid(True, alpha=0.25)
        axD.text(0.03, 0.97, f'p = {t["p_wrc"]:.4f}', transform=axD.transAxes,
                 fontsize=9, va='top', bbox=dict(facecolor='lightyellow', alpha=0.85))

        plt.suptitle(f'Testes d) e e) — {m["ticker"]}', fontweight='bold')
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


# =============================================================================
# IMAGENS ADICIONAIS — SEÇÕES 4.1–4.12 DO RELATÓRIO
# =============================================================================

# -----------------------------------------------------------------------------
# 4.1 TABELA DE CARACTERIZAÇÃO DOS ATIVOS
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 3.5))
ax.axis('off')

col_caract = ['Ativo', 'Período Início', 'Período Fim', 'Nº Pregões',
              'Preço Inicial (R$)', 'Preço Final (R$)', 'Retorno do Ativo']
rows_caract = []
for tk, c in all_caract.items():
    rows_caract.append([
        c['ticker'],
        c['periodo_ini'], c['periodo_fim'],
        str(c['n_pregoes']),
        f'{c["preco_inicial"]:.2f}', f'{c["preco_final"]:.2f}',
        f'{c["retorno_ativo"]:.2%}',
    ])

tbl_c = ax.table(cellText=rows_caract, colLabels=col_caract, loc='center', cellLoc='center')
tbl_c.auto_set_font_size(False); tbl_c.set_fontsize(10); tbl_c.scale(1.1, 2.4)
for j in range(len(col_caract)):
    tbl_c[0, j].set_facecolor('#2C5F8A')
    tbl_c[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows_caract) + 1):
    for j in range(len(col_caract)):
        tbl_c[i, j].set_facecolor('#EEF4FB' if i % 2 == 0 else 'white')
ax.set_title('Tabela 1 – Caracterização dos Ativos Analisados',
             fontsize=11, fontweight='bold', pad=16)
plt.tight_layout()
plt.savefig('results/tabela_caracterizacao.png', dpi=300, bbox_inches='tight')
plt.close()
print('Salvo: results/tabela_caracterizacao.png')

# -----------------------------------------------------------------------------
# 4.2 TABELA DAS JANELAS WALK-FORWARD
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 3.5))
ax.axis('off')

col_wf = ['Janela', 'Treino Início', 'Treino Fim', 'Teste Início', 'Teste Fim']
rows_wf = [[str(w['janela']), w['treino_inicio'], w['treino_fim'],
            w['teste_inicio'], w['teste_fim']] for w in WF_WINDOWS]

tbl_wf = ax.table(cellText=rows_wf, colLabels=col_wf, loc='center', cellLoc='center')
tbl_wf.auto_set_font_size(False); tbl_wf.set_fontsize(10); tbl_wf.scale(1.1, 2.4)
for j in range(len(col_wf)):
    tbl_wf[0, j].set_facecolor('#2C5F8A')
    tbl_wf[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows_wf) + 1):
    for j in range(len(col_wf)):
        tbl_wf[i, j].set_facecolor('#EEF4FB' if i % 2 == 0 else 'white')
ax.set_title('Tabela 2 – Janelas de Treino e Teste (Walk-Forward Expansivo)',
             fontsize=11, fontweight='bold', pad=16)
plt.tight_layout()
plt.savefig('results/tabela_wf_janelas.png', dpi=300, bbox_inches='tight')
plt.close()
print('Salvo: results/tabela_wf_janelas.png')

# -----------------------------------------------------------------------------
# 4.3 TABELA DE PARÂMETROS SELECIONADOS POR JANELA E ATIVO
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, max(4, len(tickers) * len(WF_WINDOWS) * 0.45 + 1.5)))
ax.axis('off')

col_par = ['Ativo', 'Ano Teste', 'p_min', 'p_max', 'p_bb',
           'Sharpe Treino', 'Retorno Teste']
rows_par = []
for tk in tickers:
    if tk not in all_wf:
        continue
    for w in all_wf[tk]:
        rows_par.append([
            tk, str(w['ano_teste']),
            str(w['p_min']), str(w['p_max']), str(w['p_bb']),
            f'{w["sharpe_treino"]:.2f}',
            f'{w["retorno_teste"]:.2%}',
        ])

tbl_par = ax.table(cellText=rows_par, colLabels=col_par, loc='center', cellLoc='center')
tbl_par.auto_set_font_size(False); tbl_par.set_fontsize(9); tbl_par.scale(1.0, 1.9)
for j in range(len(col_par)):
    tbl_par[0, j].set_facecolor('#2C5F8A')
    tbl_par[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows_par) + 1):
    for j in range(len(col_par)):
        tbl_par[i, j].set_facecolor('#EEF4FB' if i % 2 == 0 else 'white')
ax.set_title('Tabela 3 – Parâmetros Otimizados por Ativo e Janela Walk-Forward',
             fontsize=11, fontweight='bold', pad=16)
plt.tight_layout()
plt.savefig('results/tabela_wf_params.png', dpi=300, bbox_inches='tight')
plt.close()
print('Salvo: results/tabela_wf_params.png')

# -----------------------------------------------------------------------------
# 4.4 RETORNO ANUAL FORA DA AMOSTRA (gráfico de barras agrupadas)
# -----------------------------------------------------------------------------
anos_wf = sorted({w['ano_teste'] for tk in all_wf for w in all_wf[tk]})
x = np.arange(len(anos_wf))
width = 0.8 / len(tickers)

fig, ax = plt.subplots(figsize=(13, 5))
for i, tk in enumerate([t for t in tickers if t in all_wf]):
    rets_wf = {w['ano_teste']: w['retorno_teste'] for w in all_wf[tk]}
    vals = [rets_wf.get(a, 0) for a in anos_wf]
    ax.bar(x + i * width - 0.4 + width / 2, [v * 100 for v in vals],
           width, label=tk, color=colors[i % len(colors)], alpha=0.85)

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels([str(a) for a in anos_wf])
ax.set_title('Retorno Anual Fora da Amostra por Ativo (Walk-Forward)',
             fontsize=12, fontweight='bold')
ax.set_ylabel('Retorno (%)')
ax.set_xlabel('Ano de Teste')
ax.legend(); ax.grid(True, alpha=0.25, axis='y')
plt.tight_layout()
plt.savefig('results/retorno_anual_fora.png', dpi=300, bbox_inches='tight')
plt.close()
print('Salvo: results/retorno_anual_fora.png')

# -----------------------------------------------------------------------------
# 4.6 TABELA DE MÉTRICAS CONSOLIDADAS EXTENDIDAS
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(18, 4))
ax.axis('off')

col_ext = ['Ativo', 'Ret. Acum.', 'CAGR', 'Sharpe', 'Sortino', 'Calmar',
           'Máx. DD', '% Acerto', 'Profit F.', 'Exposição']
rows_ext = []
for m in all_metrics:
    rows_ext.append([
        m['ticker'],
        f'{m["retorno_total"]:.2%}',
        f'{m["cagr"]:.2%}',
        f'{m["sharpe"]:.2f}',
        _fmt(m['sortino'], '{:.2f}'),
        _fmt(m['calmar'],  '{:.2f}'),
        f'{m["max_drawdown"]:.2%}',
        _fmt(m['pct_acerto'],    '{:.2%}'),
        _fmt(m['profit_factor'], '{:.2f}'),
        f'{m["exposure"]:.1%}',
    ])

tbl_ext = ax.table(cellText=rows_ext, colLabels=col_ext, loc='center', cellLoc='center')
tbl_ext.auto_set_font_size(False); tbl_ext.set_fontsize(9.5); tbl_ext.scale(1.0, 2.4)
for j in range(len(col_ext)):
    tbl_ext[0, j].set_facecolor('#2C5F8A')
    tbl_ext[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows_ext) + 1):
    for j in range(len(col_ext)):
        tbl_ext[i, j].set_facecolor('#EEF4FB' if i % 2 == 0 else 'white')
ax.set_title('Tabela 5 – Métricas Consolidadas de Desempenho e Risco',
             fontsize=11, fontweight='bold', pad=16)
plt.tight_layout()
plt.savefig('results/tabela_metricas_ext.png', dpi=300, bbox_inches='tight')
plt.close()
print('Salvo: results/tabela_metricas_ext.png')

# -----------------------------------------------------------------------------
# 4.8 TABELA DE EXPOSIÇÃO E FREQUÊNCIA OPERACIONAL
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 3.5))
ax.axis('off')

col_exp = ['Ativo', 'Nº Operações', 'Dias Posicionado',
           'Exposição', 'Custo Total Est.', 'Ret. Bruto', 'Ret. Líquido']
rows_exp = []
for m in all_metrics:
    rows_exp.append([
        m['ticker'],
        str(m['n_entradas']),
        str(m['n_posic']),
        f'{m["exposure"]:.1%}',
        f'{m["custo_est"]:.2%}',
        f'{m["ret_bruto"]:.2%}',
        f'{m["retorno_total"]:.2%}',
    ])

tbl_exp = ax.table(cellText=rows_exp, colLabels=col_exp, loc='center', cellLoc='center')
tbl_exp.auto_set_font_size(False); tbl_exp.set_fontsize(10); tbl_exp.scale(1.1, 2.4)
for j in range(len(col_exp)):
    tbl_exp[0, j].set_facecolor('#2C5F8A')
    tbl_exp[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows_exp) + 1):
    for j in range(len(col_exp)):
        tbl_exp[i, j].set_facecolor('#EEF4FB' if i % 2 == 0 else 'white')
ax.set_title('Tabela 6 – Frequência Operacional e Exposição ao Mercado',
             fontsize=11, fontweight='bold', pad=16)
plt.tight_layout()
plt.savefig('results/tabela_exposicao.png', dpi=300, bbox_inches='tight')
plt.close()
print('Salvo: results/tabela_exposicao.png')

# -----------------------------------------------------------------------------
# 4.9 HISTOGRAMAS DOS RETORNOS DIÁRIOS
# -----------------------------------------------------------------------------
n_ativos = len(all_metrics)
fig, axes = plt.subplots(1, n_ativos, figsize=(4 * n_ativos, 4), sharey=False)
if n_ativos == 1:
    axes = [axes]

for ax, m in zip(axes, all_metrics):
    ret_d = all_dfs[m['ticker']]['strategy_net'].dropna() * 100
    ax.hist(ret_d, bins=50, color='steelblue', alpha=0.75, density=False)
    ax.axvline(0, color='red', lw=1.2, linestyle='--')
    ax.axvline(ret_d.mean(), color='green', lw=1.2, linestyle='-',
               label=f'Média: {ret_d.mean():.3f}%')
    ax.set_title(m['ticker'].replace('.SA', ''), fontsize=10, fontweight='bold')
    ax.set_xlabel('Retorno Diário (%)')
    ax.set_ylabel('Frequência')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

plt.suptitle('Distribuição dos Retornos Diários da Estratégia',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('results/histograma_retornos.png', dpi=300, bbox_inches='tight')
plt.close()
print('Salvo: results/histograma_retornos.png')

# -----------------------------------------------------------------------------
# 4.11 TABELA COMPARATIVA GERAL
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 3.5))
ax.axis('off')

col_comp = ['Ativo', 'Ret. Acum.', 'Sharpe', 'Máx. DD', 'Calmar',
            'DSR', 'WRC p-val', 'Sig. Estat.', 'Avaliação']

def _avaliacao(m, t):
    sharpe_ok = m['sharpe'] > 1
    dsr_ok    = t['dsr']    > 0.90
    wrc_ok    = t['p_wrc']  < 0.05
    dd_ok     = abs(m['max_drawdown']) < 0.30
    n_ok = sum([sharpe_ok, dsr_ok, wrc_ok, dd_ok])
    if n_ok >= 3: return 'Forte'
    if n_ok == 2: return 'Moderada'
    return 'Fraca'

rows_comp = []
for m in all_metrics:
    t = all_testes[m['ticker']]
    rows_comp.append([
        m['ticker'],
        f'{m["retorno_total"]:.2%}',
        f'{m["sharpe"]:.2f}',
        f'{m["max_drawdown"]:.2%}',
        _fmt(m['calmar'], '{:.2f}'),
        f'{t["dsr"]:.4f}',
        f'{t["p_wrc"]:.4f}',
        'Sim' if t['p_wrc'] < 0.05 else 'Não',
        _avaliacao(m, t),
    ])

tbl_comp = ax.table(cellText=rows_comp, colLabels=col_comp, loc='center', cellLoc='center')
tbl_comp.auto_set_font_size(False); tbl_comp.set_fontsize(9.5); tbl_comp.scale(1.0, 2.4)
for j in range(len(col_comp)):
    tbl_comp[0, j].set_facecolor('#1A4A6B')
    tbl_comp[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows_comp) + 1):
    for j in range(len(col_comp)):
        tbl_comp[i, j].set_facecolor('#EEF4FB' if i % 2 == 0 else 'white')
ax.set_title('Tabela 8 – Síntese Comparativa dos Ativos Analisados',
             fontsize=11, fontweight='bold', pad=16)
plt.tight_layout()
plt.savefig('results/tabela_comparativo.png', dpi=300, bbox_inches='tight')
plt.close()
print('Salvo: results/tabela_comparativo.png')

print('\n[Concluído] Todos os arquivos gerados em results/')
