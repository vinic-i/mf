# ================================================================
# GRADE EXPANDIDA + FUNÇÕES MODIFICADAS (TIME STOP + STOP PERCENTUAL)
# ================================================================
# Substitua as funções gerar_grade(), calcular_sinal() e aplicar_stop()
# na Célula 2 do notebook por estas versões.


def gerar_grade_expandida():
    """
    Grade EXPANDIDA para o Setup 7 — Bollinger Bands.

    Aumenta significativamente o número de combinações testadas,
    permitindo ao walk-forward encontrar parâmetros que geram
    mais trades por ativo.

    Parâmetros: periodo (média), desvios (std), e opcionalmente
    stop_loss percentual e time_stop (saída por tempo).
    """
    grade = []

    # PERÍODO da média móvel: valores mais baixos geram mais sinais
    # O padrão era apenas [10, 15, 20]. Agora expandimos para:
    for periodo in [5, 7, 10, 13, 15, 20, 25]:

        # DESVIOS: bandas mais estreitas geram mais toques
        # O padrão era [1.5, 2.0, 2.5]. Agora incluímos valores menores:
        for desvios in [1.0, 1.25, 1.5, 1.61, 2.0, 2.5, 3.0]:

            # STOP LOSS percentual: proteção de capital
            # Testamos sem stop e com stops de 3%, 5%, 8%
            for stop_loss in [None, 0.03, 0.05, 0.08]:

                # TIME STOP: saída após N candles se o alvo não for atingido
                # Isso evita que o capital fique preso por meses
                # Aumenta drasticamente o número de trades
                for time_stop in [None, 5, 10, 15, 20]:

                    grade.append({
                        'periodo': periodo,
                        'desvios': desvios,
                        'stop_loss': stop_loss,
                        'time_stop': time_stop,
                    })

    return grade


def calcular_sinal_com_time_stop(janela, params):
    """
    Setup 7 — Fechou Fora, Fechou Dentro – Bollinger.
    VERSÃO COM TIME STOP para aumentar frequência de trades.

    Parâmetros adicionais em params:
      - time_stop: int ou None. Número máximo de candles para manter posição.
                   Se None, usa apenas o alvo original (bb_mid).

    Retorna pd.Series de 0s e 1s com index=janela.index.
    """
    import pandas as pd
    import numpy as np

    close   = janela['Close']
    periodo = params['periodo']
    desvios = params['desvios']
    time_stop = params.get('time_stop', None)

    # --- Calcula as Bandas de Bollinger ---
    bb_mid   = close.rolling(periodo).mean()
    bb_std   = close.rolling(periodo).std()
    bb_upper = bb_mid + desvios * bb_std
    bb_lower = bb_mid - desvios * bb_std

    # Regra 56: candle fecha ABAIXO da banda inferior
    outside = (close < bb_lower)

    # Regra 57: candle SEGUINTE fecha DENTRO das bandas
    inside = (close > bb_lower) & (close < bb_upper)

    # Regra 58: sinal de ENTRADA
    entrada = outside.shift(1).fillna(False) & inside

    # Regra 60: sinal de SAÍDA — alvo na média central
    saida_alvo = (close >= bb_mid)

    # ----------------------------------------------------------------
    # Constrói o sinal via STATE MACHINE com TIME STOP
    # ----------------------------------------------------------------
    n_linhas    = len(close)
    sinal_arr   = np.zeros(n_linhas, dtype=int)
    posicionado = False
    candles_na_posicao = 0  # contador para time stop

    for i in range(n_linhas):
        if not posicionado:
            # Fora de trade: verifica entrada
            if bool(entrada.iloc[i]):
                posicionado  = True
                sinal_arr[i] = 1
                candles_na_posicao = 1  # inicia contagem
            # else: permanece fora → 0

        else:
            # Dentro de trade: verifica saída por alvo
            if bool(saida_alvo.iloc[i]):
                posicionado  = False
                sinal_arr[i] = 0   # saiu pelo alvo
                candles_na_posicao = 0

            # Verifica saída por time stop
            elif time_stop is not None and candles_na_posicao >= time_stop:
                posicionado  = False
                sinal_arr[i] = 0   # saiu por tempo
                candles_na_posicao = 0

            else:
                sinal_arr[i] = 1   # permanece comprado
                candles_na_posicao += 1

    sinal = pd.Series(sinal_arr, index=janela.index)
    return sinal


def aplicar_stop_modificado(strategy, janela, params):
    """
    Aplica stop loss percentual e/ou stop gain na série de retornos.

    Parâmetros em params:
      - stop_loss: float ou None. Ex: 0.05 = 5% de perda máxima.
      - stop_gain: float ou None. Ex: 0.10 = 10% de ganho máximo.
    """
    import numpy as np

    stop_loss = params.get('stop_loss', None)
    stop_gain = params.get('stop_gain', None)

    if stop_loss is None and stop_gain is None:
        return strategy

    strategy = strategy.copy()
    retorno_acumulado_trade = 0.0

    for i in range(len(strategy)):
        retorno_acumulado_trade = (
            (1 + retorno_acumulado_trade) *
            (1 + float(strategy.iloc[i])) - 1
        )

        stop_loss_atingido = (
            stop_loss is not None and
            retorno_acumulado_trade < -stop_loss
        )

        stop_gain_atingido = (
            stop_gain is not None and
            retorno_acumulado_trade > stop_gain
        )

        if stop_loss_atingido or stop_gain_atingido:
            strategy.iloc[i]        = 0.0
            retorno_acumulado_trade  = 0.0

    return strategy


# Contagem de combinações
total_comb = len(gerar_grade_expandida())
print(f"Grade expandida: {total_comb} combinações de parâmetros")
print(f"
Isso representa um aumento de {total_comb/9:.0f}x em relação à grade original (9 combinações)")
print("
Novos parâmetros incluídos:")
print("  - Períodos: 5, 7, 10, 13, 15, 20, 25 (antes: 10, 15, 20)")
print("  - Desvios: 1.0, 1.25, 1.5, 1.61, 2.0, 2.5, 3.0 (antes: 1.5, 2.0, 2.5)")
print("  - Stop Loss: None, 3%, 5%, 8% (novo)")
print("  - Time Stop: None, 5, 10, 15, 20 candles (novo)")
