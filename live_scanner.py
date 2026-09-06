#!/usr/bin/env python3
"""
BRUN A/P — LIVE SCANNER
=======================
Runs the validated engine over a watchlist, finds setups that are live RIGHT NOW,
and sends the best 1-2 to Telegram with entry / stop / target so you can pull the
chart up on TradingView mobile and decide.

CONFIG (validated on 200 NSE symbols, 2 years, cost 0.20R):
    ct_valid minValid=2 | rr 2.5 | slBuf 0.10 | minRoomR 1.5
    vsaK 1.2 | HTF EMA on | Setups 1 and 3 only  (Setup 2 is disabled: it lost money)
    out-of-sample +0.556R across 4,761 trades, 97% of symbols profitable

RUN LOCALLY
    pip install yfinance pandas numpy requests
    set TG_TOKEN=...   &  set TG_CHAT=...
    python live_scanner.py --watchlist nifty50 --top 2

RUN FREE ON GITHUB ACTIONS
    see .github/workflows/scan.yml — fires on a schedule, no server to maintain.
"""
import os, sys, argparse, json, time
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------- engine ----
def valid(o, h, l, c):
    return abs(c - o) > max(h - max(o, c), min(o, c) - l)


def scan_symbol(O, H, L, C, V, minValid=2, fib=0.75, rr=2.5, slBuf=0.10,
                minRoomR=1.5, vsaK=1.2, htfLen=200, liveBars=3):
    """Replays the structure and returns a setup only if it is live in the
    last `liveBars` bars. Mirrors the backtested logic exactly."""
    n = len(C)
    tr_ = np.maximum(H - L, np.maximum(abs(H - np.roll(C, 1)), abs(L - np.roll(C, 1))))
    tr_[0] = H[0] - L[0]
    atr = pd.Series(tr_).ewm(alpha=1/14, adjust=False).mean().to_numpy()
    vMA = pd.Series(V).rolling(50, min_periods=5).mean().to_numpy()
    htf = pd.Series(C).ewm(span=htfLen, adjust=False).mean().to_numpy()

    legDir = 0; runExt = runExtB = None; ctr = 0
    lastHigh = lastHighB = lastLow = lastLowB = None
    legHigh = legHighB = legLow = legLowB = None
    pbFrom = 0; regime = 0
    A = AB = P = PB = None
    conflict = False; convExt = convExtB = None; dp = None; dpHit = False
    poiTop = poiBot = None; poiDir = 0; poiTier = 0; poiEffort = 0.0
    out = None

    for i in range(n):
        o, h, l, c = O[i], H[i], L[i], C[i]
        if legHigh is None or h > legHigh: legHigh, legHighB = h, i
        if legLow  is None or l < legLow:  legLow,  legLowB  = l, i
        if conflict and regime == 1 and (convExt is None or l < convExt): convExt, convExtB = l, i
        if conflict and regime == -1 and (convExt is None or h > convExt): convExt, convExtB = h, i
        if conflict and dp is not None and not dpHit:
            if (regime == 1 and l <= dp) or (regime == -1 and h >= dp): dpHit = True

        def gate(frm):
            k = 0
            for q in range(max(0, frm), i + 1):
                if valid(O[q], H[q], L[q], C[q]): k += 1
            return k >= minValid

        if regime <= 0 and lastLow is not None and c < lastLow and gate(pbFrom):
            if regime == 0 or A is None: regime = -1; A, AB = legHigh, legHighB
            elif conflict:
                A = P = convExt; AB = PB = convExtB
                conflict = False; dp = None; dpHit = False; convExt = None
            else: P, PB = A, AB; A, AB = legHigh, legHighB
            w = slice(max(0, pbFrom), i)
            if w.stop > w.start:
                poiTop, poiBot, poiDir, poiTier = H[w].max(), L[w].min(), -1, (0 if conflict else 1)
                poiEffort = V[i] / max(vMA[i], 1e-9)
            lastLow = None; legHigh = None; pbFrom = i

        if regime >= 0 and lastHigh is not None and c > lastHigh and gate(pbFrom):
            if regime == 0 or A is None: regime = 1; A, AB = legLow, legLowB
            elif conflict:
                A = P = convExt; AB = PB = convExtB
                conflict = False; dp = None; dpHit = False; convExt = None
            else: P, PB = A, AB; A, AB = legLow, legLowB
            w = slice(max(0, pbFrom), i)
            if w.stop > w.start:
                poiTop, poiBot, poiDir, poiTier = H[w].max(), L[w].min(), 1, (0 if conflict else 1)
                poiEffort = V[i] / max(vMA[i], 1e-9)
            lastHigh = None; legLow = None; pbFrom = i

        if regime == 1 and lastLow is not None and c < lastLow and not conflict and gate(pbFrom):
            conflict = True; A, AB = legHigh, legHighB
            convExt, convExtB = l, i; dpHit = False
            dp = None if P is None else legHigh - (legHigh - P) * fib
            lastLow = None; pbFrom = i
        if regime == -1 and lastHigh is not None and c > lastHigh and not conflict and gate(pbFrom):
            conflict = True; A, AB = legLow, legLowB
            convExt, convExtB = h, i; dpHit = False
            dp = None if P is None else legLow + (P - legLow) * fib
            lastHigh = None; pbFrom = i

        if P is not None and regime != 0:
            if (regime == 1 and c < P) or (regime == -1 and c > P):
                regime = -regime; P = PB = None; conflict = False
                dp = None; dpHit = False; convExt = None
                if regime == -1: A, AB = legHigh, legHighB
                else:            A, AB = legLow,  legLowB

        if legDir == 0: legDir = 1; runExt, runExtB = h, i; ctr = 0
        elif legDir == 1:
            if h > runExt: runExt, runExtB = h, i; ctr = 0
            elif valid(o, h, l, c) and c < o: ctr += 1
            if ctr >= minValid:
                lastHigh, lastHighB = runExt, runExtB; legDir = -1
                runExt, runExtB = l, i; ctr = 0; legLow, legLowB = l, i; pbFrom = i
        else:
            if l < runExt: runExt, runExtB = l, i; ctr = 0
            elif valid(o, h, l, c) and c > o: ctr += 1
            if ctr >= minValid:
                lastLow, lastLowB = runExt, runExtB; legDir = 1
                runExt, runExtB = h, i; ctr = 0; legHigh, legHighB = h, i; pbFrom = i

        # ---- is a setup live on this bar? ----
        if i >= n - liveBars and poiTop is not None:
            a_ = max(atr[i], 1e-9); e = s_ = t_ = None; tier = 0; dd = 0
            if conflict and dpHit and dp is not None and P is not None and regime != 0:
                e, s_ = dp, P
                if abs(e - s_) > 0: t_ = e + rr * abs(e - s_) * regime; tier, dd = 3, regime
            elif not conflict and poiTier == 1 and poiDir == regime:
                e = poiTop if poiDir == 1 else poiBot
                s_ = (poiBot - slBuf * a_) if poiDir == 1 else (poiTop + slBuf * a_)
                if abs(e - s_) > 0: t_ = e + rr * abs(e - s_) * poiDir; tier, dd = 1, poiDir
            if tier and t_ is not None:
                risk = abs(e - s_); ok = True
                if minRoomR > 0 and P is not None and abs(e - P) < minRoomR * risk: ok = False
                if vsaK > 0 and poiEffort < vsaK: ok = False
                if (dd == 1) != (C[i] > htf[i]): ok = False           # HTF alignment
                if ok:
                    dist = abs(C[-1] - e) / max(a_, 1e-9)
                    out = dict(tier=tier, dir='LONG' if dd == 1 else 'SHORT',
                               entry=float(e), stop=float(s_), target=float(t_),
                               rr=round(abs(t_ - e) / risk, 2), effort=round(poiEffort, 2),
                               atr_away=round(dist, 2), last=float(C[-1]),
                               regime='BULL' if regime == 1 else 'BEAR')
    return out


# ------------------------------------------------------------- watchlists ---
# Indices and commodities. yfinance tickers differ from NSE equity symbols:
# ^NSEI = Nifty 50, ^NSEBANK = Bank Nifty, CL=F = WTI crude, BZ=F = Brent.
# MCX crude is not on yfinance; CL=F is the closest proxy and moves with it.
INDICES = {'^NSEI': 'NIFTY', '^NSEBANK': 'BANKNIFTY', 'CL=F': 'CRUDE', 'BZ=F': 'BRENT'}

NIFTY50 = ("RELIANCE TCS HDFCBANK ICICIBANK INFY HINDUNILVR ITC SBIN BHARTIARTL BAJFINANCE "
           "KOTAKBANK LT AXISBANK ASIANPAINT MARUTI SUNPHARMA TITAN ULTRACEMCO WIPRO NESTLEIND "
           "ONGC NTPC POWERGRID TATAMOTORS TATASTEEL M&M HCLTECH ADANIENT JSWSTEEL COALINDIA "
           "BAJAJFINSV GRASIM HINDALCO INDUSINDBK DRREDDY CIPLA BRITANNIA EICHERMOT APOLLOHOSP "
           "DIVISLAB TECHM HEROMOTOCO BPCL TATACONSUM SBILIFE HDFCLIFE ADANIPORTS UPL LTIM SHRIRAMFIN").split()


def fetch(sym, interval='15m', period='60d'):
    import yfinance as yf
    # indices and futures carry their own ticker format; equities need .NS
    t = sym if (sym in INDICES or sym.endswith('.NS') or '=' in sym or sym.startswith('^')) else f"{sym}.NS"
    df = yf.download(t, interval=interval, period=period, progress=False, auto_adjust=False)
    if df is None or len(df) < 400: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return (df['Open'].to_numpy(float), df['High'].to_numpy(float),
            df['Low'].to_numpy(float), df['Close'].to_numpy(float),
            np.maximum(df['Volume'].to_numpy(float), 1.0))


def telegram(msg, token=None, chat=None):
    token = token or os.environ.get('TG_TOKEN')
    chat  = chat  or os.environ.get('TG_CHAT')
    if not token or not chat:
        print("[no TG_TOKEN/TG_CHAT — printing instead]\n" + msg); return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={'chat_id': chat, 'text': msg, 'parse_mode': 'HTML'}, timeout=20)
        return r.ok
    except Exception as e:
        print("telegram failed:", e); return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='yf', choices=['yf', 'kite'],
                    help='kite = real-time via Kite Connect; yf = ~15 min delayed')
    ap.add_argument('--watchlist', default='all', choices=['all', 'indices', 'equities'],
                    help='all = Nifty, Bank Nifty, crude and the Nifty 50 stocks')
    ap.add_argument('--symbols', default='', help='comma-separated, overrides watchlist')
    ap.add_argument('--interval', default='15m')
    ap.add_argument('--period', default='60d')
    ap.add_argument('--top', type=int, default=5, help='how many setups to send')
    ap.add_argument('--maxatr', type=float, default=1.5,
                    help='only alert if price is within this many ATR of the entry')
    a = ap.parse_args()

    if a.symbols:
        syms = [s.strip() for s in a.symbols.split(',') if s.strip()]
    elif a.watchlist == 'indices':
        syms = list(INDICES)
    elif a.watchlist == 'equities':
        syms = NIFTY50
    else:                                   # 'all' — indices first, then equities
        syms = list(INDICES) + NIFTY50

    # pick the data source
    if a.source == 'kite':
        import kite_data
        getbars = lambda s: kite_data.fetch_kite(s, a.interval, days=90)
        ltp_map = kite_data.live_price(syms)
        print(f"scanning {len(syms)} symbols on {a.interval} via KITE (real-time)")
    else:
        getbars = lambda s: fetch(s, a.interval, a.period)
        ltp_map = {}
        print(f"scanning {len(syms)} symbols on {a.interval} via yfinance (delayed)")
    found = []
    for s in syms:
        try:
            d = getbars(s)
            if d is None: continue
            r = scan_symbol(*d)
            if r and s in ltp_map:
                # real-time last price overrides the last closed bar
                r['last'] = float(ltp_map[s])
                atr_est = abs(r['entry'] - r['stop']) / max(0.10, 1e-9)
                r['atr_away'] = round(abs(r['last'] - r['entry']) / max(atr_est, 1e-9), 2)
            if r and r['atr_away'] <= a.maxatr:
                r['symbol'] = INDICES.get(s, s); found.append(r)
        except Exception as e:
            print(f"  {s}: {e}")
        time.sleep(0.15)

    # closest to entry first — those are the ones about to trigger
    found.sort(key=lambda x: x['atr_away'])
    if not found:
        print("no live setups"); return

    ist = pd.Timestamp.now(tz='Asia/Kolkata').strftime('%d %b %H:%M')
    lines = [f"<b>BRUN A/P — {ist} IST</b>", f"<i>{len(found)} live, showing top {a.top}</i>", ""]
    for r in found[:a.top]:
        arrow = "🟢 LONG" if r['dir'] == 'LONG' else "🔴 SHORT"
        lines += [f"<b>{r['symbol']}</b>  {arrow}  (Setup {r['tier']})",
                  f"  Entry  <b>{r['entry']:.2f}</b>",
                  f"  Stop   {r['stop']:.2f}",
                  f"  Target {r['target']:.2f}   ({r['rr']}R)",
                  f"  last {r['last']:.2f} · {r['atr_away']} ATR away",
                  f"  regime {r['regime']} · effort {r['effort']}x", ""]
    lines.append("<i>Verify on TradingView before entering.</i>")
    msg = "\n".join(lines)
    telegram(msg)

    # publish for the mobile app (GitHub Pages serves this file)
    payload = dict(updated=pd.Timestamp.now(tz='Asia/Kolkata').isoformat(),
                   interval=a.interval, source=a.source, scanned=len(syms), signals=found[:12])
    with open('signals.json', 'w') as f:
        json.dump(payload, f, indent=1, default=float)
    print(f"wrote signals.json ({len(found[:12])} signals)")
    print(msg.replace('<b>','').replace('</b>','').replace('<i>','').replace('</i>',''))


if __name__ == '__main__':
    main()
