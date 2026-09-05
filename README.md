# market-image-prediction
Independent quant research project based on predicting future market 'images'


Universe: SPY, QQQ, IWM
Decision clock: every 15 minutes during regular US market hours
Input: previous 64 × 15-minute bars
Tensor: [channels=6, scales=1, time=64]
Channels: log return, high-low range, log-volume z-score,
          realised volatility, VWAP deviation, time-of-day
Target: next-day realised-volatility tercile
Validation: expanding/rolling walk-forward; no random split
Economic question: does the predicted state improve a volatility-aware
                   strategy after conservative costs?