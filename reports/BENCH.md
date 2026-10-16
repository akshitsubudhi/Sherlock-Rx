# Scale benchmark

_Measured 2026-10-17 on arm64 (8 cores) with `sherlock bench`. Peak RSS is cumulative across the run._

| Scale | Territories | Sales rows | All rows | Parquet MB | Generate s | Open s | RCA cold p50 ms | RCA warm p50 ms | Scan p50 ms | Chat turn p50 ms | NLU p50 ms | RCA/s (1 thread) | RCA/s (8 threads) | RCA/s (4 processes) | Notes index build s | Peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 113 | 75,936 | 332,435 | 2.4 | 0.21 | 0.01 | 68.3 | 55.2 | 8.5 | 9.9 | 0.47 | 17.1 | 12.0 | 51.4 | 0.03 | 345 |
| 10.0 | 1,134 | 762,048 | 3,084,416 | 17.7 | 1.44 | 0.01 | 108.9 | 61.9 | 23.9 | 12.2 | 0.54 | 13.9 | 10.7 | 24.5 | 0.23 | 697 |
| 50.0 | 5,670 | 3,810,240 | 15,311,051 | 63.0 | 7.11 | 0.01 | 202.7 | 92.6 | 66.7 | 29.5 | 0.47 | 9.0 | 8.5 | 13.9 | 1.12 | 1,493 |
| 100.0 | 11,340 | 7,620,480 | 30,594,079 | 118.1 | 14.54 | 0.04 | 252.5 | 118.3 | 74.2 | 35.8 | 0.46 | 6.9 | 6.3 | 8.2 | 2.19 | 1,536 |
