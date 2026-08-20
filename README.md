# diffusers-movie-server

MiniMax-H3(旧 /home/animede/minimax-h3、port 8611)と LTX-2.5(旧 /home/animede/diffusers-ltx2_5、port 8000)の統合サーバ。

- gateway/  : 統一API + プロキシ + プロセスマネージャ(port 8630)
- backends/minimax-h3 : H3 バックエンド(内部 port 8631、専用venv torch2.9+cu128)
- backends/ltx2_5     : LTX-2.5 バックエンド(内部 port 8632、専用venv torch2.11+cu130)

計画: docs/INTEGRATION_PLAN.md 参照。Phase 0 進行中。
