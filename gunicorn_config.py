# gunicorn_config.py
# Render free tier: 512MB RAM, 0.1 CPU
# tensorflow-cpu model inference is slow — 120s timeout prevents 502

bind             = "0.0.0.0:10000"
workers          = 1          # MUST be 1 — each worker loads the model (~300MB)
threads          = 1          # single-threaded is fine for free tier
timeout          = 120        # seconds — prevents 502 during model.predict()
preload_app      = False      # MUST be False — let lazy-load work
worker_class     = "sync"
max_requests     = 100        # recycle worker after 100 requests to prevent memory creep
max_requests_jitter = 10
loglevel         = "info"
accesslog        = "-"        # stdout → visible in Render logs
errorlog         = "-"        # stdout → visible in Render logs
