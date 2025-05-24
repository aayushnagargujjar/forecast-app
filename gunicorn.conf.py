# gunicorn.conf.py
bind = "0.0.0.0:10000"
workers = 1
timeout = 60  # Increase if necessary
