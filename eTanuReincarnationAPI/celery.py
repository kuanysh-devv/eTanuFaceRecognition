from __future__ import absolute_import, unicode_literals

import os
from datetime import datetime
from celery import Celery
from celery.schedules import crontab

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "eTanuReincarnationAPI.settings")
app = Celery("eTanuReincarnationAPI")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.conf.enable_utc = False
app.conf.timezone = 'Asia/Almaty'
app.autodiscover_tasks()
