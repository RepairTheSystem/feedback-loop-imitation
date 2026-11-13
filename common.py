"""
Общий код с repeat_switch: модель, метрики, бутстрап, инференс.

Код переиспользуется импортом, а не переносом — чтобы не ломать repeat_switch и его самодостаточный
ноутбук для Kaggle. Модули этого проекта названы иначе (sim, fl_*), чтобы не конфликтовать по именам
с repeat_switch/{data, metrics, model, train}.py.
"""
import os
import sys

RS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rs_core")   # копия модулей repeat_switch
if RS_DIR not in sys.path:
    sys.path.insert(0, RS_DIR)

from metrics import bootstrap_diff, per_user, summarize, topk  # noqa: E402,F401
from model import OneStageRec  # noqa: E402,F401
from train import nn_outputs  # noqa: E402,F401
