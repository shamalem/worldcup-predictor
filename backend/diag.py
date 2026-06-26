import traceback
from app.prediction import service
from app.score_service import score_service
print("outcome ready:", service.ready, "| score ready:", score_service.ready)
try:
    r = service.predict("Brazil", "Germany", 2014, "Final", True, "none")
    print("OUTCOME OK:", r["predicted_result"])
except Exception:
    print("OUTCOME FAILED:"); traceback.print_exc()
try:
    s = score_service.predict("Brazil", "Germany")
    print("SCORE OK:", s["most_likely_score"]["text"])
except Exception:
    print("SCORE FAILED:"); traceback.print_exc()
