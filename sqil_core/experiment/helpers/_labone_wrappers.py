import json

from laboneq import serializers


def w_save(obj, path):
    serializers.save(obj, path)
    # Re-open the file, read the content, and re-save it pretty-printed
    with open(path, "r", encoding="utf-8") as f:  # AB
        data = json.load(f)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


"maybe I need to add something here1"
