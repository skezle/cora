import os
import time
import shutil
import glob


path = "."
now = time.time()

while True:
    for filename in glob.glob("/tmp/nle*"):
        mtime = os.path.getmtime(filename)
        if mtime < now - 60 * 30:  # older than 30 minutes
            print(filename)
            #shutil.rmtree(filename)
    time.sleep(60)