import psutil as ps
import time
start_cpu = ps.cpu_percent()
start_mem = ps.virtual_memory()[2]
max_cpu = 0
max_mem = 0
avg_cpu = 0
avg_mem = 0
for x in range(100):
    max_cpu = max(max_cpu, ps.cpu_percent())
    max_mem = max(max_mem, ps.virtual_memory()[2])
    avg_cpu += ps.cpu_percent()
    avg_mem += ps.virtual_memory()[2]
    time.sleep(0.1)
print("pid cpu {}".format(max_cpu - start_cpu))
print("pid mem {}".format(max_mem - start_mem))
print("avg cpu {}".format(avg_cpu / 100 - start_cpu))
print("avg mem {}".format(avg_mem / 100 - start_mem))