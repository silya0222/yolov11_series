import socket
import time

from clearml import Task

task = Task.init(project_name="test", task_name="mac_to_windows_test")
task.execute_remotely(queue_name="default", clone=False, exit_process=True)

print("Running on:", socket.gethostname())
for i in range(5):
    print("step", i)
    time.sleep(1)
