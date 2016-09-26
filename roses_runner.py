import subprocess;

command = "python roses.py ";
sr = [1];
fl = [60, 120, 300, 600, 900];

for sampling in sr:
	for feature in fl:
		cmd = command + " --sr=" + str(sampling) + " --fl=" + str(feature);
		print(cmd);
		subprocess.call(cmd, shell=True);
