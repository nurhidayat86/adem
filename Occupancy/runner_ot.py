import subprocess;

command = "python threshold.py ";
ot = [100];
sr = [1];
fl = [300, 600, 900, 1800, 3600];

for thres in ot:
	for sampling in sr:
		for feature in fl:
			cmd = command + "--ot=" + str(thres) + " --sr=" + str(sampling) + " --fl=" + str(feature);
			print(cmd);
			subprocess.call(cmd, shell=True);
