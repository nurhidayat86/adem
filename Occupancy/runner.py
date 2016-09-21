import subprocess;

command = "python pcasvmconf_trmonth.py ";
sr = [1, 2, 3, 4, 5, 10, 30, 60, 300, 600, 900, 1800];
fl = [300, 600, 900, 1800, 3600];
h = ['r2'];

for house in h:
	for sampling in sr:
		for feature in fl:
			cmd = command + "--house=" + house + " --sr=" + str(sampling) + " --fl=" + str(feature);
			print(cmd);
			subprocess.call(cmd, shell=True);
