@echo off
Rem count start = 0, incrementing value = 1, max value = 6
FOR /L %%x IN (0, 1, 6) DO (
	start python worker.py %%x
    	timeout 25 >nul
    	echo "started instance %%x"
	)

pause