@echo off
bin\python -m pip freeze > tmp
bin\python -m pip uninstall -y -r tmp
del tmp
bin\python -m pip cache purge
pause
