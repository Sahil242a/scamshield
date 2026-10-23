@echo off
REM Rebuild the dataset (including your corrections in data\feedback.csv) and retrain. Takes ~2 minutes.
python data\build_dataset.py || goto :error
python train.py || goto :error
echo.
echo Done. Restart the server (python -m uvicorn api:app) to use the new model.
goto :eof
:error
echo Something went wrong - see the message above.
