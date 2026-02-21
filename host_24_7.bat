@echo off
echo ====================================================
echo   SKIN CARE AI - 24/7 CLOUD DEPLOYMENT TOOL
echo ====================================================
echo.
set /p giturl="PASTE your GitHub Repository Link here: "
if "%giturl%"=="" goto error

echo.
echo [1/3] Connecting to GitHub...
git remote add origin %giturl%
git branch -M main

echo.
echo [2/3] Uploading your code to the Cloud...
git push -u origin main

echo.
echo [3/3] Final Step:
echo 1. Go to https://dashboard.render.com/
echo 2. Click "New" -> "Web Service"
echo 3. Select your "my-skin-ai" repository
echo.
echo Your permanent 24/7 link will be generated in 2 minutes!
pause
exit

:error
echo ERROR: You must provide a GitHub link (e.g., https://github.com/user/repo)
pause
exit
