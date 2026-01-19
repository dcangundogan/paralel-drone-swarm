<<<<<<< HEAD
@echo off
echo ============================================
echo  DRONE SWARM WITH SENSORS
echo ============================================
echo.

set DRONES=15

if not "%1"=="" set DRONES=%1

echo Starting with %DRONES% drones...
echo.

python run_with_sensors.py --drones %DRONES% --no-gazebo

pause
=======
@echo off
echo ============================================
echo  DRONE SWARM WITH SENSORS
echo ============================================
echo.

set DRONES=15

if not "%1"=="" set DRONES=%1

echo Starting with %DRONES% drones...
echo.

python run_with_sensors.py --drones %DRONES% --no-gazebo

pause
>>>>>>> 08031cee7f16bb92e769bcc3e346d79078f6f8a2
