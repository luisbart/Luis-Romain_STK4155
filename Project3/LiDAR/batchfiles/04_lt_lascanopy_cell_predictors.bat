: a batch script for extracting predictors from the point cloud

: set output folders

	: allows you to run the script from other folders
	set PATH=%PATH%;C:\lastools\bin;
cls
	
	:inputfolder for delta-z laz files 
	set INPUT=S:\temp\LB\DC\Lidar\LAZ_DZ
	set CELLS=S:\temp\LB\DC\Lidar\predictors_pol

lascanopy -i %INPUT%\*.laz -lop %CELLS%\input\input_pol.shp names -o %CELLS%\output\output_predictors_pol.csv ^
-drop_z_above 50 -drop_class 7 -merged -b_upper 99 -c 0 50 -min -max -std -avg -kur -ske -p 5 25 50 75 90

lascanopy -i %INPUT%\*.laz -lop %CELLS%\input\input_pol.shp names -o %CELLS%\output\output_predictors_pol_INT.csv ^
-drop_z_above 50 -drop_class 7 -merged -b_upper 99 -int_min -int_max -int_std -int_avg -int_kur -int_ske -int_p 5 25 50 75 90

pause