: batchfile for compressing the LiDAR data (LAS to LAZ), index the files and extract info

: set output folders

	: allows you to run the script from other folders
	set PATH=%PATH%;C:\lastools\bin;
cls


	:inputfolder for delta-z laz files 
	set INPUT=C:\Users\luis.barreiro\Downloads
	set OUTPUT=S:\temp\LB\DC\Lidar\LAZ
	set INFO=S:\temp\LB\DC\Lidar\QC


laszip -i %INPUT%\*.las -odir %OUTPUT% -olaz
lasindex -i %OUTPUT%\*.laz

lasinfo -i %OUTPUT%\*.laz -merged -compute_density -histo scan_angle 2 -o %INFO%\DC_lidarinfo.txt


pause