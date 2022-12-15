: a batch script for making delta-z

: set output folders

	: allows you to run the script from other folders
	set PATH=%PATH%;C:\lastools\bin;

	:input folder containing the original las files
	set LAZ_INPUT=S:\temp\LB\DC\Lidar\LAZ

	:outputfolder for tiled laz files (make sure it exists and is empty)
	set LAZ_OUTPUT=S:\temp\LB\DC\Lidar\LAZ_DZ

	:set number of cores
	set CORES=16


lasheight -i %LAZ_INPUT%\*.laz -odir %LAZ_OUTPUT% -olaz -replace_z -buffered 50 -cores %CORES%
lasindex -i %LAZ_OUTPUT%\*.laz -cores %CORES%



pause