

: set output folders

	: allows you to run the script from other folders
	set PATH=%PATH%;C:\lastools\bin;

	: set folders
	set LAZ_DZ=S:\temp\LB\DC\Lidar\LAZ_DZ
	set NDSM=S:\temp\LB\DC\Lidar\raster_NDSM

:decide number of cores to use
set CORES=32

:create NDSM 1x1m m highest hit
:lasgrid -i %LAZ_DZ%\*.laz -odir %NDSM%\tiles_50cm -obil -step 0.5  -elevation -highest -drop_z_above 50 -keep_class 5 -nodata 0 -nbits 16 -fill 3 -subsample 3 -mem 2000 -cores %CORES%
:lasgrid -i %NDSM%\tiles_50cm\*.bil -o %NDSM%\DC_nDSM_50cm_Class5.bil -merged -step 0.5 -mem 2000

lasgrid -i %LAZ_DZ%\*.laz -o %NDSM%\DC_nDSM_1m_Class5.bil -merged -step 1 -elevation -highest -drop_z_above 50 -keep_class 5 -nodata 0 -nbits 16 -fill 3 -subsample 3 -mem 2000


pause