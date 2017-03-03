software and libraries used and setup


1) ununtu 14.04 LTS 
2) Anaconda 4.3.0
	Python 2.7 version : Anaconda2-4.3.0-Linux-x86_64.sh 
	download from https://www.continuum.io/downloads
			after download run
	sudo -H bash Anaconda2-4.3.0-Linux-x86_64.sh -b -f -p /anaconda""
			this will install anaconda environment in the '/' directory
	sudo /anaconda/bin/conda install jupyter-notebook

3) make
	sudo apt-get install make
4) g++-4.8.4 
	sudo apt-get insatll g++
	Note: ununtu 16.04 by default g++5.0 is provided. This version missmacths with xgboost will creat linking errors with xgboost. If you are on ubuntu 16.04 make sure u install g++4.8 using command sudo apt-get install g++-4.8 and restart the computer.

5) xgboost
Note:  
sudo /anacona/bin/pip install xgboost && sudo /anacona/bin/pip install --upgrade xgboost
can install the xgoost library, but this version may not support parameter colsample_by_level that we use in this project.

alternatively you can download the source code and complie by the following steps
git clone --recursive https://github.com/dmlc/xgboost 
cd xgboost
make -j4 (j stands for number of processors)
cd python-package
sudo /anaconda/bin/python setup.py install
reboot the computer


Steps to startup

After you have extracter the CapstonePraject.zip" file

1) first run "python extractnformat.py" from the command prompt.
2) next launch and run DataAnalysisandImplementation.ipynb using jupyter-notebook (note: it will need atleat18Gb of RAM)

