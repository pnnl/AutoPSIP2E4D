# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:40:49 2022

@author: mang464 john775 robi526

Run this in a python environment launched from the directory where the PSIP 
data are stored.

Will need an input file with full paths to:
    1) Electrode locations
    2) Electrode Sequence
    3) Meshfile
    4) Output Options file
    5) Inversion Options file

"""
import os
import time
import pandas as pd
import numpy as np
import subprocess as sp
import glob
import matplotlib.pyplot as mpl
import matplotlib

# Global Variables
thisfreq = 1
e4dpath = '/home/mang464/codes/E4d/bin/e4d'
visitpath = '/home/mang464/codes/VisIt/bin/visit'
pxpath = '/home/mang464/codes/E4d/bin/px'
mpipath = '/home/mang464/codes/petsc/arch-linux-c-opt/bin/mpirun'
nproc = 20
matplotlib.use('Agg')

# Open and read the input file
#inp_file = sys.argv[1]
inp_file = 'AutoPSIP_inputfile.py'
f = open(inp_file,'r')
locfile = f.readline()
seqfile = f.readline()
meshfile = f.readline()
outopts = f.readline()
invopts = f.readline()
f.close()

# log the starting directory
ldir = os.getcwd()

def outputElecs():
    SRVout.write(str(len(locs))+'\n')
    locs.to_csv(SRVout, sep='\t', index=True, header=False, line_terminator='\n', float_format='%.3f')

# Find data files that need survey files
while True:
    print (time.strftime("%I:%M:%S")+' checking for new data files')
    for fil in os.listdir('.'):
        if fil.endswith('.csv'):
            print('-----------------------------------------------------------------')
            print('\nFound a PSIP data file that needs inversion.')
            print('\nProcessing data file '+fil)
        
            # Make a directory to store the files using the file name
            rprojdir = fil.split('.',1)[0]+'real'
            iprojdir = fil.split('.',1)[0]+'imag'
            os.mkdir(rprojdir)
            os.mkdir(iprojdir)
            
            # Read in the electrode locations and data collection sequence
            locs=pd.read_csv(locfile.strip(), index_col=0, delimiter='\t', names=['enum', 'x', 'y', 'z', 'flag'])
            seq=pd.read_csv(seqfile.strip(), header=7, index_col=0, delimiter=',', names=['mnum', 'A', 'B', 'M', 'N'])
            print('\nElectrode location file and data collection sequence file have been read.')
            
            ## GENERATE A SCRIPT FILE FROM THE DATA FILE ##
            dfs_impedance=[]
            dfs_phase=[]
            df=seq.copy()
            eoh_cnt=0
            line_cnt=0
            f = open(fil)
            lines=f.readlines() # read the data from the PSIP file
            f.close()
             
            # find out line where PSIP data starts
            for line in lines:
                line_cnt+=1
                if "***End_Of_Header***" in line:
                    eoh_cnt+=1
                if eoh_cnt==2:
                    break
             
            # read in PSIP data without header
            data = pd.read_csv(fil, skiprows=line_cnt+1)
            print('\nPSIP Data file has been read.')
             
            # Find the index and value of the frequency closest to that specified
            data = data.rename(columns={"Frequency[Hz]":"Freq"})
            freqs = data.Freq.unique()
            find1 = np.argmin(abs(freqs-thisfreq))
            pickedfreq = freqs[find1]
            
            # Clip the data to that collected at the specified frequency
            data2 = data.query('Freq==@freqs[@find1]')
            
            # Name the survey file like the data file
            SRV_name=fil.split('.')[0] + '_freq_1Hz.srv'      
             
            # add A B M N to each measurement
            data2 = data2.rename(columns={"Loop": "mnum"}).set_index('mnum')
            data2 = data2.merge(seq,left_index=True,right_index=True)
         
            # add standard deviations for each measurement
            data2['Imp_stdDev']=0.05*data2['Impedance[Ohms]'] + 0.01
            data2['Phase_stdDev']=0.001
            # use large std deviation where phase shift is positive
            data2.loc[data2['Phase_Shift[rad]'] >= 0, 'Phase_stdDev'] = 1
                 
            # open the output file for writing
            SRVout=open(SRV_name, 'w') #open file
            outputElecs() # write out electrodes
            SRVout.write(str(len(data2))+'\n') # write out data
            data2.to_csv(SRVout, sep='\t', columns=['A', 'B', 'M', 'N', 'Impedance[Ohms]', 'Imp_stdDev', 'Phase_Shift[rad]', 'Phase_stdDev'] , index=True, header=False, line_terminator='\n', float_format='%.5f')
            SRVout.close()
            print('\nSurvey file has been written.')
            
            ## GENERATE E4D INPUT FILE ##
            INPout = open('e4d.inp', 'w')
            INPout.write('ERTTank3\n')# Write run mode
            INPout.write(meshfile.split('.',1)[0].split('/')[-1]+'.1.node\n') # Write mesh configuration file
            INPout.write(SRV_name+'\n') # Write survey file
            INPout.write('average\n')# Write starting conductivity
            INPout.write(outopts.split('/')[-1])# Write output options file
            INPout.write(invopts.split('/')[-1])# Write inverse options file
            INPout.write('\nnone\n')# Write reference conductivity
            INPout.close()
            os.system('mv e4d.inp ./'+rprojdir)
            print('\nE4D Real Conductivity Input file has been written.')
            
            # Move everything to project directory
            os.system('cp ' +locfile.strip()+  ' ./' + rprojdir)
            os.system('cp ' +seqfile.strip()+  ' ./' + rprojdir)
            os.system('cp ' +meshfile.split('.',1)[0]+  '* ./' + rprojdir) # moves all meshfiles
            os.system('cp ' +outopts.strip()+  '* ./' + rprojdir)
            os.system('cp ' +invopts.strip()+  '* ./' + rprojdir)
            os.system('mv '+SRV_name+' ./'+rprojdir)
            os.system('mv '+fil+' ./'+rprojdir)
    
            print('\nFiles copied to project directory.')
            print('\nRunning E4D real conductivity inversion.')
        
            # CD to the real project directory and run E4D
            os.chdir(rprojdir)
            e4dcmd = mpipath+' -np ' +str(nproc)+' '+e4dpath
            realproc = sp.Popen(e4dcmd,shell=True) ## This needs a timeout limit (built in to slurm batch scripts)
            realproc.wait() ## Not sure this is working how you think it is
            #os.system('mpirun -np '+str(nproc)+' '+e4dpath)
            # Run the imaginary conductivity inversion
            print('\nE4D real conductivity inversion complete')
            
            # Read the final results from the real conductivity inversion
            print('\nReading results from real conductivity inversion')
            realres = glob.glob('sigma*')
            realres.sort(key=os.path.getmtime)
            realfin = 'sigma.'+str(len(realres)-1)      
            realsol = open(realfin)
            realhead = realsol.readline()
            realclist = realsol.readlines()
            realsol.close()
            nnodes = int(realhead.split()[0])
            chisq = float(realhead.split()[2])
            # chisq ="{:f}".format(chisq)
            
            # Print the results to the inverse results text file for px
            invrestxt = open('invres.list','w')
            invrestxt.write(str(len(realres))+'\n')
            for i in range(len(realres)):
                invrestxt.write(realres[i]+'\t'+str(i)+'\n')
            invrestxt.close()
            
            # Build real conductivity result images using px for VisIt
            xmf_fil = fil.split('.',1)[0]+'real'
            pxcmd = pxpath+' -f '+meshfile.split('.',1)[0].split('/')[-1] +' invres.list '+xmf_fil+' 0'
            os.system(pxcmd)
            
            # Write a VisIt script and call VisIt to generate an image of the real results
            viscrpt = open('visit.scr','w')
            viscrpt.write('OpenDatabase("'+xmf_fil+'.xmf")\n')
            viscrpt.write('AddPlot("Pseudocolor","Real_conductivity")\n')
            viscrpt.write('AddPlot("Mesh","1")\n')
            viscrpt.write('DrawPlots()\n')
            viscrpt.write('v = GetView3D()\n')
            viscrpt.write('v.viewNormal=(0,-1,0)\n')
            # viscrpt.write('v.focus=(0.1905, 0.0127, 0.01524)\n')
            viscrpt.write('v.viewUp=(0, 0, 1)\n')
            viscrpt.write('SetView3D(v)\n')
            # viscrpt.write('v.viewAngle=30\n')
            # viscrpt.write('v.parallelScale=0.244289\n')
            # viscrpt.write('v.nearPlane=-0.488579\n')
            # viscrpt.write('v.farPlane=0.488579\n')
            # viscrpt.write('v.imagePan=(0,0)\n')
            # viscrpt.write('v.imageZoom=1\n')
            # viscrpt.write('v.perspective=1\n')
            # viscrpt.write('v.eyeAngle=2\n')
            # viscrpt.write('v.centerOfRotationSet=0\n')
            # viscrpt.write('v.centerOfRotation=(0.1905, 0.0127, 0.1524)\n')
            # viscrpt.write('v.axis3DScaleFlag=0\n')
            # viscrpt.write('v.axis3DScales=(1, 1, 1)\n')
            # viscrpt.write('v.shear=(0, 0, 1)\n')
            # viscrpt.write('v.windowValid=1\n')
            viscrpt.write('s = SaveWindowAttributes()\n')
            viscrpt.write('s.format = s.PNG\n')
            viscrpt.write('s.fileName ="'+xmf_fil+'"\n')
            viscrpt.write('s.width, s.height = 1024,768\n')
            viscrpt.write('s.screenCapture = 0\n')
            viscrpt.write('s.outputToCurrentDirectory=1\n')
            viscrpt.write('SetSaveWindowAttributes(s)\n')
            viscrpt.write('SetTimeSliderState(TimeSliderGetNStates()-1)\n')
            viscrpt.write('SaveWindow()\n')
            viscrpt.write('exit()\n')
            viscrpt.close()
            os.system(visitpath+' -cli -nowin -s visit.scr')
            
            # Plot the data fit for the real inversion
            print('\nPlotting real conductivity data fit using MatPlotLib')
            outoptsfil = open(outopts.strip().split('/')[-1])
            dfil = outoptsfil.readlines()
            outoptsfil.close()
            df = open(dfil[1].strip())
            dfdat = pd.read_csv(df, index_col=0, delimiter='\s+', header=0,names=['datan', 'A', 'B', 'M', 'N', 'Data', 'Model'])
            df.close()
            mpl.figure()
            mpl.xlabel('Data Value')
            mpl.ylabel('Model Value')
            mpl.plot('Data','Model',data=dfdat,marker='s',mec='k',mfc='k',ms=15,linestyle='none',label='E4D Results')
            mpl.plot([np.min([dfdat.loc[:,'Data'],dfdat.loc[:,'Model']]),np.max([dfdat.loc[:,'Data'],dfdat.loc[:,'Model']])],\
                            [np.min([dfdat.loc[:,'Data'],dfdat.loc[:,'Model']]),np.max([dfdat.loc[:,'Data'],dfdat.loc[:,'Model']])],\
                            linestyle='dashed',color='k',lw=3,label='1:1')
            mpl.legend()
            mpl.savefig('RealDataFit.png',dpi=300)
            
            # Write the results out for starting conductivity of imaginary inversion
            print('\nUsing results to create starting file for imaginary inversion')
            imaginit = open('sigmai.0','w')
            imaginit.write(str(nnodes)+'\t'+str(2)+'\t'+f'{chisq}\n')
            for node in range(nnodes):
                realc = float(realclist[node].strip())
                imagc = float(realclist[node].strip())*0.05
                imaginit.write(f'{realc:.12f}'+'\t'+f'{imagc:.12f}\n')
                # imaginit.write("{:.12f}".format(realc)+'\t'+"{:.12f}".format(imagc)+'\n')
            imaginit.close()
            
            # Move or copy the files to the imaginary project directory
            os.system('mv sigmai.0 ../'+iprojdir)
            os.system('cp ' +locfile.strip()+  ' ../' + iprojdir)
            os.system('cp ' +seqfile.strip()+  ' ../' + iprojdir)
            os.system('cp ' +meshfile.split('.',1)[0]+  '* ../' + iprojdir) # moves all meshfiles
            os.system('cp ' +outopts.strip()+  '* ../' + iprojdir)
            os.system('cp ' +invopts.strip()+  '* ../' + iprojdir)
            os.system('cp '+SRV_name+' ../'+iprojdir)
            os.system('cp '+fil+' ../'+iprojdir)
            print('\nFiles copied to imaginary project directory.')
            
            # CD to imaginary directory
            os.chdir('../'+iprojdir)
            
            ## GENERATE E4D INPUT FILE ##
            # Note that the inverse options file is used for both real and imaginary inversions
            INPout = open('e4d.inp', 'w')
            INPout.write('SIPTank3\n')# Write run mode
            INPout.write(meshfile.split('.',1)[0].split('/')[-1]+'.1.node\n') # Write mesh configuration file
            INPout.write(SRV_name+'\n') # Write survey file
            INPout.write('sigmai.0\n')# Write starting conductivity
            INPout.write(outopts.split('/')[-1])# Write output options file
            INPout.write(invopts.split('/')[-1] + ' ' + invopts.split('/')[-1])# Write inverse options file 
            INPout.write('\nnone\n')# Write reference conductivity
            INPout.close()
            print('\nE4D Imaginary Conductivity Input file has been written.')
            print('\nRunning E4D imaginary conductivity inversion.')
        
            # CD to the imaginary project directory and run E4D
            e4dcmd = mpipath+' -np ' +str(nproc)+' '+e4dpath
            imagproc = sp.Popen(e4dcmd,shell=True) ## This needs a timeout limit (built in to slurm batch scripts)
            imagproc.wait()
            #os.system('mpirun -np '+str(nproc)+' '+e4dpath)
            # Run the imaginary conductivity inversion
            print('\nE4D imaginary conductivity inversion complete')
    
            # Read the final results from the real conductivity inversion
            print('\nReading results from imaginary conductivity inversion')
            realres = glob.glob('sigmai*')
            realres.sort(key=os.path.getmtime)
            
            # Print the results to the inverse results text file for px
            invrestxt = open('invres.list','w')
            invrestxt.write(str(len(realres))+'\n')
            for i in range(len(realres)):
                invrestxt.write(realres[i]+'\t'+str(i)+'\n')
            invrestxt.close()
            
            # Build real conductivity result images using px for VisIt
            xmf_fil = fil.split('.',1)[0]+'imag'
            pxcmd = pxpath+' -f '+meshfile.split('.',1)[0].split('/')[-1] +' invres.list '+xmf_fil+' 0'
            os.system(pxcmd)
            
            # Call VisIt to generate an image of the imaginary results
            # Write a VisIt script and call VisIt to generate an image of the real results
            viscrpt = open('visit.scr','w')
            viscrpt.write('OpenDatabase("'+xmf_fil+'.xmf")\n')
            viscrpt.write('AddPlot("Pseudocolor","Imag_conductivity")\n')
            viscrpt.write('AddPlot("Mesh","1")\n')
            viscrpt.write('DrawPlots()\n')
            viscrpt.write('v = GetView3D()\n')
            viscrpt.write('v.viewNormal=(0,-1,0)\n')
            # viscrpt.write('v.focus=(0.1905, 0.0127, 0.01524)\n')
            viscrpt.write('v.viewUp=(0, 0, 1)\n')
            viscrpt.write('SetView3D(v)\n')
            # viscrpt.write('v.viewAngle=30\n')
            # viscrpt.write('v.parallelScale=0.244289\n')
            # viscrpt.write('v.nearPlane=-0.488579\n')
            # viscrpt.write('v.farPlane=0.488579\n')
            # viscrpt.write('v.imagePan=(0,0)\n')
            # viscrpt.write('v.imageZoom=1\n')
            # viscrpt.write('v.perspective=1\n')
            # viscrpt.write('v.eyeAngle=2\n')
            # viscrpt.write('v.centerOfRotationSet=0\n')
            # viscrpt.write('v.centerOfRotation=(0.1905, 0.0127, 0.1524)\n')
            # viscrpt.write('v.axis3DScaleFlag=0\n')
            # viscrpt.write('v.axis3DScales=(1, 1, 1)\n')
            # viscrpt.write('v.shear=(0, 0, 1)\n')
            # viscrpt.write('v.windowValid=1\n')
            viscrpt.write('s = SaveWindowAttributes()\n')
            viscrpt.write('s.format = s.PNG\n')
            viscrpt.write('s.fileName ="'+xmf_fil+'"\n')
            viscrpt.write('s.width, s.height = 1024,768\n')
            viscrpt.write('s.screenCapture = 0\n')
            viscrpt.write('s.outputToCurrentDirectory=1\n')
            viscrpt.write('SetSaveWindowAttributes(s)\n')
            viscrpt.write('SetTimeSliderState(TimeSliderGetNStates()-1)\n')
            viscrpt.write('SaveWindow()\n')
            viscrpt.write('exit()\n')
            viscrpt.close()
            os.system(visitpath+' -cli -nowin -s visit.scr')
           
            # Plot the data fit for the imaginary inversion
            print('\nPlotting imaginary conductivity data fit using MatPlotLib')
            outoptsfil = open(outopts.strip().split('/')[-1])
            dfil = outoptsfil.readlines()
            outoptsfil.close()
            df = open(dfil[1].strip())
            dfdat = pd.read_csv(df, index_col=0, delimiter='\s+', header=0,names=['datan', 'A', 'B', 'M', 'N', 'RData', 'RModel','IData','IModel'])
            df.close()
            mpl.figure()
            mpl.xlabel('Data Value')
            mpl.ylabel('Model Value')
            mpl.plot('IData','IModel',data=dfdat,marker='s',mec='k',mfc='k',ms=15,linestyle='none',label='E4D Results')
            mpl.plot([np.min([dfdat.loc[:,'IData'],dfdat.loc[:,'IModel']]),np.max([dfdat.loc[:,'IData'],dfdat.loc[:,'IModel']])],\
                            [np.min([dfdat.loc[:,'IData'],dfdat.loc[:,'IModel']]),np.max([dfdat.loc[:,'IData'],dfdat.loc[:,'IModel']])],\
                            linestyle='dashed',color='k',lw=3,label='1:1')
            mpl.legend()
            mpl.savefig('ImagDataFit.png',dpi=300)
            
            # CD back to the launch directory to process the next data file
            os.chdir(ldir)
    time.sleep(10)