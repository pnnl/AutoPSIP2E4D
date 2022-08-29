Authors: Adam R. Mangel, Judy Robinson, Tim C. Johnson
Affiliation: Pacific Northwest National Laboratory

Description:
This code will read data from the PSIP frequency domain complex resistivity instrument from Ontash and Ermac.
The data will be re-ordered in a format for input into E4D.
The code will call E4D in the appropriate inverse mode and associated parameters.
Once the inversion is complete, the code will read the results and plot.
The code will save several images and place them in a folder where the user can locate the data for interpretation.

The code consists of several python scripts that run on two separate computers; a host and a server.

The host computer is a local Windows machine that runs the PSIP data acquisition software:

EstablishSSH.py establishes an SSH connection between the host computer and the server.
AutoPSIP_push.py sends new PSIP data files from the host computer to the server once they are collected.

The server computer is a multicore Linux machine capable of high-performance computing:

AutoPSIP2E4D.py performs the PSIP data analysis using the options specified in AutoPSIP_inputfile.py.

More information is available in the AutoPSIP2E4D user guide.