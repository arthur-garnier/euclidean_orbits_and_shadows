    EUCLIDEAN_ORBITS_AND_SHADOWS-Python3

This is a package for Python 3.10, loaded with OpenCV 4.10 and Imageio 2.36 to plot orbits in Euclidean Reissner-Nordstrom (ERN) and Bertotti-Robinson (BR) spacetimes, as well as to drow (animated) shadows of the former, by backward ray-tracing. A classical (Lorentzian) Reissner-Nordstrom ray-tracer is included for comparison with the usual Lorentzian world.

The code is an analogue of the package knds_orbits_and_shadows (https://pypi.org/project/knds-orbits-and-shadows/) for classical Kerr-Newman-(anti) de Sitter black holes. 

---------------------------------------------------------------------------------------------------

First, install Python 3.10 (along with the libraries numpy, scipy, matplotlib, cmath, os, pickel) and its packages opencv-python 4.10 (https://pypi.org/project/opencv-python/) and imageio 2.36 (https://pypi.org/project/imageio/). The latter package is used to handle create gif files for shadows of black holes.

Alternatively, the package can be installed using pip, via the command "pip install elev_orbits_and_shadows"; see also https://pypi.org/project/elev-orbits-and-shadows/

Next, put the content of the present folder anywhere and in the examples.py file, change the first line to match the directory of the files (and images!).
Execute the file examples.py; it uses all the functions of the programs, so it should be a good indicator of the sanity of the package. It is divided in three parts: the first one tests the orbit and shadow display, the second one creates four gif files named 'comet_*.gif' depicting animated orbits and the third one creates the folders figureLorentzian_gif and figureEuclidean_gif, containing gif files, that represent the shadow of a Lorentzian (resp. Euclidean) Reissner-Nordstrom spacetime, with a background celestial sphere that moves diagonally. The full execution takes about one minute and a half on a 12-core 2.60 GHz CPU with 16 Go of RAM.

---------------------------------------------------------------------------------------------------

Description of the main functions of the package:



- orbit computes the trajectory of a test particle in an Euclidean or Lorentzian Reissner-Nordstrom (RN) spacetime.

The synthax is Vecc=orbit(Type,Mass,Charge,Tau,N,IniConds,lim), where Type is a string taking two values; if set to "Lorentzian", then a classical RN spacetime is considered, while a Euclidean one is taken when Type is set to "Euclidean". Next, Mass is the mass and Charge is the charge of the central body, appearing in the RN line element. The variable Tau is the maximal value of the Euclidean time at which the trajectory is computed and N is the number of discretization points. The vector IniConds records the initial datum of the geodesic, in Schwarzschild coordinates (and geometric units G=c=1), IniConds=(r0,theta0,phi0,\dot{r}0,\dot{theta}0,\dot{phi}0). Finally, the variable lim denotes the maximal radius at which the orbit is drawn.

The output is a vector Vecc containing the position (r,theta,phi) (in geometric units G=c=1) of the trajectory at each node k*Tau/N (0<k<n) of the discretization.



- orbit_BR computes the trajectory of a test particle in a BR spacetime.

The synthax is Vecc=orbit_BR(epsilon,Charge,mass,charge,Tau,N,IniConds,lim), where epsilon is the signature of the metric (epsilon=1 in the Euclidean case, and epsilon=-1 in the Lorentzian case), Charge is the constant Q of the spacetime, while mass and charge are the parameters m,q of the metric. The rest of the variables and the output have the same description and meaning as for the function orbit described above.



- shadow draws the shadow of an RN spacetime, with a standard image (jpeg, png...).

The synthax is shadow(Type,Mass,Charge,v,Image), where Type, Mass and Charge are the parameters of the metric (in geometric units) and v is the velocity of the particle when hitting the eye of the observer, considered as being the velocity at infinity. In the Type="Lorentzian" case, if the velocity v is set to v<1 (resp. v=1, v>1), then a timelike (resp. null, spacelike) geodesic is drawn in the spacetime
The variable Image is a string formed with the name (with extension) of the picture to transform (the file should be in the same folder as the .py files). The picture is encoded as a (Nx,Ny,3) hypermatrix using the library cv2.

The output is the computed picture, as an (internal) matplotlib figure.



- deflection depicts the total deflection angle of orbits coming from infinity, to the eye of an observer, for an RN spacetime.

The synthax is deflection(Type,Mass,Charge,v,N), where, again, Type, Mass and Charge are the parameters of the metric (in geometric units) and v is the velocity of the orbit hitting the observer, again viewed is the velocity at infinity. The variable N is the resolution of the desired picture.

The output is a matplotlib figure showing the deflections on an N x N image, along with a colorbar: the color of a pixel represents the value of the total deflection angle of the corresponding orbit.



- The gif creating routine consists of four functions 'shadow4gif', 'make_gif', 'DatFile4gif' and 'make_gif_with_DatFile' that are designed to create gif files depicting the shadow of a black hole with a moving celestial sphere.

The first function (shadow4gif) is an non-display analogue of the program shadow described above; this is an auxiliary function. It is called as shadow4gif(Type,Mass,Charge,v,Image_matrix), with Type,Mass,Charge,v as above. The variable Image_matrix is a hypermatrix of size (Nx,Ny,3) containing the BGR values of pixels of an image with resolution (Nx,Ny).
The main function (make_gif) has synthax make_gif(Nimages,Name,Image,Resol,Shifts,Direction,FPS,Type,Mass,Charge,v), with Type,Mass,Charge,v as before.
The variable Nimages is the number of images for the gif animation.
The variable Name (a string) is the name of the folder that will be created by the program and containing the gif file.
The variable Image is the background image to use for the celestial sphere, as above.
The variable Resol is a list with two integers representing the desired resolution for the gif.
The variable Shifts is a list with three entries: the first two correspond to the respective horizontal and vertical shifts (in number of pixels) defining the corner of the starting image. The third entry is a coefficient defining the frame rate of the animation, in number of pixels (i.e. at each step, the portion of the image is shifted by this number of pixels: the higher this number, the lower the frame rate.)
The variable Direction can take eight values: when set to "h+" (resp. to "h-", "v+", "v-", "d1+", "d1-", "d2+", "d2-") the screen moves horizontally from left to right (resp. horizontally from right to left, vertically from bottom to top, vertically from top to bottom, diagonally from bottom-left to top-right, diagonally from top-right to bottom-left, diagonally from top-left to bottom-right, diagonally from bottom-right to top-left). Please note that it is the screen (celestial sphere) that moves, and not the camera: this is important when the black hole is not spherically symmetric.
The variable FPS defines the number of frames per second for the gif animation.

The output is a gif file, created in the new folder Name_gif.


The other two functions are made to create several gifs out of a single set of data, allowing to call the heavy function shadow only once.
More precisely, the function DatFile4gif is called as DatFile4gif(Resol,Type,Mass,Charge,v), with each variable having the same meaning as above. The program creates the new folder 'dat_files' (if it doesn't exist already) and puts there a .dat file, named file_Resol_Type_Mass_Charge_v.dat. This file contains all the variables needed to create any gif that could be made using a command of the form make_gif(-,-,-,Resol,-,-,-,Type,Mass,Charge,v). Basically, the program stores the hypermatrix obtained with the function shadow4gif, applied to a specific hypermatrix Image_matrix of the appropriate size, encoded as a permutation of its pixels. The same permutation can then be applied to any other image of the same size, without having to call shadow again.
The other function make_gif_with_DatFile has the same synthax and output as make_gif. But instead of calling the program shadow, this function looks for a .dat file with appropriate parameters inside the folder 'dat_files' to render the images. If no such file is found, an error is returned and the user should first use the function DatFile4gif to create it.



---------------------------------------------------------------------------------------------------

For more details on the equations and modelization, the reader is refered to the article by the author, available at https://link.springer.com/article/10.1140/epjc/s10052-024-12719-4
For any question, suggestion, commentary, remark, the user is invited to contact the author by email at arthur.garnier[at]math[dot]cnrs[dot]fr.