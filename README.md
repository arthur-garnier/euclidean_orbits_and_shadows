    EUCLIDEAN_ORBITS_AND_SHADOWS

This is a code for Python 3.10, to plot orbits in Euclidean Reissner-Nordstrom (ERN) and Bertotti-Robinson (BR) spacetimes, as well as to drow shadows of the former, by backward ray-tracing. A classical Reissner-Nordstrom ray-tracer is included for comparison with the usual Lorentzian world.

---------------------------------------------------------------------------------------------------

This code needs the following Python packages: numpy, scipy, matplotlib, cmath and cv2 (OpenCV). To manage file directories, the test file example.py also uses the package os.

Once these are installed, put the content of the present folder anywhere and in the example.py file, change the first line to match the directory of the files (and images!).
Execute the file example.py; it displays all the functions of the program, so it should be a good indicator of the sanity of the code. The full execution takes about 15 seconds on an 8-core 3 GHz CPU with 16 Go of RAM.

---------------------------------------------------------------------------------------------------

Description of the functions:


- orbit_ERN computes the trajectory of a test particle in an ERN spacetime.

The synthax is orbit_ERN(Mass,Charge,Tau,N,IniConds,lim), where Mass is the mass and Charge is the charge of the central body, appearing in the ERN line element. The variable Tau is the maximal value of the Euclidean time at which the trajectory is computed and N is the number of discretization points. The vector IniConds records the initial datum of the geodesic, in Schwarzschild coordinates (and geometric units G=c=1), IniConds=(r0,theta0,phi0,\dot{r}0,\dot{theta}0,\dot{phi}0). Finally, the variable lim denotes the maximal radius at which the orbit is drawn.

The output is a matplotlib figure plotting the orbit, along with the sphere of radius the outer horizon (in the sub-extremal case only), in 3D.


- orbit_BR computes the trajectory of a test particle in a BR spacetime.

The synthax is orbit_BR(epsilon,Charge,mass,charge,Tau,N,IniConds,lim), where epsilon is the signature of the metric (epsilon=1 in the Euclidean case, and epsilon=-1 in the Lorentzian case), Charge is the constant Q of the spacetime, while mass and charge are the parameters m,q of the metric. The rest of the variables and the output have the same description and meaning as for the command orbit_ERN described above.


- shadow_ERN draws the shadow of an ERN spacetime, with a standard image (jpeg, png...).

The synthax is shadow_ERN(Mass,Charge,v,Image) where, again, Mass and Charge are the parameters of the metric and v is the velocity of the particle when hitting the eye of the observer, considered as being the velocity at infinity.
The variable Image is a string formed with the name (with extension) of the picture to transform (the file should be in the same folder as the .py files). The picture is encoded as a 3 x M x N hypermatrix using the library cv2.

The output is the computed picture, as an (internal) matplotlib figure.


- shadow_LRN draws the shadow of a usual (Lorentzian) Reissner-Nordstrom spacetime, with a standard image (jpeg, png...). This is included for comparison with the Euclidean framework.

The synthax and output are the same as for shadow_ERN. If the velocity v is set to v<1 (resp. v=1, v>1), then a timelike (resp. null, spacelike) geodesic is drawn in the spacetime.


- deflection_ERN depicts the total deflection angle of orbits coming from infinity, to the eye of an observer, for an ERN spacetime.

The synthax is deflection_ERN(Mass,Charge,v,N), where Mass and Charge are the parameters of the metric and v is the velocity of the orbit hitting the observer, again viewed is the velocity at infinity. The variable N is the resolution of the desired picture.

The output is a matplotlib figure showing the deflections on an N x N image, along with a colorbar: the color of a pixel represents the value of the total deflection angle of the corresponding orbit.


- deflection_LRN depicts the total deflection angle of orbits coming from infinity, to the eye of an observer, for a usual (Lorentzian) Reissner-Nordstrom metric. This is included for comparison with the Euclidean framework.

The synthax and output are the same as for deflection_ERN and the same remark on the velocity as in shadow_LRN holds.


---------------------------------------------------------------------------------------------------

For more details on the equations and modelization, the reader is refered to the article by the author, available at ???
For any question, suggestion, commentary, remark, the user is invited to contact the author by email at arthur.garnier[at]math[dot]cnrs[dot]fr.