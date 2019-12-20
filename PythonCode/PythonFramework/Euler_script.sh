# More infos at https://scicomp.ethz.ch/wiki/Fenics
git clone https://github.com/valentinjacot/differentiation_framework

cd differentiation_framework/PythonCode/PythonFramework/

module load new gcc/4.8.2 open_mpi/1.6.5 boost/1.59.0_py2.7.9 mpfr/3.1.2_gmp6 qt/4.8.4 netcdf/4.3.2 eigen/3.2.1 swig/3.0.5 suitesparse/4.4.4 fenics/1.6.0

cp ../Dependencies/functionFactory.py .

bsub -n 1 -W 4:00 -R "rusage[mem=2048]" python ./Framework.py

# Then remove the plotting commands and save the function u into a XML file with the following command:
File('u.xml') << u
u

# This XML file you can then load in jupyter with:

N=120
mesh = UnitSquareMesh(N, N)
Vhat = FunctionSpace(mesh, 'P', 1)
u = Function(Vhat,'../PythonFramework/u.xml')
plot(u)
