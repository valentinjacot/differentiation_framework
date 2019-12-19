#! /usr/bin/env python
#
from fenics import *
from mshr import *

def heat_explicit ( ):

#*****************************************************************************80
#
## heat_explicit, 2D heat equation on rectangle with interior hole.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    22 October 2018
#
#  Author:
#
#    John Burkardt
#
  import matplotlib.pyplot as plt
#
#  Define the domain.
#
  circle_x = 0.5
  circle_y = 0.5
  circle_r = 0.25

  domain = Rectangle(Point(-1.0,-1.0), Point(1.,1.)) \
        - Circle(Point(circle_x,circle_y),circle_r)
#
#  Mesh the domain.
#
  mesh = generate_mesh ( domain, 10 )
#
#  Plot the mesh.
#
  plot ( mesh, title = 'heat_explicit Mesh' )
  filename = 'heat_explicit_mesh.png'
  plt.savefig ( filename )
  print ( '  Graphics saved as "%s"' % ( filename ) )
  plt.close ( )
#
#  Define the function space.
#
  V = FunctionSpace ( mesh, "Lagrange", 1 )
#
#  Define the boundary conditions.
#  These could depend on time as well as space.
#
  rect_u = 10.0

  def rect_on ( x, on_boundary ):
    return ( on_boundary and ( (abs( x[0]-1.0 ) < 1.e-8) or \
                               (abs( x[0]+1.0 ) < 1.e-8) or \
                               (abs( x[1]-1.0 ) < 1.e-8) or \
                               (abs( x[1]+1.0 ) < 1.e-8) ) )

  rect_bc = DirichletBC ( V, rect_u, rect_on )

  circle_u = 100.0
  def circle_on ( x, on_boundary ):
    r = sqrt ( ( x[0] - circle_x ) ** 2 + ( x[1] - circle_y ) ** 2 )
    return ( on_boundary and ( r < circle_r * 1.1 ) )

  circle_bc = DirichletBC ( V, circle_u, circle_on )
#
  bc = [ rect_bc, circle_bc ]
#
#  Define the trial functions (u) and test functions (v).
#
  u = TrialFunction ( V )
  v = TestFunction ( V )
#
#  UOLD must be a Function.
#
  uold = Function ( V )
#
#  Define the form.
#  The form Auvt seems to be much more picky than the right hand side fuvt.
#  I can't seem include a /dt divisor on Auvt, for instance.
#
  Auvt = inner ( u, v ) * dx
#
#  The diffusivity is a constant.
#
  k = Constant ( 1.0 )
#
#  The source term is zero. 
#
  f = Expression ( "0.0", degree = 10 )
#
#  Define time things.
#
  t_init = 0.0
  t_final = 0.05
  t_num = 1000
  dt = ( t_final - t_init ) / t_num
#
#  Create U_INIT.
#
  u_init = Expression ( "40.0", degree = 10 )
#
#  U <-- the initial condition.
#
#  You have a choice of "project" or "interpolate".
#
# u = project ( u_init, V )
#
  u = interpolate ( u_init, V )
#
#  T <-- the initial time.
#
  t = t_init
#
#  Time loop.
#
  for j in range ( 0, t_num + 1 ):

    if ( j % 100 == 0 ):
      label = 'Time = %g' % ( t )
      plot ( u, title = label )
      filename = 'heat_explicit_solution_%d.png' % ( j )
      plt.savefig ( filename )
      print ( '  Graphics saved as "%s"' % ( filename ) )
      plt.close ( )
#
#  Copy UOLD function <--- U function
#
    uold.assign ( u )
#
#  Update fvt, the form for the right hand side.
#
    fvt = inner ( uold, v ) * dx \
       - dt * k * inner ( grad ( uold ), grad ( v ) ) * dx \
       + dt * inner ( f, v ) * dx
#
#  U <-- solution of the variational problem Auvt = fvt.
#
    solve ( Auvt == fvt, u, bc )
#
#  T <-- T + DT
#
    t = t + dt
#
#  Terminate.
#
  return

def heat_explicit_test ( ):

#*****************************************************************************80
#
## heat_explicit_test tests heat_explicit.
#
#  Modified:
#
#    23 October 2018
#
#  Author:
#
#    John Burkardt
#
  import time

  print ( time.ctime ( time.time() ) )
#
#  Report level = only warnings or higher.
#
  level = 30
  set_log_level ( level )

  print ( '' )
  print ( 'heat_explicit_test:' )
  print ( '  FENICS/Python version' )
  print ( '  Time-dependent heat equation.' )

  heat_explicit ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'heat_explicit_test:' )
  print ( '  Normal end of execution.' )
  print ( '' )
  print ( time.ctime ( time.time() ) )
  return

if ( __name__ == '__main__' ):

  heat_explicit_test ( )
