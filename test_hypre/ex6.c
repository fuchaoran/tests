/*
   Example 5

   Interface:    Linear-Algebraic (IJ)

   Compile with: make ex5

   Sample run:   mpirun -np 4 ex5

   Description:  This example solves the 2-D Laplacian problem with zero boundary
                 conditions on an n x n grid.  The number of unknowns is N=n^2.
                 The standard 5-point stencil is used, and we solve for the
                 interior nodes only.

                 This example solves the same problem as Example 3.  Available
                 solvers are AMG, PCG, and PCG with AMG or Parasails
                 preconditioners.  */

#include <math.h>
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"

#include "vis.c"

int hypre_FlexGMRESModifyPCAMGExample(void *precond_data, int iterations,
                                      double rel_residual_norm);


int main (int argc, char *argv[])
{
   int i;
   int myid, num_procs;
   int N, n;

   int ilower, iupper;
   int local_size, extra;

   int solver_id;
   int vis, print_system;

   double h, h2;
   ilower = 0;
   iupper =59999;
   local_size = 60000;
   HYPRE_IJMatrix A;


   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   HYPRE_IJMatrixRead( "mat", MPI_COMM_WORLD,HYPRE_PARCSR, &A );
   HYPRE_IJMatrixPrint(A, "IJ.out.A2");



   /* Clean up */
   HYPRE_IJMatrixDestroy(A);


   /* Finalize MPI*/
   MPI_Finalize();

   return(0);
}

/*--------------------------------------------------------------------------
   hypre_FlexGMRESModifyPCAMGExample -

    This is an example (not recommended)
   of how we can modify things about AMG that
   affect the solve phase based on how FlexGMRES is doing...For
   another preconditioner it may make sense to modify the tolerance..

 *--------------------------------------------------------------------------*/

int hypre_FlexGMRESModifyPCAMGExample(void *precond_data, int iterations,
                                   double rel_residual_norm)
{


   if (rel_residual_norm > .1)
   {
      HYPRE_BoomerAMGSetNumSweeps(precond_data, 10);
   }
   else
   {
      HYPRE_BoomerAMGSetNumSweeps(precond_data, 1);
   }


   return 0;
}
