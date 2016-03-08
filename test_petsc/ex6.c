/*
 Implementation of Conjugate Gradient in parallel
 */
static char help[] = "Test file for Conjugate Gradient\n";
#include <petscksp.h>
#include "petsc.h"
#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{

  /* Input matrix market file and output PETSc binary file */
  char        inputFile[128],outputFile[128],buf[128];

  /* number rows, columns, non zeros etc */
  int         m,n,nnz,col,row;

  /*We compute no of nozeros per row for PETSc Mat object pre-allocation*/  
  int *nnzPtr;
  /*Maximum nonzero in nay row */
  int maxNNZperRow=0;
  /*Row number containing max non zero elements */
  int maxRowNum = 0;
  /*Just no of comments that will be ignore during successive read of file */
  int numComments=0;

  /* This is  variable of type double */
  PetscScalar val;

  /*File handle for read and write*/
  FILE*       file;
  /*File handle for writing nonzero elements distribution per row */
  FILE 	      *fileRowDist;

  /*PETSc Viewer is used for writing PETSc Mat object in binary format */
   PetscViewer view;

  KSP                ksp;
  PC                 pc;
  Mat                A;
  Vec                X,B;
  MPI_Comm           comm;
  KSPConvergedReason reason;
  PetscInt           i,its;
  PetscErrorCode     ierr;
  PetscScalar        v;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-options_left",NULL);CHKERRQ(ierr);
  comm = MPI_COMM_SELF;
 /*Get name of matrix market file from command line options and Open file*/
  ierr = PetscOptionsGetString(PETSC_NULL,"-fin",inputFile,127,PETSC_NULL);
  ierr = PetscFOpen(PETSC_COMM_SELF,inputFile,"r",&file);

  /* Just count the comment lines in the file */
  while(1)
  {
  	fgets(buf,128,file);
        /*If line starts with %, its a comment */
        if(buf[0] == '%')
	{
	   numComments++; 
	}
	else
	{
	   /*Set Pointer to Start of File */
	   fseek(file, 0, SEEK_SET );
           int num = numComments;

	   /* and just move pointer to the entry in the file which indicates row nums, col nums and non zero elements */
	   while(num--)
	   	fgets(buf,128,file);
	   break;
	}
  }

  /*Reads size of sparse matrix from matrix market file */
  fscanf(file,"%d %d %d\n",&m,&n,&nnz);
  printf ("ROWS = %d, COLUMNS = %d, NO OF NON-ZEROS = %d\n",m,n,nnz);

  /*Now we will calculate non zero elelments distribution per row */
  nnzPtr = (int *) calloc (sizeof(int),  m);

  /*This is similar to calculate histogram or frequency of elements in the array */
  for (i=0; !feof(file); i++) 
  {
  	  fscanf(file,"%d %d %le\n",&row,&col,&val);
	  row = row-1; col = col-1 ;
	  nnzPtr[row]++;
  }

  printf("\n ROW DISTRIBUTION CALCULATED....WRITING TO THE FILE..!");
  fflush(stdout);

  /*Write row distribution to the file ROW_STR.dat */
  fileRowDist =  fopen ("ROW_DISTR.dat", "w");
  for (i=0; i< m; i++)
  {
     fprintf(fileRowDist, "%d\t %d\n", i, nnzPtr[i]);
     /*Find max num of of nonzero for any row of the matrix and that row number */
     if( maxNNZperRow < nnzPtr[i] )
     {	  /*store max nonzero for any row*/
	  maxNNZperRow =  nnzPtr[i];
	  /*row that contains max non zero elements*/
          maxRowNum = i; 
          
     }
  }
  /*Close File */
  fclose(fileRowDist);

  printf("\n MAX NONZERO FOR ANY ROW ARE : %d & ROW NUM IS : %d", maxNNZperRow, maxRowNum );
  
  /* Again set the file pointer the fist data record in matrix market file*
   * Note that we can directly move ponts with fseek, but as this is text file 
   * we are simple reading line by line
   */
  fseek(file, 0, SEEK_SET );
  numComments++;
  while(numComments--)
	fgets(buf,128,file);


  /* Its important to pre-allocate memory by passing max non zero for any row in the matrix */
  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,m,n,maxNNZperRow,PETSC_NULL,&A);
  /* OR we can also pass row distribution of nozero elements for every row */
  /* ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,m,n,0,nnzPtr,&pMat);*/

  /*Now Set matrix elements values form matrix market file */
  for (i=0; i<nnz; i++) 
  {
	    /*Read matrix element from matrix market file*/
	    fscanf(file,"%d %d %le\n",&row,&col,&val);
            /*In matrix market format, rows and columns starts from 1 */
	    row = row-1; col = col-1 ;
	    /* For every non zero element,insert that value at row,col position */	
	    ierr = MatSetValues(A,1,&row,1,&col,&val,INSERT_VALUES);
  }
  fclose(file);
  /*Matrix Read Complete */
  ierr = PetscPrintf(PETSC_COMM_SELF,"\n MATRIX READ...DONE!");

  /*Now assemeble the matrix */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
 
  /* Now open output file for writing into PETSc Binary FOrmat*/
  ierr = PetscOptionsGetString(PETSC_NULL,"-fout",outputFile,127,PETSC_NULL);CHKERRQ(ierr);
  /*With the PETSc Viewer write output to File*/
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outputFile,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
  /*Matview will dump the Mat object to binary file */
  ierr = MatView(A,view);CHKERRQ(ierr);
 
  /*
   * Construct the vector
   * and a suitable rhs / initial guess
   */
  ierr = VecCreateSeq(comm,m,&B);CHKERRQ(ierr);
  ierr = VecDuplicate(B,&X);CHKERRQ(ierr);


  for (i=0; i<m; i++) {
    v = 1;
    ierr = VecSetValues(B,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    v = 0;
    ierr = VecSetValues(X,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(B);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(B);CHKERRQ(ierr);
  //printf("\nThe Kershaw matrix:\n\n"); MatView(A,0);

  /*
   * A Conjugate Gradient method
   * with ILU(0) preconditioning
   */
  ierr = KSPCreate(comm,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);

  ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);

  /*
   * ILU preconditioner;
   * The iterative method will break down unless you comment in the SetShift
   * line below, or use the -pc_factor_shift_positive_definite option.
   * Run the code twice: once as given to see the negative pivot and the
   * divergence behaviour, then comment in the Shift line, or add the
   * command line option, and see that the pivots are all positive and
   * the method converges.
   */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  /* ierr = PCFactorSetShiftType(prec,MAT_SHIFT_POSITIVE_DEFINITE);CHKERRQ(ierr); */

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);

  

  /*
   * Solve the system;
   * without the shift this will diverge with
   * an indefinite preconditioner
   */
  ierr = KSPSolve(ksp,B,X);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
 
    ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
    printf("\nConvergence in %d iterations.\n",(int)its);
    //VecView(X,PETSC_VIEWER_STDOUT_WORLD);
  
  printf("\n");

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&B);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  PetscFinalize();
  PetscFunctionReturn(0);
}
