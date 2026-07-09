#include "TSystem.h"

double check_mem(std::string s = "", bool verbose=true){

   ProcInfo_t p;
   if (verbose) printf("%s - ",s.c_str());
   gSystem->GetProcInfo(&p);
   if (verbose)
     printf(" Rmem = %8.3f MB, Vmem = %8.f3 MB  \n",
	    p.fMemResident * 1e-3,  /// Real memory to watch for leaks                                              
	    p.fMemVirtual  * 1e-3
	    );
   return p.fMemResident * 1e-3;
}
