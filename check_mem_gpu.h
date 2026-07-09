// ROOT-free check_mem: host resident memory (/proc/self/statm) + GPU memory in
// use (cudaMemGetInfo). Drop-in for the mentor's check_mem.h on the standalone.
#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>
#include <cuda_runtime.h>

inline double check_mem(std::string s = "", bool verbose = true) {
   long dummy = 0, rss_pages = 0;
   std::ifstream f("/proc/self/statm");
   if (f) f >> dummy >> rss_pages;
   double rssMB = rss_pages * (sysconf(_SC_PAGESIZE) / 1024.0) / 1024.0;

   size_t gfree = 0, gtotal = 0;
   cudaMemGetInfo(&gfree, &gtotal);
   double gusedMB = (gtotal - gfree) / (1024.0 * 1024.0);

   if (verbose)
      printf("%s - host RSS = %8.3f MB, GPU used = %8.3f MB\n", s.c_str(), rssMB, gusedMB);
   return rssMB;
}
