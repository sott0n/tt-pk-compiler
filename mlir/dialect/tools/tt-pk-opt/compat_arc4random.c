/* Shim for a prebuilt LLVM linked on a newer glibc (>= 2.38) when the host
 * glibc is 2.35. Covers arc4random (added in 2.36) and the ISO C23
 * strto{,u}ll overloads (switched on in 2.38). Each shim defers to the
 * nearest equivalent on the host libc. */
#include <errno.h>
#include <stdlib.h>

#if defined(__GLIBC__) && (__GLIBC__ < 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ < 36))
unsigned int arc4random(void) {
  return (unsigned int)(((unsigned long)rand() << 16) ^ (unsigned long)rand());
}
#endif

#if defined(__GLIBC__) && (__GLIBC__ < 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ < 38))
long long __isoc23_strtoll(const char *nptr, char **endptr, int base) {
  return strtoll(nptr, endptr, base);
}

unsigned long long __isoc23_strtoull(const char *nptr, char **endptr,
                                     int base) {
  return strtoull(nptr, endptr, base);
}

long __isoc23_strtol(const char *nptr, char **endptr, int base) {
  return strtol(nptr, endptr, base);
}

unsigned long __isoc23_strtoul(const char *nptr, char **endptr, int base) {
  return strtoul(nptr, endptr, base);
}
#endif
