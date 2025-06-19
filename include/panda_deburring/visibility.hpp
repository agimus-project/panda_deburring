#ifndef PANDA_DEBURRING__VISIBILITY_HPP_
#define PANDA_DEBURRING__VISIBILITY_HPP_

// This logic was borrowed (then namespaced) from the examples on the gcc wiki:
//     https://gcc.gnu.org/wiki/Visibility

// Define PANDA_DEBURRING_[EXPORT, IMPORT, LOCAL]
// based on the OS
#if defined _WIN32 || defined __CYGWIN__

#ifdef __GNUC__
#define PANDA_DEBURRING_EXPORT __attribute__((dllexport))
#define PANDA_DEBURRING_IMPORT __attribute__((dllimport))
#else
#define PANDA_DEBURRING_EXPORT __declspec(dllexport)
#define PANDA_DEBURRING_IMPORT __declspec(dllimport)
#endif

// All symbols are hidden by default in windows
#define PANDA_DEBURRING_LOCAL

#else  // defined _WIN32 || defined __CYGWIN__

#if __GNUC__ >= 4
#define PANDA_DEBURRING_EXPORT __attribute__((visibility("default")))
#define PANDA_DEBURRING_IMPORT __attribute__((visibility("default")))
#define PANDA_DEBURRING_LOCAL __attribute__((visibility("hidden")))
#else
#define PANDA_DEBURRING_EXPORT
#define PANDA_DEBURRING_IMPORT
#define PANDA_DEBURRING_LOCAL
#endif

#endif  // defined _WIN32 || defined __CYGWIN__

// Define PANDA_DEBURRING_[PUBLIC, PRIVATE] based the following
// definitions forwarded by the build system:
// - PANDA_DEBURRING_IS_SHARED (If the project is a shared lib)
// - PANDA_DEBURRING_EXPORT (If we are building it directly)
#ifdef PANDA_DEBURRING_IS_SHARED

// LFC lib is shared (.so)
#ifdef PANDA_DEBURRING_DO_EXPORT
// We are building the shared lib -> EXPORT symbols
#define PANDA_DEBURRING_PUBLIC PANDA_DEBURRING_EXPORT
#else
// We are linking to the shared lib -> IMPORT symbols
#define PANDA_DEBURRING_PUBLIC PANDA_DEBURRING_IMPORT
#endif

#define PANDA_DEBURRING_PRIVATE PANDA_DEBURRING_LOCAL

#else  // PANDA_DEBURRING_IS_SHARED

// LFC lib is static (.a)
#define PANDA_DEBURRING_PRIVATE
#define PANDA_DEBURRING_PUBLIC

#endif

#endif  // PANDA_DEBURRING__VISIBILITY_HPP_
