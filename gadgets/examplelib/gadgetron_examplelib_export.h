#ifndef GADGETRON_EXAMPLELIB_EXPORT_H_
#define GADGETRON_EXAMPLELIB_EXPORT_H_

#if defined (WIN32)
#if defined (__BUILD_GADGETRON_EXAMPLELIB__)
#define EXPORTGADGETSEXAMPLELIB __declspec(dllexport)
#else
#define EXPORTGADGETSEXAMPLELIB __declspec(dllimport)
#endif
#else
#define EXPORTGADGETSEXAMPLELIB
#endif

#endif /* EXAMPLE_EXPORT_H_ */

