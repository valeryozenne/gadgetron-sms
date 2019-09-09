#ifndef GADGETRON_EPI_LIRYC_EXPORT_H_
#define GADGETRON_EPI_LIRYC_EXPORT_H_

#if defined (WIN32)
#if defined (__BUILD_GADGETRON_EPI_LIRYC__)
#define EXPORTGADGETS_EPI_LIRYC __declspec(dllexport)
#else
#define EXPORTGADGETS_EPI_LIRYC __declspec(dllimport)
#endif
#else
#define EXPORTGADGETS_EPI_LIRYC
#endif

#endif /* GADGETRON_EPI_EXPORT_H_ */
