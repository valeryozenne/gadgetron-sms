#ifndef GADGETRON_MULTIBAND_EXPORT_H_
#define GADGETRON_MULTIBAND_EXPORT_H_

#if defined (WIN32)
#if defined (__BUILD_GADGETRON_MULTIBAND__)
#define EXPORTGADGETSMULTIBAND __declspec(dllexport)
#else
#define EXPORTGADGETSMULTIBAND __declspec(dllimport)
#endif
#else
#define EXPORTGADGETSMULTIBAND
#endif

#endif /* GADGETRON_MULTIBAND_EXPORT_H_ */
