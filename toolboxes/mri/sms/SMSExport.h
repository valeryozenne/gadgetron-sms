/** \file       SMSExport.h
    \brief      Implement windows export/import for SMS toolbox
    \author     Souheil Inati
*/

#pragma once

#if defined (WIN32)
    #if defined (__BUILD_GADGETRON_SMS__)
        #define EXPORTSMS __declspec(dllexport)
    #else
        #define EXPORTSMS __declspec(dllimport)
    #endif
#else
    #define EXPORTSMS
#endif
