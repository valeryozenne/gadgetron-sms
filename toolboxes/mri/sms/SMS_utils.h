#pragma once
#ifndef SMS_UTILS_H
#define SMS_UTILS_H

#include "Gadget.h"
#include "hoNDArray.h"

typedef struct s_EPICorrection
{
    AcquisitionHeader hdr;
    hoNDArray< std::complex<float> > correction;
}t_EPICorrection;
