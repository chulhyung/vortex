ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# Needs BOTH the in-core TCU (mode 0) and the DTCU (mode 1).
CONFIGS := $(if $(findstring -DVX_CFG_EXT_TCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TCU_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DTCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DTCU_ENABLE)
# NT=8 calibration: DTCU matrix array == one NT=8 in-core TCU (32 MACs/cycle).
CONFIGS += -DDTCU_MACS_PER_CYCLE=32

PROJECT := dtcu_pi0_t_fwd_k1

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -DFIXTURE_DIR='"$(SRC_DIR)/inputs"'

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

KERNEL_LIB := vortex2

include ../common.mk
