# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra

.PHONY: all clean vload_batch

all: libhtp_deser_tap.so

libhtp_deser_tap.so: htp_deser_tap.c
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $< -ldl -pthread

vload_batch: vload_batch.c
ifndef QNN_SDK_ROOT
	$(error QNN_SDK_ROOT must point to your QAIRT SDK root containing include/QNN)
endif
	$(CC) $(CFLAGS) -I"$(QNN_SDK_ROOT)/include/QNN" -o $@ $< -ldl

clean:
	rm -f libhtp_deser_tap.so vload_batch
