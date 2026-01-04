"""TensorRT engine runner for image embeddings."""
from __future__ import annotations

import logging
import os
from typing import Iterable, Tuple

import numpy as np


class TensorRTRunner:
    def __init__(
        self,
        engine_path: str,
        input_shape: Iterable[int],
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        if not engine_path:
            raise ValueError("engine_path is required")
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            import tensorrt as trt
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"TensorRT/pycuda import failed: {exc}") from exc

        self.cuda = cuda
        self.trt = trt
        self.logger = logger or logging.getLogger("trt_runner")
        self.input_shape = tuple(int(dim) for dim in input_shape)

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as handle, trt.Runtime(trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.use_v3 = hasattr(self.context, "execute_async_v3")

        if hasattr(self.engine, "num_io_tensors"):
            self.input_name, self.output_name = self._resolve_io_names()
            if hasattr(self.context, "set_input_shape"):
                self.context.set_input_shape(self.input_name, self.input_shape)
            input_shape = self._resolve_shape(self.context.get_tensor_shape(self.input_name), self.input_shape)
            output_shape = self._resolve_shape(self.context.get_tensor_shape(self.output_name), None)
            if output_shape is None:
                output_shape = self._resolve_shape(self.engine.get_tensor_shape(self.output_name), None)
            if output_shape is None:
                raise RuntimeError("Output shape is dynamic; engine must be built with static output.")
            self.input_shape = input_shape
            self.output_shape = output_shape
            self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
            self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
            self.input_idx = None
            self.output_idx = None
        else:
            self.input_idx = self._find_binding(is_input=True)
            self.output_idx = self._find_binding(is_input=False)
            try:
                self.context.set_binding_shape(self.input_idx, self.input_shape)
            except Exception:
                pass
            input_shape = self._resolve_shape(self.context.get_binding_shape(self.input_idx), self.input_shape)
            output_shape = self._resolve_shape(self.context.get_binding_shape(self.output_idx), None)
            if output_shape is None:
                output_shape = self._resolve_shape(self.engine.get_binding_shape(self.output_idx), None)
            if output_shape is None:
                raise RuntimeError("Output shape is dynamic; engine must be built with static output.")
            self.input_shape = input_shape
            self.output_shape = output_shape
            self.input_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_idx))
            self.output_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_idx))

        self.d_input = cuda.mem_alloc(trt.volume(self.input_shape) * np.dtype(self.input_dtype).itemsize)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=self.output_dtype)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        if self.input_idx is not None:
            self.bindings = [0] * self.engine.num_bindings
            self.bindings[self.input_idx] = int(self.d_input)
            self.bindings[self.output_idx] = int(self.d_output)
        else:
            self.bindings = None
            if hasattr(self.context, "set_tensor_address"):
                self.context.set_tensor_address(self.input_name, int(self.d_input))
                self.context.set_tensor_address(self.output_name, int(self.d_output))
        self.stream = cuda.Stream()

        self.logger.info(
            "TensorRT engine loaded: input=%s output=%s dtype=%s",
            self.input_shape,
            self.output_shape,
            self.input_dtype,
        )

    def _find_binding(self, *, is_input: bool) -> int:
        for idx in range(self.engine.num_bindings):
            if self.engine.binding_is_input(idx) == is_input:
                return idx
        raise RuntimeError("Failed to locate TensorRT bindings")

    def _resolve_io_names(self) -> tuple[str, str]:
        input_name = None
        output_name = None
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            mode = self.engine.get_tensor_mode(name)
            if mode == self.trt.TensorIOMode.INPUT:
                input_name = name
            elif mode == self.trt.TensorIOMode.OUTPUT:
                output_name = name
        if not input_name or not output_name:
            raise RuntimeError("Failed to resolve TensorRT input/output names")
        return input_name, output_name

    @staticmethod
    def _resolve_shape(shape: Tuple[int, ...], fallback: Tuple[int, ...] | None) -> Tuple[int, ...] | None:
        resolved = tuple(int(dim) for dim in shape)
        if any(dim < 0 for dim in resolved):
            return fallback
        return resolved

    def infer(self, input_array: np.ndarray) -> np.ndarray:
        if input_array is None:
            raise ValueError("input_array is required")
        arr = np.ascontiguousarray(input_array)
        if arr.shape != self.input_shape:
            raise ValueError(f"Input shape {arr.shape} does not match {self.input_shape}")
        if arr.dtype != self.input_dtype:
            arr = arr.astype(self.input_dtype)

        self.cuda.memcpy_htod_async(self.d_input, arr, self.stream)
        if self.use_v3:
            if hasattr(self.context, "set_tensor_address"):
                self.context.set_tensor_address(self.input_name, int(self.d_input))
                self.context.set_tensor_address(self.output_name, int(self.d_output))
            self.context.execute_async_v3(self.stream.handle)
        else:
            self.context.execute_async_v2(self.bindings, self.stream.handle)
        self.cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        output = np.array(self.h_output, copy=True).reshape(-1)
        return output.astype(np.float32, copy=False)
