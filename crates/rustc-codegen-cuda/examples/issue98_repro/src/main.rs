/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Repro for issue #98: f32::exp() in a kernel pulls in libdevice
//! (`__nv_expf`), which auto-switches the pipeline into NVVM IR mode.
//! The emitted opaque-pointer (LLVM 20 dialect) NVVM IR is rejected by
//! libNVVM with "parse expected type" when the runtime arch is
//! pre-Blackwell (e.g. sm_86).

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn swiglu_libdevice(x: &[f32], y: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let xi = x[i];
            let yi = y[i];
            let sigmoid = 1.0f32 / (1.0f32 + (-xi).exp());
            *out_elem = xi * sigmoid * yi;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let host_x = vec![-2.0f32; 4];
    let host_y = vec![1.0f32; 4];
    let dev_x = DeviceBuffer::from_host(&stream, &host_x)?;
    let dev_y = DeviceBuffer::from_host(&stream, &host_y)?;
    let mut dev_out = DeviceBuffer::from_host(&stream, &[0.0f32; 4])?;

    module.swiglu_libdevice(
        &stream,
        LaunchConfig::for_num_elems(4),
        &dev_x,
        &dev_y,
        &mut dev_out,
    )?;

    let actual = dev_out.to_host_vec(&stream)?;
    let expected: Vec<f32> = host_x
        .iter()
        .zip(host_y.iter())
        .map(|(&x, &y)| {
            let s = 1.0f32 / (1.0f32 + (-x).exp());
            x * s * y
        })
        .collect();
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-5);
    }
    println!("PASS");
    Ok(())
}
