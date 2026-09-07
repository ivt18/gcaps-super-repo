# GCAPS Driver Implementation

This folder includes the implementation of GCAPS approach in Tegra driver.

> **Measurement instrumentation.** `ioctl_ctrl.c.patch` additionally records a
> structured `GCAPS_EV ts=... add=... rlupd=... elapsed_us=... preempted=...
> resumed=... elapsed_ns=...` entry per runlist-update ioctl. This is what
> [`gcaps_userspace/scripts/measure_preempt_overhead.py`](../gcaps_userspace/readme.md#measuring-preemption-overhead)
> uses to attribute per-preemption overhead and reconstruct suspend intervals.
> Rebuild and redeploy `nvgpu.ko` (steps below) to pick it up.
>
> **The record is stored, not printed.** It used to be emitted by two
> `pr_info()` calls placed *inside* the `cs_lock` critical section — inside the
> interval the record reports, and inside the interval every other GCAPS task
> blocks on. `printk()` formats into the log buffer and, if a console is
> registered for its level, writes the line out synchronously (milliseconds on
> a serial console), so that cost landed both in the reported ε and in the
> benchmark's response window. The ioctl now only stores the record — eight
> stores and one atomic increment, after `rt_mutex_unlock()` and after the
> closing `ktime_get()` — into a lock-free 8192-entry ring, drained out of band:
>
> ```bash
> cat /proc/gcaps_events        # "#" summary + one GCAPS_EV line per record
> : > /proc/gcaps_events        # reset the ring (root)
> ```
>
> 8192 records is over an order of magnitude more than the ~860 `GCAPS_EV` lines
> that fit in the default 128 KiB kernel log buffer, so a busy run no longer
> loses its early events. Load the module with `gcaps_ev_printk=1` to restore
> the legacy `dmesg` lines (`process <pid> elapsed time:` included) for
> debugging — but that puts `printk` back on the ioctl path, so do not measure
> with it set.

## Test Environments
- Nvidia Jetson Xavier NX
- L4T R35.2.1 with Jetpack 5.0.2
- CUDA 11.4

## Preparations
(The steps are mainly adopted from this guide: [Kernel Customization](https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3273/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/kernel_custom.html#).)

### Download the Kernel and Driver Source
1. Download the BSP source from: https://developer.nvidia.com/embedded/jetson-linux-r3521
2. Unzip the source
    ```bash
    # On PC
    cd <download directory>
    tar -xjf public_sources.tbz2
    cd Linux_for_Tegra/source/public
    tar -xjf kernel_src.tbz2
    ```

### Install the Cross-Compilation Tools
(The steps are mainly adopted from this guide: [Jetson Linux Toolchain](https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3273/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/xavier_toolchain.html).)
1. Install the prerequisites:
    ```bash
    # On PC
    sudo apt install build-essential bc
    ```
2. Download the [toolchain](http://releases.linaro.org/components/toolchain/binaries/7.3-2018.05/aarch64-linux-gnu/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu.tar.xz)
3. Extract 
    ```bash
    # On PC
    mkdir $HOME/l4t-gcc
    cd $HOME/l4t-gcc
    tar xf gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu.tar.xz
    ```

## Apply the Patches
1. Direct to `nvgpu` driver path
    ```bash
    # On PC
    cd <path to downloaded BSP source>/Linux_for_Tegra/source/public/kernel/nvgpu
    ```
2. Overwrite files with patches. The patch files can be found at gcaps-super-repo/gcaps_driver_patch.
    ```bash
    # On PC
    patch drivers/gpu/nvgpu/os/linux/ioctl_ctrl.c <path to file>/ioctl_ctrl.c.patch
    patch drivers/gpu/nvgpu/include/nvgpu/sched.h <path to file>/sched.h.patch
    patch drivers/gpu/nvgpu/os/linux/sched.c <path to file/sched.c.patch
    patch include/uapi/linux/nvgpu-ctrl.h <path to file>/nvgpu-ctrl.h
    ```

## Compilation and Installation
1. Cross compile on PC
    ```bash
    # On PC
    cd <path to downloaded BSP source>/Linux_for_Tegra/source/public/kernel/kernel-5.10

    # set up environment variables
    export CROSS_COMPILE=$HOME/l4t-gcc/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-
    export LOCALVERSION=-tegra
    export TEGRA_KERNEL_OUT=$PWD/kernel_out
    export INSTALL_MOD_PATH=$PWD/rootfs/mod

    rm -rf $INSTALL_MOD_PATH 2> /dev/null
    mkdir -p $INSTALL_MOD_PATH
    mkdir -p $TEGRA_KERNEL_OUT
    make ARCH=arm64 O=$TEGRA_KERNEL_OUT tegra_defconfig
    make ARCH=arm64 O=$TEGRA_KERNEL_OUT menuconfig

    make Image modules ARCH=arm64 O=$TEGRA_KERNEL_OUT -j8 &&
    make ARCH=arm64 O=$TEGRA_KERNEL_OUT modules_install
    ```
2. Copy three files from PC to the target platform
`$INSTALL_MOD_PATH/lib/modules/5.10.104-tegra/kernel/drivers/gpu/nvgpu/nvgpu.ko`
`$TEGRA_KERNEL_OUT/arch/arm64/boot/Image`
`<BSP source directory>/Linux_for_Tegra/source/public/kernel/nvgpu/include/uapi/linux/nvgpu-ctrl.h`
3. Install the changes on the target platform
    ```bash
    # On Jetson
    cd <path to copied files>
    sudo cp nvgpu.ko /lib/modules/5.10.104-tegra/kernel/drivers/gpu/nvgpu/nvgpu.ko
    sudo cp Image /boot/Image
    sudo cp nvgpu-ctrl.h /usr/src/linux-headers-5.10.104-tegra-ubuntu20.04_aarch64/nvgpu/include/uapi/linux/nvgpu-ctrl.h
    sync
    sudo reboot
    ```
    :red_circle: Note: for `nvgpu-ctrl.h`, remember to change the corresponding path in userspace [Makefile](../gcaps_userspace/Makefile#L34).

:smiley: **After all the above steps are done, now you can move forward and test GCAPS preemptive GPU approach with [GCAPS userspace implementation](../gcaps_userspace/readme.md).**