# Predicting Pulsar Stars with C++ and LibTorch

This project is a C++ application that uses the LibTorch library (the C++ frontend for PyTorch) to predict whether a celestial object is a pulsar star based on a given dataset.

## Dataset

The dataset used for this project contains eight continuous variables and a single target class variable.

**Features:**
1.  Mean of the integrated profile
2.  Standard deviation of the integrated profile
3.  Excess kurtosis of the integrated profile
4.  Skewness of the integrated profile
5.  Mean of the DM-SNR curve
6.  Standard deviation of the DM-SNR curve
7.  Excess kurtosis of the DM-SNR curve
8.  Skewness of the DM-SNR curve

**Target Class:**
9.  `target_class`: `1` for a pulsar star, `0` otherwise.

---

## Prerequisites

Before you begin, ensure you have the following installed:
*   **CMake**: For building the project.
*   **A C++ Compiler**: Like Clang (on macOS) or GCC (on Linux).
*   **LibTorch**: The PyTorch C++ library. You can download it from the [official PyTorch website](https://pytorch.org/get-started/locally/).
*   **(macOS only) Homebrew**: For installing dependencies.

---

## Building the Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/unrealJuanpa/PredictingPulsarStars.git
    cd PredictingPulsarStars
    ```

2.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```

3.  **Run CMake:**
    Point CMake to the location of your unzipped LibTorch library.
    ```bash
    cmake -DCMAKE_PREFIX_PATH=/path/to/your/libtorch ..
    ```
    *Replace `/path/to/your/libtorch` with the actual absolute path to your LibTorch folder.*

4.  **Compile the project:**
    ```bash
    make
    ```
    This will create an executable named `main` inside the `build` directory.

---

## Running the Application

From within the `build` directory, simply run the executable:
```bash
./main
```

---

## Troubleshooting on macOS

Running C++ applications with external dynamic libraries on macOS (especially Apple Silicon) can sometimes be tricky due to the operating system's security policies (Gatekeeper). Here are solutions to the most common errors.

### Error 1: `library load disallowed by system policy` or `Trying to load an unsigned library`

If you see an error message similar to this:
```
dyld[...]: Library not loaded: @rpath/libc10.dylib
...
Reason: ... not valid for use in process: library load disallowed by system policy
```
or
```
Reason: ... Trying to load an unsigned library
```

This happens because macOS requires downloaded libraries to be properly signed. The fix is to re-sign all the dynamic libraries (`.dylib` files) in LibTorch with an ad-hoc signature.

**Solution:** Run the following command in your terminal. It will find all `.dylib` files in your LibTorch installation and sign them.

```bash
find /path/to/your/libtorch/lib -name "*.dylib" -exec codesign -f -s - {} \;
```
*Remember to replace `/path/to/your/libtorch` with the correct path.*

### Error 2: `Library not loaded: /opt/homebrew/opt/libomp/lib/libomp.dylib`

If you encounter this error:
```
dyld[...]: Library not loaded: /opt/homebrew/opt/libomp/lib/libomp.dylib
...
Reason: tried: '/opt/homebrew/opt/libomp/lib/libomp.dylib' (no such file)
```

This means LibTorch depends on the OpenMP library (`libomp`), but it wasn't found in the expected Homebrew path.

**Solution:** Install `libomp` using Homebrew.

```bash
brew install libomp
```

After running this, try executing your program again.
