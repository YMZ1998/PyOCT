"""Realistic CPU simulation of spectral-domain OCT B-scan acquisition."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


N_SPECTRAL = 2048
N_ASCANS = 256
LAMBDA_MIN_NM = 1250.0
LAMBDA_MAX_NM = 1350.0
RNG_SEED = 20260904


def camera_wavenumbers():
    """Mildly nonlinear spectrometer sampling grid in rad/m."""
    pixel = np.linspace(-1.0, 1.0, N_SPECTRAL)
    wavelength_center = (LAMBDA_MIN_NM + LAMBDA_MAX_NM) / 2
    wavelength_half_span = (LAMBDA_MAX_NM - LAMBDA_MIN_NM) / 2
    wavelength_nm = (
        wavelength_center + wavelength_half_span * pixel + 2.0 * pixel**3
    )
    return 2 * np.pi / (wavelength_nm * 1e-9)


def source_spectrum(k):
    k0 = np.mean(k)
    sigma_k = abs(k[-1] - k[0]) / 4.5
    source = np.exp(-0.5 * ((k - k0) / sigma_k) ** 2)
    return source / source.max()


def add_reflector(signal, k, depth_m, amplitude, phase=0.0):
    signal += amplitude * np.cos(2 * k * depth_m + phase)


def simulate_raw_bscan(rng):
    """Create camera counts with layers, speckle, beads, DC and noise."""
    k_camera = camera_wavenumbers()
    source = source_spectrum(k_camera)
    x_mm = np.linspace(0.0, 2.5, N_ASCANS)

    # Two slightly curved tissue interfaces.
    surface_mm = 1.20 + 0.055 * np.sin(2 * np.pi * x_mm / 2.5)
    lower_mm = surface_mm + 0.48 + 0.025 * np.sin(
        4 * np.pi * x_mm / 2.5 + 0.7
    )

    dark_counts = 450.0
    reference_counts = 28000.0 * source
    fixed_pattern = 35.0 * np.sin(2 * np.pi * np.arange(N_SPECTRAL) / 37.0)
    mean_background = dark_counts + reference_counts + fixed_pattern

    # Average 32 reference-only frames as a realistic background acquisition.
    background = rng.poisson(
        np.maximum(mean_background, 1.0), size=(32, N_SPECTRAL)
    ).mean(axis=0)
    raw = np.empty((N_SPECTRAL, N_ASCANS), dtype=np.float64)

    beads = [
        (0.28, 1.55, 0.90),
        (0.72, 1.43, 0.75),
        (1.18, 1.62, 0.65),
        (1.67, 1.48, 0.80),
        (2.18, 1.58, 0.70),
    ]

    for ix, lateral_mm in enumerate(x_mm):
        fringe = np.zeros(N_SPECTRAL)
        surface_m = surface_mm[ix] * 1e-3
        lower_m = lower_mm[ix] * 1e-3

        # Specular interfaces with lateral reflectivity variation.
        add_reflector(
            fringe, k_camera, surface_m,
            0.90 * (1 + 0.12 * np.sin(ix / 13)),
        )
        add_reflector(fringe, k_camera, lower_m, 0.32, phase=0.4)

        # Random sub-resolution scatterers form coherent speckle.
        depths = rng.uniform(surface_m + 0.03e-3, lower_m, 55)
        amplitudes = rng.rayleigh(0.055, depths.size)
        phases = rng.uniform(0, 2 * np.pi, depths.size)
        attenuation = np.exp(-(depths - surface_m) / 0.75e-3)
        for depth, amplitude, phase, decay in zip(
            depths, amplitudes, phases, attenuation
        ):
            add_reflector(fringe, k_camera, depth, amplitude * decay, phase)

        # Gaussian lateral PSF creates compact bead responses.
        for bead_x, bead_z, amplitude in beads:
            lateral_psf = np.exp(-0.5 * ((lateral_mm - bead_x) / 0.025) ** 2)
            add_reflector(
                fringe, k_camera, bead_z * 1e-3,
                amplitude * lateral_psf, phase=0.25,
            )

        expected = np.maximum(
            mean_background + 1050.0 * source * fringe, 1.0
        )
        # Photon shot noise plus camera read noise.
        raw[:, ix] = rng.poisson(expected) + rng.normal(0.0, 8.0, N_SPECTRAL)

    return raw, background, k_camera, x_mm, surface_mm, lower_mm


def reconstruct_bscan(raw, background, k_camera):
    """Background subtraction, k-linearization, Hann window and depth FFT."""
    corrected = raw - background[:, None]
    order = np.argsort(k_camera)
    k_sorted = k_camera[order]
    corrected = corrected[order, :]
    k_linear = np.linspace(k_sorted[0], k_sorted[-1], N_SPECTRAL)

    linearized = np.empty_like(corrected)
    for ix in range(N_ASCANS):
        linearized[:, ix] = np.interp(
            k_linear, k_sorted, corrected[:, ix]
        )

    linearized -= np.mean(linearized, axis=0, keepdims=True)
    windowed = linearized * np.hanning(N_SPECTRAL)[:, None]
    magnitude = np.abs(
        np.fft.fft(windowed, axis=0)[:N_SPECTRAL // 2, :]
    )

    dk = k_linear[1] - k_linear[0]
    frequency = np.fft.fftfreq(N_SPECTRAL, d=dk)[:N_SPECTRAL // 2]
    depth_mm = np.pi * frequency * 1e3  # cos(2*k*z) gives z = pi*f_k
    return magnitude, depth_mm, linearized


def relative_db(values, floor_db=-55.0):
    db = 20 * np.log10(values + np.finfo(float).eps)
    db -= np.max(db)
    return np.clip(db, floor_db, 0.0)


def plot_results(raw, background, k_camera, magnitude, depth_mm, x_mm,
                 surface_mm, lower_mm):
    center = N_ASCANS // 2
    bscan_db = relative_db(magnitude)
    aline_db = relative_db(magnitude[:, center])

    figure = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.0, 1.45))
    ax_raw = figure.add_subplot(grid[0, 0])
    ax_aline = figure.add_subplot(grid[1, 0])
    ax_bscan = figure.add_subplot(grid[:, 1])

    ax_raw.plot(k_camera / 1e6, raw[:, center], lw=0.7, label='camera signal')
    ax_raw.plot(k_camera / 1e6, background, lw=1.0, label='background')
    ax_raw.set(title='Simulated spectrometer acquisition',
               xlabel='Wavenumber (Mrad/m)', ylabel='Camera counts')
    ax_raw.legend(fontsize=8)

    ax_aline.plot(depth_mm, aline_db, color='black', lw=0.9)
    ax_aline.set(title='Reconstructed center A-line', xlabel='Depth (mm)',
                 ylabel='Relative intensity (dB)', xlim=(0.7, 2.3),
                 ylim=(-55, 2))
    ax_aline.grid(alpha=0.25)

    image = ax_bscan.imshow(
        bscan_db, cmap='gray', aspect='auto',
        extent=(x_mm[0], x_mm[-1], depth_mm[-1], depth_mm[0]),
        vmin=-55, vmax=0,
    )
    ax_bscan.plot(x_mm, surface_mm, '--', color='cyan', lw=0.8,
                  label='true interfaces')
    ax_bscan.plot(x_mm, lower_mm, '--', color='cyan', lw=0.8)
    ax_bscan.set(title='Realistic simulated SD-OCT B-scan',
                 xlabel='Lateral position (mm)', ylabel='Depth (mm)',
                 ylim=(2.3, 0.7))
    ax_bscan.legend(loc='lower right', fontsize=8)
    figure.colorbar(image, ax=ax_bscan, label='Relative intensity (dB)')
    return figure


def main():
    rng = np.random.default_rng(RNG_SEED)
    raw, background, k_camera, x_mm, surface_mm, lower_mm = (
        simulate_raw_bscan(rng)
    )
    magnitude, depth_mm, _ = reconstruct_bscan(raw, background, k_camera)
    figure = plot_results(
        raw, background, k_camera, magnitude, depth_mm,
        x_mm, surface_mm, lower_mm,
    )
    output = Path(__file__).with_name('simulated_oct_bscan.png')
    figure.savefig(output, dpi=180, bbox_inches='tight')
    print(f'Simulated B-scan saved to: {output}')
    if 'agg' not in plt.get_backend().lower():
        plt.show()


if __name__ == '__main__':
    main()
