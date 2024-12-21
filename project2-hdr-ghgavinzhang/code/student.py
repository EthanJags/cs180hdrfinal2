"""
HDR stencil code - student.py
CS 1290 Computational Photography, Brown U.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tqdm


# ========================================================================
# RADIANCE MAP RECONSTRUCTION
# ========================================================================


def solve_g(Z, B, l, w):
    """
    Given a set of pixel values observed for several pixels in several
    images with different exposure times, this function returns the
    imaging system's response function g as well as the log film irradiance
    values for the observed pixels.

    Args:
        Z[i,j]: the pixel values of pixel location number i in image j.
        B[i,j]: the log delta t, or log shutter speed, for image j at pixel i
                (will be the same value for each i within the same j).
        l       lamdba, the constant that determines the amount of
                smoothness.
        w[z]:   the weighting function value for pixel value z (where z is between 0 - 255).

    Returns:
        g[z]:   the log exposure corresponding to pixel value z (where z is between 0 - 255).
        lE[i]:  the log film irradiance at pixel location i.

    """
    
    # Get dimensions
    pixels_per_image, num_images = Z.shape
    num_pixel_values = 256
    
    # Calculate total number of equations needed
    data_equations = pixels_per_image * num_images  # One per pixel measurement
    smoothness_equations = num_pixel_values - 2     # One per adjacent pixel value pair
    constraint_equations = 1                        # Fix middle response
    total_equations = data_equations + smoothness_equations + constraint_equations
    
    # Initialize system of equations
    num_variables = num_pixel_values + pixels_per_image  # g values + log irradiances
    A = np.zeros((total_equations, num_variables))
    b = np.zeros(total_equations)
    
    # Fill data fitting equations
    for i in range(pixels_per_image):
        for j in range(num_images):
            eq = i * num_images + j
            z = Z[i,j]
            weight = w[z]
            
            # g(Z_ij)
            A[eq, z] = weight
            # -log(E_i) 
            A[eq, num_pixel_values + i] = -weight
            # log(Δt_j)
            b[eq] = weight * B[i,j]
            
    # Add constraint g(128) = 0
    eq = data_equations
    A[eq, 128] = 1
    
    # Add smoothness equations
    for z in range(1, num_pixel_values-1):
        eq = data_equations + 1 + (z-1)
        weight = l * w[z]
        A[eq, z-1:z+2] = [weight, -2*weight, weight]
        
    # Solve system
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Split solution into g and log irradiances
    g = x[:num_pixel_values]
    lE = x[num_pixel_values:]
    
    return g, lE



def hdr(file_names, g_red, g_green, g_blue, w, exposure_matrix, nr_exposures):
    """
    Given the imaging system's response function g (per channel), a weighting function
    for pixel intensity values, and an exposure matrix containing the log shutter
    speed for each image, reconstruct the HDR radiance map in accordance to section
    2.2 of Debevec and Malik 1997.

    Args:
        file_names:           exposure stack image filenames
        g_red:                response function g for the red channel.
        g_green:              response function g for the green channel.
        g_blue:               response function g for the blue channel.
        w[z]:                 the weighting function value for pixel value z
                              (where z is between 0 - 255).
        exposure_matrix[i,j]: the log delta t, or log shutter speed, for image j at pixel i
                              (will be the same value for each i within the same j).
        nr_exposures:         number of images / exposures

    Returns:
        hdr:                  the hdr radiance map.
    """
    # Load and convert images
    images = np.array([cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB) for fn in file_names])
    
    # Initialize output array
    height, width, _ = images[0].shape
    hdr_radiance_map = np.zeros((height, width, 3), dtype=np.float32)
    
    # Create array of response functions
    g_functions = np.array([g_red, g_green, g_blue])
    
    # Get log shutter speeds
    ln_t = exposure_matrix[1, :]
    
    # Process each color channel
    for channel in range(3):
        # Get pixel values for this channel across all exposures
        Z = images[:, :, :, channel]  # [nr_exposures, height, width]
        
        # Apply response function
        g_values = g_functions[channel][Z]
        
        # Get weights
        weights = w[Z]
        
        # Calculate weighted numerator and denominator
        numerator = np.sum(weights * (g_values - ln_t[:, np.newaxis, np.newaxis]), axis=0)
        denominator = np.sum(weights, axis=0)
        
        # Calculate final radiance values
        valid_pixels = denominator > 0
        hdr_radiance_map[..., channel][valid_pixels] = np.exp(
            numerator[valid_pixels] / denominator[valid_pixels]
        )
        
    return hdr_radiance_map


# ========================================================================
# TONE MAPPING
# ========================================================================


def tm_global_simple(hdr_radiance_map):
    """
    Simple global tone mapping function (Reinhard et al.)

    Equation:
        E_display = E_world / (1 + E_world)

    Args:
        hdr_radiance_map (np.array): HDR radiance map of the image
                                     with shape (H, W, 3)
    Returns:
        np.array of image with values in range [0.0, 1.0]
    """

    # Apply Reinhard global operator
    tone_mapped = np.divide(hdr_radiance_map, (1.0 + hdr_radiance_map))
    
    # Normalize to [0,1] range
    normalized = (tone_mapped - tone_mapped.min()) / (tone_mapped.max() - tone_mapped.min())
    
    return normalized


def reinhard_local_tone_mapping(hdr_radiance_map, gamma=2.2, a=0.18):
    """
    Apply Reinhard local tone mapping to an HDR radiance map.

    Args:
        hdr_radiance_map: The HDR radiance map (numpy array).
        gamma: Gamma value for gamma correction.
        a: A key value determining the overall brightness of the image.

    Returns:
        tone_mapped_img: The tone-mapped image.
    """
    # Calculate luminance using perceptual weights
    weights = np.array([0.27, 0.67, 0.06])
    luminance = np.sum(hdr_radiance_map * weights[None, None, :], axis=2)
    
    # Calculate log average luminance
    eps = 1e-6  # Small constant to avoid log(0)
    log_avg = np.exp(np.mean(np.log(luminance + eps)))
    
    # Scale luminance by key value and average
    scaled_luminance = (a / log_avg) * luminance
    
    # Apply local Reinhard operator
    adapted_luminance = scaled_luminance / (1 + scaled_luminance)
    
    # Scale colors while preserving ratios
    luminance_ratio = (adapted_luminance / (luminance + eps))[:, :, None]
    tone_mapped_img = hdr_radiance_map * luminance_ratio
    
    # Post-processing
    tone_mapped_img = np.clip(tone_mapped_img, 0, 1)
    tone_mapped_img = np.power(tone_mapped_img, 1/gamma)
    
    return tone_mapped_img


def bilateral_filter(img, diameter, sigma_color, sigma_space):
    # Setup spatial kernel
    radius = diameter // 2
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    spatial_kernel = np.exp(-(x**2 + y**2)/(2 * sigma_space**2))
    
    # Pad image
    padded = np.pad(img, radius, mode='symmetric')
    
    # Filter image
    output = np.zeros_like(img)
    for i in tqdm.tqdm(range(img.shape[0])):
        for j in range(img.shape[1]):
            # Extract window and compute range kernel
            window = padded[i:i+diameter, j:j+diameter]
            center_val = padded[i+radius, j+radius]
            range_kernel = np.exp(-((window - center_val)**2)/(2 * sigma_color**2))
            
            # Apply kernels and normalize
            weights = spatial_kernel * range_kernel
            norm = np.sum(weights)
            if norm > 0:
                output[i,j] = np.sum(window * weights) / norm
            else:
                output[i,j] = img[i,j]
    
    # Clip output to [0, 255]
    output = np.clip(output, 0, 255)
                
    return output

def tm_durand(hdr_radiance_map):
    # 1. Input is already linear RGB values
    
    # 2. Compute intensity by averaging color channels
    intensity = np.mean(hdr_radiance_map, axis=2)
    intensity = np.maximum(intensity, 1e-6)  # Avoid divide by zero
    
    # 3. Compute chrominance channels
    chrominance = np.zeros_like(hdr_radiance_map)
    for c in range(3):
        chrominance[...,c] = hdr_radiance_map[...,c] / intensity
        
    # 4. Take log of intensity
    log_intensity = np.log2(intensity)
    
    # 5. Apply bilateral filter to log intensity
    bilateral = bilateral_filter(log_intensity, 
                          diameter=9,
                          sigma_color=45,
                          sigma_space=45)
    
    # 6. Compute detail layer
    detail = log_intensity - bilateral
    
    # 7. Scale the base layer
    base_max = np.max(bilateral)
    base_min = np.min(bilateral)
    dynamic_range = 5.0  # Target dynamic range in stops
    scaling_factor = dynamic_range / (base_max - base_min)
    base = (bilateral - base_max) * scaling_factor
    
    # 8. Reconstruct log intensity
    log_output = base + detail
    output_intensity = np.power(2, log_output)
    
    # 9. Restore color by multiplying with chrominance
    output = np.zeros_like(hdr_radiance_map)
    for c in range(3):
        output[...,c] = output_intensity * chrominance[...,c]
        
    # 10. Apply gamma and clip to [0,1]
    gamma = 0.6
    output = np.power(output, gamma)
    output = np.clip(output, 0, 1)
    
    return output, bilateral, base, detail
