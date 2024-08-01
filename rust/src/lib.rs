use std::ffi::{c_char, CStr};

use image::{ImageBuffer, ImageResult, Rgb};

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn load_image_rs(path: *const c_char) -> CRgbChannels {
    let path = unsafe { CStr::from_ptr(path).to_string_lossy() };
    let channels =
        load_image(path.to_string()).unwrap_or_else(|_| panic!("Failed to load image at {}", path));
    println!("Successfully load image at {}", path);
    channels.into()
}

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn save_image_rs(path: *const c_char, channels: CRgbChannels) {
    let path = unsafe { CStr::from_ptr(path).to_string_lossy() };
    let channels = RgbChannels::from(channels);
    save_image(&channels, path.to_string())
        .unwrap_or_else(|_| panic!("Failed to save image at {}", path));
    println!("Successfully save image at {}", path);
    let _ = CRgbChannels::from(channels);
}

#[no_mangle]
pub extern "C" fn free_rgb_channels(channels: CRgbChannels) {
    let channels = RgbChannels::from(channels);
    drop(channels);
}

#[derive(Default, Clone)]
struct RgbChannels {
    width: u32,
    height: u32,
    r: Vec<f32>,
    g: Vec<f32>,
    b: Vec<f32>,
}

impl RgbChannels {
    fn new() -> Self {
        Self::default()
    }

    fn assembly(&self) -> Vec<u8> {
        let mut ret = vec![];
        for i in 0..self.r.len() {
            ret.push((self.r[i] * 255.) as u8);
            ret.push((self.g[i] * 255.) as u8);
            ret.push((self.b[i] * 255.) as u8);
        }
        ret
    }
}

#[repr(C)]
pub struct CRgbChannels {
    width: u32,
    height: u32,
    r: *mut f32,
    g: *mut f32,
    b: *mut f32,
}

impl From<RgbChannels> for CRgbChannels {
    fn from(value: RgbChannels) -> Self {
        let mut r = value.r.into_boxed_slice();
        let r_ptr = r.as_mut_ptr();
        Box::into_raw(r);
        let mut g = value.g.into_boxed_slice();
        let g_ptr = g.as_mut_ptr();
        Box::into_raw(g);
        let mut b = value.b.into_boxed_slice();
        let b_ptr = b.as_mut_ptr();
        Box::into_raw(b);
        CRgbChannels {
            width: value.width,
            height: value.height,
            r: r_ptr,
            g: g_ptr,
            b: b_ptr,
        }
    }
}

impl From<CRgbChannels> for RgbChannels {
    fn from(value: CRgbChannels) -> Self {
        let size = (value.width * value.height) as usize;
        unsafe {
            RgbChannels {
                width: value.width,
                height: value.height,
                r: Vec::from_raw_parts(value.r, size, size),
                g: Vec::from_raw_parts(value.g, size, size),
                b: Vec::from_raw_parts(value.b, size, size),
            }
        }
    }
}

fn load_image(path: impl AsRef<std::path::Path>) -> ImageResult<RgbChannels> {
    let img = image::ImageReader::open(path)?.decode()?.to_rgb32f();
    let (width, height) = (img.width(), img.height());
    let size = width * height;
    let mut rgb = RgbChannels::new();
    for pixel in img.pixels() {
        rgb.r.push(pixel[0]);
        rgb.g.push(pixel[1]);
        rgb.b.push(pixel[2]);
    }
    rgb.width = width;
    rgb.height = height;

    assert!(rgb.r.len() == size as usize);
    assert!(rgb.g.len() == size as usize);
    assert!(rgb.b.len() == size as usize);

    Ok(rgb)
}

fn save_image(rgb: &RgbChannels, path: impl AsRef<std::path::Path>) -> ImageResult<()> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        image::ImageBuffer::from_vec(rgb.width, rgb.height, rgb.assembly())
            .expect("failed to construct image buffer");
    img.save(path)?;
    Ok(())
}
