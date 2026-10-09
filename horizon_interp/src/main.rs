use anyhow::{Context, Result, ensure};
use delaunator::{Point, triangulate};
use image::{Rgba, RgbaImage};
use std::{fs, path::Path};

struct Sample {
    x: f64,
    y: f64,
    value: f64,
}

pub fn interpolate_horizon(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    width: u32,
    height: u32,
) -> Result<()> {
    ensure!(width >= 2 && height >= 2, "Invalid image dimensions!");

    let content = fs::read_to_string(input)?;
    let mut samples = Vec::new();
    let mut inline = None;
    let mut xline = None;

    for line in content.lines() {
        let line = line.trim();

        if let Some(value) = line.strip_prefix("ILINE :") {
            inline = value.trim().parse::<i32>().ok();
            continue;
        }

        if let Some(value) = line.strip_prefix("XLINE :") {
            xline = value.trim().parse::<i32>().ok();
            continue;
        }

        let fields: Vec<f64> = line
            .split_whitespace()
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();

        if fields.len() != 4 || !fields.iter().all(|v| v.is_finite()) {
            continue;
        }

        samples.push(Sample {
            y: fields[0],
            x: fields[1],
            value: fields[3],
        });
    }

    ensure!(
        samples.len() >= 3,
        "At least three valid samples are required"
    );

    let points: Vec<Point> = samples.iter().map(|s| Point { x: s.x, y: x.y }).collect();

    let mesh = triangulate(&points);

    ensure!(!mesh.triangles.is_empty(), "Triangulation failed");

    let min_x = samples.iter().map(|s| s.x).fold(f64::INFINITY, f64::min);
    let max_x = samples
        .iter()
        .map(|s| s.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = samples.iter().map(|s| s.y).fold(f64::INFINITY, f64::min);
    let max_y = samples
        .iter()
        .map(|s| s.y)
        .fold(f64::NEG_INFINITY, f64::max);

    ensure!(max_x > min_x && max_y > min_y, "Invalid spatial extent");

    let min_v = samples
        .iter()
        .map(|s| s.value)
        .fold(f64::INFINITY, f64::min);
    let max_v = samples
        .iter()
        .map(|s| s.value)
        .fold(f64::NEG_INFINITY, f64::max);

    let mut values = vec![None; (width * height) as usize];

    let px = |x: f64| (x - min_x) / (max_x - min_x) * (width - 1) as f64;
    let py = |y: f64| (max_y - y) / (max_y - min_y) * (height - 1) as f64;

    for tri in mesh.triangles.chunks_exact(3) {
        let ids = [tri[0], tri[1], tri[2]];
        let p = ids.map(|i| (px(samples[i].x), py(samples[i].y)));

        let area = (p[1].0 - p[0].0) * (p[2].1 - p[0].1) - (p[1].1 - p[0].1) * (p[2].0 - p[0].0);

        if area.abs() < 1e-12 {
            continue;
        }

        let x0 = p
            .iter()
            .map(|p| p.0)
            .fold(f64::INFINITY, f64::min)
            .floor()
            .max(0.0) as u32;
        let x1 = p
            .iter()
            .map(|p| p.0)
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil()
            .min((width - 1) as f64) as u32;
        let y0 = p
            .iter()
            .map(|p| p.1)
            .fold(f64::INFINITY, f64::min)
            .floor()
            .max(0.0) as u32;
        let y1 = p
            .iter()
            .map(|p| p.1)
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil()
            .min((height - 1) as f64) as u32;

        for y in y0..=y1 {
            for x in x0..=x1 {
                let qx = x as f64;
                let qy = y as f64;

                let w1 = ((p[1].0 - qx) * (p[2].1 - qy) - (p[1].1 - qy) * (p[2].0 - qx)) / area;
                let w2 = ((p[2].0 - qx) * (p[0].1 - qy) - (p[2].1 - qy) * (p[0].0 - qx)) / area;
                let w3 = 1.0 - w1 - w2;

                if w1 >= -1e-9 && w2 >= -1e-9 && w3 >= -1e-9 {
                    let value = w1 * samples[ids[0]].value
                        + w2 * samples[ids[1]].value
                        + w3 * samples[ids[2]].value;

                    values[(y * width + x) as usize] = Some(value);
                }
            }
        }
    }

    let mut image = RgbaImage::new(width, height);

    for (i, pixel) in image.pixels_mut().enumerate() {
        if let Some(value) = values[i] {
            let normalized = if max_v > min_v {
                ((value - min_v) / (max_v - min_v) * 255.0).round() as u8
            } else {
                128
            };

            *pixel = Rgba([normalized, normalized, normalized, 255]);
        } else {
            *pixel = Rgba([0, 0, 0, 0]);
        }
    }

    image
        .save(output)
        .context("Failed to save interpolated horizon")?;

    Ok(())
}

fn main() {
    interpolate_horizon(
        "../data/inversao_ab140topo.txt",
        "horizon_interpolated.png",
        2048,
        2048,
    );
}
