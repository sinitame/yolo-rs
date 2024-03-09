use std::cmp::Ordering;

use anyhow::{anyhow, bail, Result};
use image::Rgb;
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use tract_onnx::{
    prelude::{
        tract_itertools::Itertools, tvec, Framework, InferenceModelExt, IntoTValue, IntoTensor,
        TValue, TVec, Tensor,
    },
    tract_hir::tract_ndarray::Array4,
};

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<String>>();
    let model_path = args.get(1).ok_or(anyhow!("Model path not specified."))?;
    let input_path = args.get(2).ok_or(anyhow!("Input path not specified."))?;
    let output_path = args.get(3).ok_or(anyhow!("Output path not specified."))?;
    let threshold = args
        .get(4)
        .ok_or(anyhow!("Expected threshold value."))?
        .parse::<f32>()
        .unwrap_or(0.0);
    let max_iou = args
        .get(5)
        .ok_or(anyhow!("Expected max IOU value."))?
        .parse::<f32>()
        .unwrap_or(0.0);

    // Load model
    let typed_model = tract_onnx::onnx()
        .model_for_path(&model_path)?
        .into_typed()?;
    let optimized_model = typed_model.into_optimized()?;
    let runnable_model = optimized_model.into_runnable()?;

    // Load input
    let input = pre_process_input(&input_path)?;

    // Execute model
    let output = runnable_model.run(tvec![input])?;
    let reshaped_output = reshape_output(output)?;
    dbg!(&reshaped_output);
    let detections = post_process_output(
        reshaped_output,
        threshold,
        //Some(vec![(10, 13), (16, 30), (33, 23)]),
        //Some((80, 80)),
        Some(vec![(116, 90), (156, 198), (373, 326)]),
        Some((20, 20)),
    )?;
    let filterd_detection = nms_filter(&detections, max_iou);
    //let filterd_detection = detections;
    render_detections(input_path, &filterd_detection, output_path)?;
    Ok(())
}

fn reshape_output(outputs: TVec<TValue>) -> Result<TValue> {
    let (grid_x, grid_y) = (20_usize, 20_usize);
    let out_0 = outputs[2].clone().into_tensor(); // 1, 255, 80, 80
    let reshaped_out = out_0.into_shape(&[1, 3, 85, grid_x, grid_y])?; // 1, 3, 85, 80, 80
    let transposed_out = reshaped_out.move_axis(2, 4)?; // 1, 3, 80, 80, 85
    let out = transposed_out.into_shape(&[1, 3 * grid_x * grid_y, 85])?; // 1, X, 85
    Ok(out.into_tvalue())
}

fn pre_process_input(path: &str) -> Result<TValue> {
    let image = image::open(path).unwrap().to_rgb8();
    let resized =
        image::imageops::resize(&image, 640, 640, ::image::imageops::FilterType::Triangle);
    let image = Array4::from_shape_fn((1, 3, 640, 640), |(_, c, y, x)| {
        resized[(x as _, y as _)][c] as f32 / 255.0
    })
    .into_tvalue();
    Ok(image)
}

fn post_process_output(
    raw_output: TValue,
    threshold: f32,
    anchors: Option<Vec<(usize, usize)>>,
    grid_size: Option<(usize, usize)>,
) -> Result<Vec<YoloDetection>> {
    let output_array = raw_output.to_array_view::<f32>()?;
    let detections = output_array
        .rows()
        .into_iter()
        .enumerate()
        .map(|(idx, it)| -> Result<YoloDetection> {
            let raw_detection = it
                .as_slice()
                .ok_or(anyhow!("Could not convert raw detection to slice."))?;
            if let Some((anchors, (grid_x, grid_y))) = anchors.as_ref().zip(grid_size.as_ref()) {
                let anchor = idx % anchors.len();
                let i = (idx / anchors.len()) % grid_x;
                let j = (idx / anchors.len()) / grid_x;
                YoloDetection::from_raw_detection_with_anchor(
                    raw_detection,
                    anchors[anchor],
                    i,
                    j,
                    640 / grid_x,
                    anchor == 0,
                )
            } else {
                YoloDetection::from_raw_detection(raw_detection)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(detections
        .into_iter()
        .filter(|it| it.confidence > threshold)
        .collect_vec())
}

/// Calculate Intersection Over Union (IOU) between two bounding boxes.
fn iou(a: &YoloDetection, b: &YoloDetection) -> f32 {
    let area_a = a.area();
    let area_b = b.area();

    let top_left = (a.x.max(b.x), a.y.max(b.y));
    let bottom_right = (a.x + a.width.min(b.width), a.y + a.height.min(b.height));

    let intersection =
        (bottom_right.0 - top_left.0).max(0.0) * (bottom_right.1 - top_left.1).max(0.0);

    intersection / (area_a + area_b - intersection)
}

fn nms_filter(detections: &[YoloDetection], max_iou: f32) -> Vec<YoloDetection> {
    let ordered_detections = detections
        .iter()
        .sorted_by(|d1, d2| d1.confidence.partial_cmp(&d2.confidence).unwrap())
        .collect_vec();

    ordered_detections
        .iter()
        .enumerate()
        .filter(|(idx, d1)| {
            ordered_detections
                .iter()
                .skip(*idx + 1)
                .filter(|d2| d1.class_index == d2.class_index)
                .all(|d2| iou(d1, d2) < max_iou)
        })
        .map(|(_, d)| (*d).clone())
        .collect_vec()
}

#[derive(Debug, Clone)]
pub struct YoloDetection {
    /// Top-Left Bounds Coordinate in X-Axis
    pub x: f32,
    // Top-Left Bounds Coordinate in Y-Axis
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub class_index: usize,
    pub confidence: f32,
}

impl YoloDetection {
    fn area(&self) -> f32 {
        self.width * self.height
    }

    fn grid_sensitivity_adjustment_pos(x: f32) -> f32 {
        let alpha = 2.0;
        x * alpha + (1.0 - alpha) * 0.5
    }

    fn grid_sensitivity_adjustment_size(x: f32, anchor_x: usize) -> f32 {
        (x * x * 4.0) * anchor_x as f32
    }

    fn from_raw_detection_with_anchor(
        raw: &[f32],
        anchor: (usize, usize),
        i: usize,
        j: usize,
        stride: usize,
        should_debug: bool,
    ) -> Result<Self> {
        let ([cx, cy, w, h, box_confidence], confidence) = raw.split_at(5) else {
            bail!("Expected raw detection to have 85 elements")
        };
        let (best_class, _best_confidence) = confidence
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
            .unwrap();

        let adjusted_cx = (Self::grid_sensitivity_adjustment_pos(*cx) + i as f32) * stride as f32;
        let adjusted_cy = (Self::grid_sensitivity_adjustment_pos(*cy) + j as f32) * stride as f32;
        let adjusted_w = Self::grid_sensitivity_adjustment_size(*w, anchor.0);
        let adjusted_h = Self::grid_sensitivity_adjustment_size(*h, anchor.1);
        if should_debug {
            println!("[{i}, {j}]({cx}, {cy}, {w}, {h}) ax {adjusted_cx}, ay {adjusted_cy}, aw {adjusted_w}, ah {adjusted_h}.");
        }
        Ok(Self {
            x: (adjusted_cx - adjusted_w / 2.0) / 640.0,
            y: (adjusted_cy - adjusted_h / 2.0) / 640.0,
            width: adjusted_w / 640.0,
            height: adjusted_h / 640.0,
            class_index: best_class,
            confidence: *box_confidence,
        })
    }

    // Raw detection is: [x, y, w, h, class1_prob, ..., class80_prob]
    fn from_raw_detection(raw: &[f32]) -> Result<Self> {
        let ([cx, cy, w, h, box_confidence], confidence) = raw.split_at(5) else {
            bail!("Expected raw detection to have 85 elements")
        };
        let (best_class, _best_confidence) = confidence
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
            .unwrap();
        Ok(YoloDetection {
            x: (*cx - w / 2.0) / 640.0,
            y: (*cy - h / 2.0) / 640.0,
            width: *w / 640.0,
            height: *h / 640.0,
            class_index: best_class,
            confidence: *box_confidence,
        })
    }
}

pub fn render_detections(
    image_path: &str,
    detections: &Vec<YoloDetection>,
    output_path: &str,
) -> Result<()> {
    let image = image::open(image_path).unwrap();
    let mut image = image.to_rgb8();
    for detection in detections.iter() {
        dbg!(detection.class_index);
        dbg!(detection.confidence);
        dbg!(detection.x);
        dbg!(detection.y);
        dbg!(detection.width);
        dbg!(detection.height);
        let x = (detection.x) * image.width() as f32;
        let y = (detection.y) * image.height() as f32;
        let width = (detection.width) * image.width() as f32;
        let height = (detection.height) * image.height() as f32;
        draw_hollow_rect_mut(
            &mut image,
            Rect::at(x as i32, y as i32).of_size(width as u32, height as u32),
            Rgb([255u8, 0u8, 0u8]),
        );
    }

    image.save(output_path).unwrap();

    Ok(())
}
