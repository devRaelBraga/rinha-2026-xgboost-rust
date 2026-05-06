use std::fs::File;
use std::io::Read;
use std::path::Path;

const NLIST: usize = 4096;
const NPROBE: usize = 16;
const DIM: usize = 14;

pub struct IvfIndex {
    centroids: Vec<f32>,
    offsets: Vec<u32>,
    vectors: Vec<i16>,
    labels: Vec<u8>,
}

impl IvfIndex {
    pub fn load(
        centroids_path: impl AsRef<Path>,
        offsets_path: impl AsRef<Path>,
        vectors_path: impl AsRef<Path>,
        labels_path: impl AsRef<Path>,
    ) -> std::io::Result<Self> {
        let mut f_centroids = File::open(centroids_path)?;
        let len_centroids = f_centroids.metadata()?.len() as usize;
        let mut centroids = vec![0.0f32; len_centroids / 4];
        let slice_centroids = unsafe { std::slice::from_raw_parts_mut(centroids.as_mut_ptr() as *mut u8, len_centroids) };
        f_centroids.read_exact(slice_centroids)?;

        let mut f_offsets = File::open(offsets_path)?;
        let len_offsets = f_offsets.metadata()?.len() as usize;
        let mut offsets = vec![0u32; len_offsets / 4];
        let slice_offsets = unsafe { std::slice::from_raw_parts_mut(offsets.as_mut_ptr() as *mut u8, len_offsets) };
        f_offsets.read_exact(slice_offsets)?;

        let mut f_vectors = File::open(vectors_path)?;
        let len_vectors = f_vectors.metadata()?.len() as usize;
        let mut vectors = vec![0i16; len_vectors / 2];
        let slice_vectors = unsafe { std::slice::from_raw_parts_mut(vectors.as_mut_ptr() as *mut u8, len_vectors) };
        f_vectors.read_exact(slice_vectors)?;

        let mut f_labels = File::open(labels_path)?;
        let len_labels = f_labels.metadata()?.len() as usize;
        let mut labels = vec![0u8; len_labels];
        f_labels.read_exact(&mut labels)?;

        Ok(Self {
            centroids,
            offsets,
            vectors,
            labels,
        })
    }

    pub fn search(&self, query: &[f32; 14]) -> f64 {
        let mut top_cells = [0usize; NPROBE];
        let mut top_cell_dists = [f32::MAX; NPROBE];

        for i in 0..NLIST {
            let offset = i * DIM;
            let mut d = 0.0f32;
            for j in 0..DIM {
                let diff = query[j] - self.centroids[offset + j];
                d += diff * diff;
            }

            if d < top_cell_dists[NPROBE - 1] {
                let mut pos = NPROBE - 1;
                while pos > 0 && d < top_cell_dists[pos - 1] {
                    top_cell_dists[pos] = top_cell_dists[pos - 1];
                    top_cells[pos] = top_cells[pos - 1];
                    pos -= 1;
                }
                top_cell_dists[pos] = d;
                top_cells[pos] = i;
            }
        }

        let mut q = [0i16; DIM];
        for i in 0..DIM {
            let v = query[i];
            q[i] = if v < 0.0 {
                (v * 10000.0 - 0.5) as i16
            } else {
                (v * 10000.0 + 0.5) as i16
            };
        }

        let mut top_dist = [u64::MAX; 5];
        let mut top_idx = [0usize; 5];

        for &cell_id in &top_cells {
            let start = self.offsets[cell_id] as usize;
            let end = self.offsets[cell_id + 1] as usize;

            for i in start..end {
                let offset = i * DIM;
                
                let v0 = self.vectors[offset] as i32 - q[0] as i32;
                let v1 = self.vectors[offset + 1] as i32 - q[1] as i32;
                let v2 = self.vectors[offset + 2] as i32 - q[2] as i32;
                let v3 = self.vectors[offset + 3] as i32 - q[3] as i32;
                let v4 = self.vectors[offset + 4] as i32 - q[4] as i32;
                let v5 = self.vectors[offset + 5] as i32 - q[5] as i32;
                let v6 = self.vectors[offset + 6] as i32 - q[6] as i32;
                let v7 = self.vectors[offset + 7] as i32 - q[7] as i32;

                let mut dist = (v0 * v0) as u64
                    + (v1 * v1) as u64
                    + (v2 * v2) as u64
                    + (v3 * v3) as u64
                    + (v4 * v4) as u64
                    + (v5 * v5) as u64
                    + (v6 * v6) as u64
                    + (v7 * v7) as u64;

                if dist >= top_dist[4] {
                    continue;
                }

                let v8 = self.vectors[offset + 8] as i32 - q[8] as i32;
                let v9 = self.vectors[offset + 9] as i32 - q[9] as i32;
                let v10 = self.vectors[offset + 10] as i32 - q[10] as i32;
                let v11 = self.vectors[offset + 11] as i32 - q[11] as i32;
                let v12 = self.vectors[offset + 12] as i32 - q[12] as i32;
                let v13 = self.vectors[offset + 13] as i32 - q[13] as i32;

                dist += (v8 * v8) as u64
                    + (v9 * v9) as u64
                    + (v10 * v10) as u64
                    + (v11 * v11) as u64
                    + (v12 * v12) as u64
                    + (v13 * v13) as u64;

                if dist < top_dist[4] {
                    let mut pos = 4;
                    while pos > 0 && dist < top_dist[pos - 1] {
                        top_dist[pos] = top_dist[pos - 1];
                        top_idx[pos] = top_idx[pos - 1];
                        pos -= 1;
                    }
                    top_dist[pos] = dist;
                    top_idx[pos] = i;
                }
            }
        }

        let mut fraud_count = 0;
        for i in 0..5 {
            let idx_val = top_idx[i];
            if idx_val == 0 && top_dist[i] == u64::MAX {
                continue;
            }
            let byte_idx = idx_val / 8;
            let bit_idx = idx_val % 8;
            if (self.labels[byte_idx] & (1 << bit_idx)) != 0 {
                fraud_count += 1;
            }
        }

        fraud_count as f64 / 5.0
    }
}
