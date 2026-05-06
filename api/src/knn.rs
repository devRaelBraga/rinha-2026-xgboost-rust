use std::fs::File;
use std::io::Read;
use std::path::Path;

const NLIST: usize = 4096;
const NPROBE: usize = 16;
const DIM: usize = 14;

pub struct IvfIndex {
    centroids: Vec<[f32; 16]>,
    offsets: Vec<u32>,
    vectors: Vec<[i16; 16]>,
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
        let mut raw_centroids = vec![0.0f32; len_centroids / 4];
        let slice_centroids = unsafe { std::slice::from_raw_parts_mut(raw_centroids.as_mut_ptr() as *mut u8, len_centroids) };
        f_centroids.read_exact(slice_centroids)?;

        let num_centroids = raw_centroids.len() / DIM;
        let mut centroids = Vec::with_capacity(num_centroids);
        for chunk in raw_centroids.chunks_exact(DIM) {
            let mut arr = [0.0f32; 16];
            arr[..DIM].copy_from_slice(chunk);
            centroids.push(arr);
        }

        let mut f_offsets = File::open(offsets_path)?;
        let len_offsets = f_offsets.metadata()?.len() as usize;
        let mut offsets = vec![0u32; len_offsets / 4];
        let slice_offsets = unsafe { std::slice::from_raw_parts_mut(offsets.as_mut_ptr() as *mut u8, len_offsets) };
        f_offsets.read_exact(slice_offsets)?;

        let mut f_vectors = File::open(vectors_path)?;
        let len_vectors = f_vectors.metadata()?.len() as usize;
        let mut raw_vectors = vec![0i16; len_vectors / 2];
        let slice_vectors = unsafe { std::slice::from_raw_parts_mut(raw_vectors.as_mut_ptr() as *mut u8, len_vectors) };
        f_vectors.read_exact(slice_vectors)?;

        let num_vectors = raw_vectors.len() / DIM;
        let mut vectors = Vec::with_capacity(num_vectors);
        for chunk in raw_vectors.chunks_exact(DIM) {
            let mut arr = [0i16; 16];
            arr[..DIM].copy_from_slice(chunk);
            vectors.push(arr);
        }

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
        let mut q_f32 = [0.0f32; 16];
        q_f32[..DIM].copy_from_slice(query);

        let mut top_cells = [0usize; NPROBE];
        let mut top_cell_dists = [f32::MAX; NPROBE];

        for i in 0..NLIST {
            let target_centroid = &self.centroids[i];
            let mut d = 0.0f32;
            for j in 0..16 {
                let diff = q_f32[j] - target_centroid[j];
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

        let mut q = [0i16; 16];
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
                let target_vec = &self.vectors[i];
                
                let mut dist: u64 = 0;
                for j in 0..16 {
                    let diff = target_vec[j] as i32 - q[j] as i32;
                    dist += (diff * diff) as u64;
                }

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
