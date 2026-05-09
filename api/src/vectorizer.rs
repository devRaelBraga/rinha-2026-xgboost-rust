use crate::models::{FraudRequest, Normalization};

const INV23: f64 = 1.0 / 23.0;
const INV6: f64 = 1.0 / 6.0;

#[inline]
fn clamp(v: f64) -> f64 {
    if v < 0.0 {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

/// Parse 2 ASCII decimal digits at `&bytes[off..off+2]` into a u32.
#[inline(always)]
fn parse2(b: &[u8], off: usize) -> u32 {
    ((b[off] - b'0') as u32) * 10 + (b[off + 1] - b'0') as u32
}

/// Parse 4 ASCII decimal digits at `&bytes[off..off+4]` into an i32.
#[inline(always)]
fn parse4(b: &[u8], off: usize) -> i32 {
    ((b[off] - b'0') as i32) * 1000
        + ((b[off + 1] - b'0') as i32) * 100
        + ((b[off + 2] - b'0') as i32) * 10
        + (b[off + 3] - b'0') as i32
}

/// Tomohiko Sakamoto's day-of-week algorithm.
/// Returns Monday=0 … Sunday=6 (matches chrono's num_days_from_monday).
#[inline]
fn day_of_week(y: i32, m: u32, d: u32) -> u32 {
    static T: [i32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if m < 3 { y - 1 } else { y };
    // Sakamoto returns Sun=0..Sat=6; shift to Mon=0..Sun=6
    ((y + y / 4 - y / 100 + y / 400 + T[(m - 1) as usize] + d as i32) % 7 + 6) as u32 % 7
}

/// Convert date+time components to seconds since Unix epoch (UTC).
/// Handles years ≥ 1970. No leap-second awareness (matches chrono).
#[inline]
fn to_unix_secs(y: i32, m: u32, d: u32, h: u32, min: u32, s: u32) -> i64 {
    // Days from 1970-01-01 using the civil-from-days algorithm
    let m = m as i64;
    let d = d as i64;
    let y = y as i64;

    // Shift March-based year so Feb is last month (handles leap day)
    let (y, m) = if m <= 2 { (y - 1, m + 9) } else { (y, m - 3) };

    let era = y / 400;
    let yoe = y - era * 400; // year-of-era [0, 399]
    let doy = (153 * m + 2) / 5 + d - 1; // day-of-year [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // day-of-era [0, 146096]
    let days = era * 146097 + doe - 719468; // days since epoch

    days * 86400 + h as i64 * 3600 + min as i64 * 60 + s as i64
}

/// Extract (year, month, day, hour, minute, second) from an ISO-8601 string.
/// Expected format: "YYYY-MM-DDTHH:MM:SS..." (at least 19 bytes).
/// Returns None if too short or non-ASCII.
#[inline]
fn parse_iso8601(s: &str) -> Option<(i32, u32, u32, u32, u32, u32)> {
    let b = s.as_bytes();
    if b.len() < 19 {
        return None;
    }
    let year = parse4(b, 0);
    let month = parse2(b, 5);
    let day = parse2(b, 8);
    let hour = parse2(b, 11);
    let minute = parse2(b, 14);
    let second = parse2(b, 17);
    Some((year, month, day, hour, minute, second))
}

pub fn vectorize(
    req: &FraudRequest,
    norm: &Normalization,
    mcc_risk: &[f32; 10000],
) -> [f32; 14] {
    let mut v = [0.0f32; 14];

    let (hour, dow) = match parse_iso8601(&req.transaction.requested_at) {
        Some((y, m, d, h, _min, _s)) => (h, day_of_week(y, m, d)),
        None => (12, 3), // fallback: noon on Wednesday
    };

    v[0] = clamp(req.transaction.amount / norm.max_amount) as f32;
    v[1] = clamp(req.transaction.installments as f64 / norm.max_installments) as f32;

    let avg_amt = if req.customer.avg_amount > 0.0 { req.customer.avg_amount } else { 1.0 };
    v[2] = clamp((req.transaction.amount / avg_amt) / norm.amount_vs_avg_ratio) as f32;

    v[3] = (hour as f64 * INV23) as f32;

    v[4] = (dow as f64 * INV6) as f32;

    if let Some(ref lt) = req.last_transaction {
        if let (Some(cur), Some(last)) = (
            parse_iso8601(&req.transaction.requested_at),
            parse_iso8601(&lt.timestamp),
        ) {
            let cur_ts = to_unix_secs(cur.0, cur.1, cur.2, cur.3, cur.4, cur.5);
            let last_ts = to_unix_secs(last.0, last.1, last.2, last.3, last.4, last.5);
            let minutes = (cur_ts - last_ts) as f64 / 60.0;
            v[5] = clamp(minutes / norm.max_minutes) as f32;
        } else {
            v[5] = -1.0;
        }
        v[6] = clamp(lt.km_from_current / norm.max_km) as f32;
    } else {
        v[5] = -1.0;
        v[6] = -1.0;
    }

    v[7] = clamp(req.terminal.km_from_home / norm.max_km) as f32;
    v[8] = clamp(req.customer.tx_count_24h as f64 / norm.max_tx_count_24h) as f32;
    v[9] = if req.terminal.is_online { 1.0 } else { 0.0 };
    v[10] = if req.terminal.card_present { 1.0 } else { 0.0 };

    v[11] = if req.customer.known_merchants.contains(&req.merchant.id) {
        0.0
    } else {
        1.0
    };

    // MCC risk from flat array lookup (zero hashing)
    let mcc_idx = parse_mcc_fast(&req.merchant.mcc);
    v[12] = if mcc_idx < 10000 { mcc_risk[mcc_idx] } else { 0.5 };
    v[13] = clamp(req.merchant.avg_amount / norm.max_merchant_avg_amount) as f32;

    v
}

/// Parse MCC code (typically 4 ASCII digits) to usize without heap allocation.
#[inline]
fn parse_mcc_fast(s: &str) -> usize {
    let mut val = 0usize;
    for &c in s.as_bytes() {
        if c < b'0' || c > b'9' {
            return 10000; // invalid → will trigger default
        }
        val = val * 10 + (c - b'0') as usize;
    }
    val
}
