use crate::models::{FraudRequest, Normalization};
use chrono::{DateTime, Datelike, Timelike, Utc};
use std::collections::HashMap;

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

pub fn vectorize(
    req: &FraudRequest,
    norm: &Normalization,
    mcc_risk: &HashMap<String, f64>,
) -> [f32; 14] {
    let mut v = [0.0f32; 14];

    let t = match req.transaction.requested_at.parse::<DateTime<Utc>>() {
        Ok(dt) => dt,
        Err(_) => Utc::now(), // Fallback
    };

    v[0] = clamp(req.transaction.amount / norm.max_amount) as f32;
    v[1] = clamp(req.transaction.installments as f64 / norm.max_installments) as f32;
    
    let avg_amt = if req.customer.avg_amount > 0.0 { req.customer.avg_amount } else { 1.0 };
    v[2] = clamp((req.transaction.amount / avg_amt) / norm.amount_vs_avg_ratio) as f32;
    
    v[3] = (t.hour() as f64 * INV23) as f32;
    
    // Day of week: Monday=0, Sunday=6
    let dow = t.weekday().num_days_from_monday(); 
    v[4] = (dow as f64 * INV6) as f32;

    if let Some(ref lt) = req.last_transaction {
        if let Ok(last_t) = lt.timestamp.parse::<DateTime<Utc>>() {
            let minutes = (t.timestamp() - last_t.timestamp()) as f64 / 60.0;
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

    v[12] = mcc_risk.get(&req.merchant.mcc).copied().unwrap_or(0.5) as f32;
    v[13] = clamp(req.merchant.avg_amount / norm.max_merchant_avg_amount) as f32;

    v
}
