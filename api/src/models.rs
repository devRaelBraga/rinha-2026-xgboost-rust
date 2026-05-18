use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct FraudRequest<'a> {
    pub id: &'a str,
    #[serde(borrow)]
    pub transaction: Transaction<'a>,
    #[serde(borrow)]
    pub customer: Customer<'a>,
    #[serde(borrow)]
    pub merchant: Merchant<'a>,
    pub terminal: Terminal,
    #[serde(borrow)]
    pub last_transaction: Option<LastTransaction<'a>>,
}

#[derive(Debug, Deserialize)]
pub struct Transaction<'a> {
    pub amount: f64,
    pub installments: i32,
    pub requested_at: &'a str,
}

#[derive(Debug, Deserialize)]
pub struct Customer<'a> {
    pub avg_amount: f64,
    pub tx_count_24h: i32,
    #[serde(borrow)]
    pub known_merchants: Vec<&'a str>,
}

#[derive(Debug, Deserialize)]
pub struct Merchant<'a> {
    pub id: &'a str,
    pub mcc: &'a str,
    pub avg_amount: f64,
}

#[derive(Debug, Deserialize)]
pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f64,
}

#[derive(Debug, Deserialize)]
pub struct LastTransaction<'a> {
    pub timestamp: &'a str,
    pub km_from_current: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Normalization {
    pub max_amount: f64,
    pub max_installments: f64,
    pub amount_vs_avg_ratio: f64,
    pub max_minutes: f64,
    pub max_km: f64,
    pub max_tx_count_24h: f64,
    pub max_merchant_avg_amount: f64,
}
