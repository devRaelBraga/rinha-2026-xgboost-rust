import http from 'k6/http';
import { check } from 'k6';
import { SharedArray } from 'k6/data';
import { Counter, Trend } from 'k6/metrics';
import { textSummary } from './k6-summary.js';
import exec from 'k6/execution';

const testData = new SharedArray('test-data', function () {
    return JSON.parse(open('./test-data.json')).entries;
});
const statsArr = new SharedArray('test-stats', function () {
    return [JSON.parse(open('./test-data.json')).stats];
});
const expectedStats = statsArr[0];

const tpCount = new Counter('tp_count');
const tnCount = new Counter('tn_count');
const fpCount = new Counter('fp_count');
const fnCount = new Counter('fn_count');
const errorCount = new Counter('error_count');

// Instrumentation: track indices of misclassified transactions
const fpIndices = new Trend('fp_indices', true);
const fnIndices = new Trend('fn_indices', true);

export const options = {
    summaryTrendStats: ['p(99)'],
    systemTags: ['status', 'method'],
    dns: {
        ttl: '5m',
        select: 'roundRobin',
    },
    scenarios: {
        default: {
            executor: 'ramping-arrival-rate',
            startRate: 1,
            timeUnit: '1s',
            preAllocatedVUs: 100,
            maxVUs: 250,
            gracefulStop: '10s',
            stages: [
                { duration: '120s', target: 900 },
            ],
        },
    },
};

export function setup() {
    console.log(
        `Dataset: ${expectedStats.total} entries, `
        + `${expectedStats.fraud_count} fraud (${expectedStats.fraud_rate}%), `
        + `${expectedStats.legit_count} legit (${expectedStats.legit_rate}%), `
        + `edge cases: ${expectedStats.edge_case_rate}%`
    );
}

export default function () {
    const idx = exec.scenario.iterationInTest;
    if (idx >= testData.length) return;
    const entry = testData[idx];
    const expectedApproved = entry.expected_approved;

    const res = http.post(
        'http://rinha-lb:9999/fraud-score',
        JSON.stringify(entry.request),
        { headers: { 'Content-Type': 'application/json' }, timeout: '2001ms' }
    );

    if (res.status === 200) {
        const body = JSON.parse(res.body);
        // Per-request scoring: compare against expectedApproved
        // expectedApproved === true  --> legit transaction
        // expectedApproved === false --> fraud transaction
        if (expectedApproved === body.approved) {
            if (body.approved) tnCount.add(1); // correctly approved legit
            else tpCount.add(1);               // correctly denied fraud
        } else {
            if (body.approved) {
                fnCount.add(1); // fraud approved (missed fraud)
                fnIndices.add(idx);
                console.warn(
                    `[FN] idx=${idx} id=${entry.request.id} `
                    + `expected_score=${entry.expected_fraud_score} `
                    + `got_approved=${body.approved} got_score=${body.fraud_score}`
                );
            } else {
                fpCount.add(1); // legit denied (false block)
                fpIndices.add(idx);
                console.warn(
                    `[FP] idx=${idx} id=${entry.request.id} `
                    + `expected_score=${entry.expected_fraud_score} `
                    + `got_approved=${body.approved} got_score=${body.fraud_score}`
                );
            }
        }
    } else {
        errorCount.add(1);
        console.warn(
            `[HTTP_ERR] idx=${idx} id=${entry.request.id} status=${res.status}`
        );
    }
}

export function handleSummary(data) {
    const K = 1000;
    const T_MAX_MS = 1000;
    const P99_MIN_MS = 1;
    const P99_MAX_MS = 2000;
    const EPSILON_MIN = 0.001;
    const BETA = 300;
    const TX_CORTE = 0.15;
    const SCORE_P99_CORTE = -3000;
    const SCORE_DET_CORTE = -3000;
    const PRECISION = __ENV.SCORE_PRECISION ? parseInt(__ENV.SCORE_PRECISION) : 2;

    const r = (v, decimals) => +v.toFixed(decimals);

    const httpDuration = data.metrics.http_req_duration.values;
    const p99 = httpDuration['p(99)'];

    const tp = data.metrics.tp_count ? data.metrics.tp_count.values.count : 0;
    const tn = data.metrics.tn_count ? data.metrics.tn_count.values.count : 0;
    const fp = data.metrics.fp_count ? data.metrics.fp_count.values.count : 0;
    const fn = data.metrics.fn_count ? data.metrics.fn_count.values.count : 0;
    const errs = data.metrics.error_count ? data.metrics.error_count.values.count : 0;

    const N = tp + tn + fp + fn + errs;

    // Erros ponderados (para a fórmula log) e contagem pura (para o corte)
    const E = (fp * 1) + (fn * 3) + (errs * 5);
    const failures = fp + fn + errs;
    const epsilon = N > 0 ? E / N : 0;
    const failureRate = N > 0 ? failures / N : 0;

    // Score P99 (log, com teto em P99_MIN_MS e corte em P99_MAX_MS).
    // p99=0 = nenhuma resposta completou; retorna 0 pra evitar Infinity no JSON.
    let p99Score;
    let p99CutTriggered = false;
    if (p99 <= 0) {
        p99Score = 0;
    } else if (p99 > P99_MAX_MS) {
        p99Score = SCORE_P99_CORTE;
        p99CutTriggered = true;
    } else {
        p99Score = K * Math.log10(T_MAX_MS / Math.max(p99, P99_MIN_MS));
    }

    // Score detecção (log com penalidade absoluta, ou corte em -3000 se falhas > 15%)
    let detScore;
    let rateComponent = 0;
    let absolutePenalty = 0;
    let cutTriggered = false;
    if (failureRate > TX_CORTE) {
        detScore = SCORE_DET_CORTE;
        cutTriggered = true;
    } else {
        rateComponent = K * Math.log10(1 / Math.max(epsilon, EPSILON_MIN));
        absolutePenalty = -BETA * Math.log10(1 + E);
        detScore = rateComponent + absolutePenalty;
    }

    const finalScore = p99Score + detScore;

    const result = {
        expected: expectedStats,
        p99: r(p99, PRECISION) + 'ms',
        scoring: {
            breakdown: {
                false_positive_detections: fp,
                false_negative_detections: fn,
                true_positive_detections: tp,
                true_negative_detections: tn,
                http_errors: errs,
            },
            failure_rate: r(failureRate * 100, PRECISION) + '%',
            weighted_errors_E: E,
            error_rate_epsilon: r(epsilon, PRECISION + 4),
            p99_score: {
                value: r(p99Score, PRECISION),
                cut_triggered: p99CutTriggered,
            },
            detection_score: {
                value: r(detScore, PRECISION),
                rate_component: cutTriggered ? null : r(rateComponent, PRECISION),
                absolute_penalty: cutTriggered ? null : r(absolutePenalty, PRECISION),
                cut_triggered: cutTriggered,
            },
            final_score: r(finalScore, PRECISION),
            raw: {
                p99_ms: p99,
                failure_rate: failureRate,
                error_rate_epsilon: epsilon,
                p99_score: p99Score,
                detection_score: detScore,
                rate_component: cutTriggered ? null : rateComponent,
                absolute_penalty: cutTriggered ? null : absolutePenalty,
                final_score: finalScore,
            },
        },
    };

    // --- Error diagnostics: collect full details of misclassified transactions ---
    const errorDetails = { false_positives: [], false_negatives: [] };

    // Extract error indices from Trend metrics
    // Trend values are stored as individual observations; we use the min/max/count
    // to know there are errors, then scan testData for matches.
    // Since we logged indices via console.warn, and Trend doesn't give us raw values,
    // we do a second pass over testData to reconstruct which entries would be errors
    // based on the expected_fraud_score distribution (edge cases near 0.6 boundary).
    if (fp > 0 || fn > 0) {
        // Re-derive the error entries by re-checking all entries.
        // We can access testData in handleSummary context since it's a SharedArray.
        for (let i = 0; i < testData.length; i++) {
            const e = testData[i];
            // Mark edge-case transactions likely to be misclassified
            if (e.expected_fraud_score === 0.4 || e.expected_fraud_score === 0.6
                || e.expected_fraud_score === 0.2 || e.expected_fraud_score === 0.8) {
                const detail = {
                    index: i,
                    id: e.request.id,
                    expected_approved: e.expected_approved,
                    expected_fraud_score: e.expected_fraud_score,
                    request: e.request,
                };
                if (e.expected_approved) {
                    // Could be a false positive (legit denied)
                    errorDetails.false_positives.push(detail);
                } else {
                    // Could be a false negative (fraud approved)
                    errorDetails.false_negatives.push(detail);
                }
            }
        }
    }

    // Summary of edge case counts
    result.edge_case_candidates = {
        fp_candidates: errorDetails.false_positives.length,
        fn_candidates: errorDetails.false_negatives.length,
        note: 'These are all edge-case entries (score 0.2/0.4/0.6/0.8) that could be misclassified. Cross-reference with console.warn [FP]/[FN] logs for actual errors.',
    };

    return {
        'results.json': JSON.stringify(result, null, 2),
        'edge-cases.json': JSON.stringify(errorDetails, null, 2),
        //stdout: textSummary(data, { indent: ' ', enableColors: true }),
    };
}
