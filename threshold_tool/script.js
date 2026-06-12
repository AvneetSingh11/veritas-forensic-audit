// Normal Distribution function
function normalDistribution(x, mean, stdDev) {
    return (1 / (stdDev * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mean) / stdDev, 2));
}

// Configuration for our two curves
const authMean = 40;
const authStdDev = 12;
const synMean = 80;
const synStdDev = 10;

// Generate Data
const labels = [];
const authData = [];
const synData = [];

for (let x = 0; x <= 100; x++) {
    labels.push(x);
    // Multiply by 100 just to scale the numbers up nicely for the chart
    authData.push(normalDistribution(x, authMean, authStdDev) * 100);
    synData.push(normalDistribution(x, synMean, synStdDev) * 100);
}

// Sum totals for accurate percentage calculations
const authTotalArea = authData.reduce((a, b) => a + b, 0);
const synTotalArea = synData.reduce((a, b) => a + b, 0);

// Colors matching CSS variables
const colorAuth = '#3B82F6';
const colorSyn = '#4CAF50';
const colorFP = 'rgba(59, 130, 246, 0.5)';
const colorFN = 'rgba(76, 175, 80, 0.5)';

// Init Chart
const ctx = document.getElementById('thresholdChart').getContext('2d');
const chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: labels,
        datasets: [
            {
                label: 'Authentic Line',
                data: authData,
                borderColor: colorAuth,
                borderWidth: 2,
                pointRadius: 0,
                fill: false,
                tension: 0.4
            },
            {
                label: 'Synthetic Line',
                data: synData,
                borderColor: colorSyn,
                borderWidth: 2,
                pointRadius: 0,
                fill: false,
                tension: 0.4
            },
            {
                label: 'False Positive Fill',
                data: [], // Updated dynamically
                backgroundColor: colorFP,
                borderWidth: 0,
                pointRadius: 0,
                fill: true,
                tension: 0.4
            },
            {
                label: 'False Negative Fill',
                data: [], // Updated dynamically
                backgroundColor: colorFN,
                borderWidth: 0,
                pointRadius: 0,
                fill: true,
                tension: 0.4
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false }, // We have our own HTML legend
            tooltip: { enabled: false },
            annotation: {
                annotations: {
                    thresholdLine: {
                        type: 'line',
                        xMin: 75,
                        xMax: 75,
                        borderColor: '#ffffff',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        label: {
                            display: true,
                            content: 'Threshold: 75%',
                            position: 'start',
                            backgroundColor: 'transparent',
                            color: '#ffffff',
                            font: { size: 12, family: 'Inter' },
                            xAdjust: 40,
                            yAdjust: -10
                        }
                    }
                }
            }
        },
        scales: {
            x: {
                grid: { color: '#333333' },
                ticks: { color: '#aaaaaa', font: { family: 'Inter' } }
            },
            y: {
                display: false, // Hide y-axis as per mockup
                min: 0
            }
        },
        animation: {
            duration: 0 // Disable animation for instant slider updates
        }
    }
});

// DOM Elements
const slider = document.getElementById('threshold-slider');
const numberInput = document.getElementById('threshold-input');
const fpRateEl = document.getElementById('fp-rate');
const fnRateEl = document.getElementById('fn-rate');
const contextDropdown = document.getElementById('context-dropdown');
const riskTextEl = document.getElementById('risk-text');
const resetBtn = document.getElementById('reset-btn');

function updateSimulation(threshold) {
    // 1. Calculate new filled data arrays
    const fpData = authData.map((val, x) => (x >= threshold ? val : 0));
    const fnData = synData.map((val, x) => (x <= threshold ? val : 0));

    // Update datasets
    chart.data.datasets[2].data = fpData;
    chart.data.datasets[3].data = fnData;

    // 2. Update annotation line
    chart.options.plugins.annotation.annotations.thresholdLine.xMin = threshold;
    chart.options.plugins.annotation.annotations.thresholdLine.xMax = threshold;
    chart.options.plugins.annotation.annotations.thresholdLine.label.content = `Threshold: ${threshold}%`;

    chart.update();

    // 3. Calculate Rates
    const fpArea = fpData.reduce((a, b) => a + b, 0);
    const fnArea = fnData.reduce((a, b) => a + b, 0);

    const fpRate = (fpArea / authTotalArea) * 100;
    const fnRate = (fnArea / synTotalArea) * 100;

    fpRateEl.textContent = fpRate.toFixed(1) + '%';
    fnRateEl.textContent = fnRate.toFixed(1) + '%';

    // 4. Update Risk Text
    updateRiskText(threshold, contextDropdown.value, fpRate, fnRate);
}

function updateRiskText(threshold, context, fpRate, fnRate) {
    let riskMsg = "";
    
    if (context === "journalism") {
        if (fpRate > 10) {
            riskMsg = "Critical Risk: Accusing innocent sources falsely, destroying publication credibility.";
            riskTextEl.style.color = "#ff5252";
        } else if (fnRate > 30) {
            riskMsg = "Risk: Failing to detect AI fakes, potentially publishing deepfakes as real news.";
            riskTextEl.style.color = "#aaaaaa";
        } else {
            riskMsg = "Balanced: Minimizing false accusations while catching obvious deepfakes.";
            riskTextEl.style.color = "#4CAF50";
        }
    } else if (context === "social") {
        if (fpRate > 20) {
            riskMsg = "Risk: Censoring authentic user content, causing user outrage and platform abandonment.";
            riskTextEl.style.color = "#ff5252";
        } else if (fnRate > 50) {
            riskMsg = "Warning: Platform flooded with undetected AI spam and synthetic media.";
            riskTextEl.style.color = "#ff9800";
        } else {
            riskMsg = "Optimized for user retention and moderate safety.";
            riskTextEl.style.color = "#4CAF50";
        }
    } else if (context === "finance") {
        if (fnRate > 5) {
            riskMsg = "Critical Risk: Permitting deepfake voice/video authentication. High financial loss probability.";
            riskTextEl.style.color = "#ff5252";
        } else if (fpRate > 40) {
            riskMsg = "Friction: Legitimate users locked out of accounts due to overly strict checks.";
            riskTextEl.style.color = "#ff9800";
        } else {
            riskMsg = "Secure: Prioritizing asset safety over user friction.";
            riskTextEl.style.color = "#4CAF50";
        }
    }
    
    riskTextEl.textContent = riskMsg;
}

// Event Listeners
slider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value, 10);
    numberInput.value = val;
    updateSimulation(val);
});

numberInput.addEventListener('input', (e) => {
    let val = parseInt(e.target.value, 10);
    if (isNaN(val)) return;
    if (val < 0) val = 0;
    if (val > 100) val = 100;
    slider.value = val;
    updateSimulation(val);
});

contextDropdown.addEventListener('change', () => {
    updateSimulation(parseInt(slider.value, 10));
});

resetBtn.addEventListener('click', () => {
    slider.value = 75;
    numberInput.value = 75;
    contextDropdown.value = "journalism";
    updateSimulation(75);
});

// Initial render
updateSimulation(75);

// --- Audio Feature Discriminator Logic ---
const discCtx = document.getElementById('discriminatorChart').getContext('2d');
const insightText = document.getElementById('insight-text');
const xAxisDropdown = document.getElementById('x-axis-dropdown');
const yAxisDropdown = document.getElementById('y-axis-dropdown');

// Generate dummy scatter data for Authentic and Synthetic
function generateScatterData(numPoints, xMin, xMax, yMin, yMax) {
    const data = [];
    for (let i = 0; i < numPoints; i++) {
        data.push({
            x: Math.random() * (xMax - xMin) + xMin,
            y: Math.random() * (yMax - yMin) + yMin
        });
    }
    return data;
}

// Features Mapping
const featureRanges = {
    authentic: {
        mfcc: { min: 60, max: 90 },
        centroid: { min: 30, max: 70 },
        zcr: { min: 20, max: 50 }
    },
    synthetic: {
        mfcc: { min: 50, max: 75 },
        centroid: { min: 50, max: 85 },
        zcr: { min: 40, max: 80 }
    }
};

let discriminatorChart = new Chart(discCtx, {
    type: 'scatter',
    data: {
        datasets: [
            {
                label: 'Authentic Speech',
                data: [], // Updated below
                backgroundColor: colorAuth,
                borderColor: '#ffffff',
                borderWidth: 1,
                pointRadius: 6,
                pointHoverRadius: 8
            },
            {
                label: 'Synthetic Deepfakes',
                data: [], // Updated below
                backgroundColor: colorSyn,
                borderColor: '#ffffff',
                borderWidth: 1,
                pointRadius: 6,
                pointHoverRadius: 8
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false } // Custom legend used
        },
        scales: {
            x: {
                grid: { color: '#333333' },
                ticks: { color: '#aaaaaa', font: { family: 'Inter' } },
                min: 0,
                max: 100
            },
            y: {
                grid: { color: '#333333' },
                ticks: { color: '#aaaaaa', font: { family: 'Inter' } },
                min: 0,
                max: 100
            }
        }
    }
});

function updateDiscriminator() {
    const xFeat = xAxisDropdown.value;
    const yFeat = yAxisDropdown.value;

    const authData = generateScatterData(
        40, 
        featureRanges.authentic[xFeat].min, featureRanges.authentic[xFeat].max,
        featureRanges.authentic[yFeat].min, featureRanges.authentic[yFeat].max
    );
    
    const synData = generateScatterData(
        40, 
        featureRanges.synthetic[xFeat].min, featureRanges.synthetic[xFeat].max,
        featureRanges.synthetic[yFeat].min, featureRanges.synthetic[yFeat].max
    );

    discriminatorChart.data.datasets[0].data = authData;
    discriminatorChart.data.datasets[1].data = synData;
    discriminatorChart.update();

    // Update Insight Text
    if (xFeat === 'mfcc' && yFeat === 'centroid') {
        insightText.textContent = "Spectral brightness isolates artificial 'metallic' vocoder artifacts, while MFCC clusters vocal tract resonance.";
    } else if (yFeat === 'zcr') {
        insightText.textContent = "Zero-Crossing Rate (ZCR) spikes often reveal robotic buzzing or high-frequency synthetic compression noise.";
    } else {
        insightText.textContent = "Analyzing distinct acoustic DNA separates organic human speech from transformer-generated clones.";
    }
}

xAxisDropdown.addEventListener('change', updateDiscriminator);
yAxisDropdown.addEventListener('change', updateDiscriminator);

// Init Discriminator
updateDiscriminator();
