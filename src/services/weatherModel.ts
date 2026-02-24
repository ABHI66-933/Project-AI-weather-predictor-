import * as tf from '@tensorflow/tfjs';
import { WeatherData, WEATHER_LABELS } from '../types';
import { format, parseISO, getDayOfYear, getMonth, getDay } from 'date-fns';

export class WeatherModelService {
  private scaler: { mean: number[]; std: number[] } | null = null;
  private labelEncoder: Map<string, number> = new Map();
  private lstmModel: tf.LayersModel | null = null;
  private classifierModel: tf.LayersModel | null = null;

  constructor() {
    WEATHER_LABELS.forEach((label, index) => {
      this.labelEncoder.set(label, index);
    });
  }

  public preprocessData(data: WeatherData[]): { features: number[][]; labels: number[]; tempLabels: number[] } {
    // Sort by date
    const sortedData = [...data].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

    const features: number[][] = [];
    const labels: number[] = [];
    const tempLabels: number[] = [];

    for (let i = 0; i < sortedData.length; i++) {
      const row = sortedData[i];
      const date = parseISO(row.date);

      // Feature Engineering
      const dayOfYear = getDayOfYear(date) / 366;
      const month = getMonth(date) / 12;
      const weekday = getDay(date) / 7;
      const isWeekend = (weekday === 0 || weekday === 6) ? 1 : 0;

      // Rolling features (simplified for this demo)
      let rollingMean3 = row.temperature_c;
      if (i >= 2) {
        rollingMean3 = (sortedData[i].temperature_c + sortedData[i-1].temperature_c + sortedData[i-2].temperature_c) / 3;
      }

      let rollingStd7 = 0;
      if (i >= 6) {
        const slice = sortedData.slice(i - 6, i + 1).map(d => d.temperature_c);
        const mean = slice.reduce((a, b) => a + b, 0) / 7;
        rollingStd7 = Math.sqrt(slice.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / 7);
      }

      const featureRow = [
        row.temperature_c,
        row.humidity,
        row.pressure_hpa,
        row.wind_speed_mps,
        row.precipitation_mm,
        dayOfYear,
        month,
        weekday,
        isWeekend,
        rollingMean3,
        rollingStd7
      ];

      features.push(featureRow);
      labels.push(this.labelEncoder.get(row.weather_label) || 0);
      tempLabels.push(row.temperature_c);
    }

    return { features, labels, tempLabels };
  }

  public async trainLSTM(features: number[][], targets: number[], onEpochEnd?: (epoch: number, logs?: tf.Logs) => void) {
    // Prepare sequences (N=3 days window)
    const windowSize = 3;
    const X: number[][][] = [];
    const y: number[] = [];

    for (let i = windowSize; i < features.length; i++) {
      X.push(features.slice(i - windowSize, i));
      y.push(targets[i]);
    }

    const xs = tf.tensor3d(X);
    const ys = tf.tensor2d(y, [y.length, 1]);

    const model = tf.sequential();
    model.add(tf.layers.lstm({
      units: 32,
      inputShape: [windowSize, features[0].length],
      returnSequences: false
    }));
    model.add(tf.layers.dense({ units: 1 }));

    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: 'meanSquaredError'
    });

    await model.fit(xs, ys, {
      epochs: 50,
      batchSize: 8,
      callbacks: {
        onEpochEnd: (epoch, logs) => onEpochEnd?.(epoch, logs)
      }
    });

    this.lstmModel = model;
    return model;
  }

  public async trainClassifier(features: number[][], labels: number[], onEpochEnd?: (epoch: number, logs?: tf.Logs) => void) {
    const xs = tf.tensor2d(features);
    const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), WEATHER_LABELS.length);

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [features[0].length] }));
    model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
    model.add(tf.layers.dense({ units: WEATHER_LABELS.length, activation: 'softmax' }));

    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    await model.fit(xs, ys, {
      epochs: 50,
      batchSize: 8,
      callbacks: {
        onEpochEnd: (epoch, logs) => onEpochEnd?.(epoch, logs)
      }
    });

    this.classifierModel = model;
    return model;
  }

  public predict(currentFeatures: number[]): { temperature: number; label: string } {
    if (!this.lstmModel || !this.classifierModel) {
      throw new Error("Models not trained");
    }

    // For LSTM, we need a sequence. In a real app we'd store history.
    // For this demo, we'll simulate a sequence by repeating the current features.
    const sequence = [currentFeatures, currentFeatures, currentFeatures];
    const lstmInput = tf.tensor3d([sequence]);
    const tempPred = this.lstmModel.predict(lstmInput) as tf.Tensor;
    const temperature = tempPred.dataSync()[0];

    const classInput = tf.tensor2d([currentFeatures]);
    const classPred = this.classifierModel.predict(classInput) as tf.Tensor;
    const labelIndex = classPred.argMax(-1).dataSync()[0];
    const label = WEATHER_LABELS[labelIndex];

    return { temperature, label };
  }
}
