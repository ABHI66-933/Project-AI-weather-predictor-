export interface WeatherData {
  date: string;
  temperature_c: number;
  humidity: number;
  pressure_hpa: number;
  wind_speed_mps: number;
  precipitation_mm: number;
  weather_label: string;
}

export interface ProcessedFeatures {
  day_of_year: number;
  month: number;
  weekday: number;
  is_weekend: number;
  rolling_mean_temp_3: number;
  rolling_std_temp_7: number;
  [key: string]: number;
}

export interface ModelMetrics {
  mae: number;
  rmse: number;
  r2: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

export type WeatherLabel = 'clear' | 'cloudy' | 'rain' | 'storm' | 'snow';

export const WEATHER_LABELS: WeatherLabel[] = ['clear', 'cloudy', 'rain', 'storm', 'snow'];
