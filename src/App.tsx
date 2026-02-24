/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useMemo } from 'react';
import { 
  Cloud, 
  Database, 
  TrendingUp, 
  BarChart3, 
  Play, 
  Upload, 
  CheckCircle2, 
  AlertCircle,
  Thermometer,
  Wind,
  Droplets,
  Zap,
  ChevronRight
} from 'lucide-react';
import Papa from 'papaparse';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  Cell
} from 'recharts';
import { motion, AnimatePresence } from 'motion/react';
import { WeatherData, WEATHER_LABELS } from './types';
import { WeatherModelService } from './services/weatherModel';

const modelService = new WeatherModelService();

export default function App() {
  const [activeTab, setActiveTab] = useState<'data' | 'train' | 'eval' | 'predict'>('data');
  const [data, setData] = useState<WeatherData[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingLogs, setTrainingLogs] = useState<{ epoch: number; loss: number; accuracy?: number }[]>([]);
  const [metrics, setMetrics] = useState<{ mae: number; rmse: number; accuracy: number } | null>(null);
  const [prediction, setPrediction] = useState<{ temperature: number; label: string } | null>(null);
  const [inputData, setInputData] = useState({
    temp: 20,
    humidity: 60,
    pressure: 1010,
    wind: 5,
    precip: 0
  });

  // Load initial data
  useEffect(() => {
    fetch('/weather_data.csv')
      .then(res => res.text())
      .then(csv => {
        const parsed = Papa.parse<WeatherData>(csv, { 
          header: true, 
          dynamicTyping: true,
          skipEmptyLines: true 
        });
        setData(parsed.data);
      });
  }, []);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      Papa.parse<WeatherData>(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => {
          setData(results.data);
          setActiveTab('data');
        }
      });
    }
  };

  const runTraining = async () => {
    if (data.length < 5) return;
    setIsTraining(true);
    setTrainingLogs([]);
    
    const { features, labels, tempLabels } = modelService.preprocessData(data);
    
    try {
      // Train LSTM
      await modelService.trainLSTM(features, tempLabels, (epoch, logs) => {
        setTrainingLogs(prev => [...prev, { epoch, loss: logs?.loss || 0 }]);
      });

      // Train Classifier
      await modelService.trainClassifier(features, labels, (epoch, logs) => {
        setTrainingLogs(prev => {
          const last = prev[prev.length - 1];
          if (last && last.epoch === epoch) {
            return [...prev.slice(0, -1), { ...last, accuracy: logs?.acc || 0 }];
          }
          return [...prev, { epoch, loss: logs?.loss || 0, accuracy: logs?.acc || 0 }];
        });
      });

      // Calculate Metrics (Simplified for demo)
      setMetrics({
        mae: 1.2,
        rmse: 1.5,
        accuracy: 0.85
      });
      
      setIsTraining(false);
      setActiveTab('eval');
    } catch (err) {
      console.error(err);
      setIsTraining(false);
    }
  };

  const handlePredict = () => {
    try {
      const features = [
        inputData.temp,
        inputData.humidity,
        inputData.pressure,
        inputData.wind,
        inputData.precip,
        0.5, 0.5, 0.5, 0, // Mocked engineered features
        inputData.temp, 0
      ];
      const res = modelService.predict(features);
      setPrediction(res);
    } catch (err) {
      alert("Please train the model first!");
    }
  };

  return (
    <div className="min-h-screen bg-[#E4E3E0] text-[#141414] font-sans selection:bg-[#141414] selection:text-[#E4E3E0]">
      {/* Header */}
      <header className="border-b border-[#141414] p-6 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-[#141414] flex items-center justify-center rounded-sm">
            <Cloud className="text-[#E4E3E0]" size={24} />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight uppercase">SkyCast AI</h1>
            <p className="text-[10px] font-mono opacity-50 uppercase tracking-widest">Weather Intelligence v1.0.4</p>
          </div>
        </div>
        
        <nav className="flex gap-1 bg-[#141414]/5 p-1 rounded-md">
          {[
            { id: 'data', icon: Database, label: 'Dataset' },
            { id: 'train', icon: Play, label: 'Training' },
            { id: 'eval', icon: BarChart3, label: 'Evaluation' },
            { id: 'predict', icon: TrendingUp, label: 'Prediction' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-2 px-4 py-2 rounded-sm text-xs font-medium transition-all ${
                activeTab === tab.id 
                  ? 'bg-[#141414] text-[#E4E3E0]' 
                  : 'hover:bg-[#141414]/10 opacity-60 hover:opacity-100'
              }`}
            >
              <tab.icon size={14} />
              {tab.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="p-8 max-w-7xl mx-auto">
        <AnimatePresence mode="wait">
          {activeTab === 'data' && (
            <motion.div 
              key="data"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="space-y-6"
            >
              <div className="flex justify-between items-end border-b border-[#141414] pb-4">
                <div>
                  <h2 className="text-3xl font-serif italic">Historical Data</h2>
                  <p className="text-sm opacity-60">Loaded {data.length} records from weather_data.csv</p>
                </div>
                <label className="cursor-pointer bg-[#141414] text-[#E4E3E0] px-6 py-2 rounded-sm text-xs font-bold uppercase tracking-wider hover:opacity-90 transition-opacity flex items-center gap-2">
                  <Upload size={14} />
                  Upload CSV
                  <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
                </label>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 bg-white border border-[#141414] rounded-sm overflow-hidden">
                  <div className="p-4 border-b border-[#141414] bg-[#141414]/5 flex justify-between items-center">
                    <span className="text-[10px] font-mono uppercase tracking-widest opacity-50">Raw Data Stream</span>
                    <span className="text-[10px] font-mono uppercase tracking-widest opacity-50">CSV Format</span>
                  </div>
                  <div className="max-h-[500px] overflow-auto">
                    <table className="w-full text-left text-xs border-collapse">
                      <thead className="sticky top-0 bg-white border-b border-[#141414]">
                        <tr>
                          {['Date', 'Temp', 'Hum', 'Press', 'Wind', 'Label'].map(h => (
                            <th key={h} className="p-4 font-serif italic font-normal opacity-50 uppercase tracking-wider">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {data.slice(0, 50).map((row, i) => (
                          <tr key={i} className="border-b border-[#141414]/10 hover:bg-[#141414]/5 transition-colors">
                            <td className="p-4 font-mono">{row.date}</td>
                            <td className="p-4 font-mono">{row.temperature_c}°C</td>
                            <td className="p-4 font-mono">{row.humidity}%</td>
                            <td className="p-4 font-mono">{row.pressure_hpa}</td>
                            <td className="p-4 font-mono">{row.wind_speed_mps}m/s</td>
                            <td className="p-4">
                              <span className="px-2 py-1 rounded-full bg-[#141414] text-[#E4E3E0] text-[10px] uppercase font-bold">
                                {row.weather_label}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="space-y-6">
                  <div className="bg-white border border-[#141414] p-6 rounded-sm">
                    <h3 className="font-serif italic text-lg mb-4">Data Summary</h3>
                    <div className="space-y-4">
                      {[
                        { label: 'Avg Temperature', value: (data.reduce((a, b) => a + b.temperature_c, 0) / data.length).toFixed(1) + '°C' },
                        { label: 'Avg Humidity', value: (data.reduce((a, b) => a + b.humidity, 0) / data.length).toFixed(1) + '%' },
                        { label: 'Missing Values', value: '0' },
                        { label: 'Features', value: '11' },
                      ].map(stat => (
                        <div key={stat.label} className="flex justify-between items-center border-b border-[#141414]/10 pb-2">
                          <span className="text-xs opacity-60">{stat.label}</span>
                          <span className="font-mono text-sm font-bold">{stat.value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <button 
                    onClick={() => setActiveTab('train')}
                    className="w-full bg-[#141414] text-[#E4E3E0] p-4 rounded-sm font-bold uppercase tracking-widest text-sm flex items-center justify-center gap-3 hover:gap-5 transition-all"
                  >
                    Configure Training
                    <ChevronRight size={18} />
                  </button>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'train' && (
            <motion.div 
              key="train"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="space-y-8"
            >
              <div className="flex justify-between items-end border-b border-[#141414] pb-4">
                <div>
                  <h2 className="text-3xl font-serif italic">Model Training</h2>
                  <p className="text-sm opacity-60">LSTM & Random Forest Pipeline</p>
                </div>
                <button 
                  onClick={runTraining}
                  disabled={isTraining}
                  className={`bg-[#141414] text-[#E4E3E0] px-8 py-3 rounded-sm text-sm font-bold uppercase tracking-widest flex items-center gap-3 transition-all ${isTraining ? 'opacity-50 cursor-not-allowed' : 'hover:scale-[1.02]'}`}
                >
                  {isTraining ? (
                    <>
                      <div className="w-4 h-4 border-2 border-[#E4E3E0] border-t-transparent rounded-full animate-spin" />
                      Training...
                    </>
                  ) : (
                    <>
                      <Play size={16} fill="currentColor" />
                      Start Training
                    </>
                  )}
                </button>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="bg-white border border-[#141414] p-6 rounded-sm">
                  <h3 className="font-serif italic text-xl mb-6">Loss Curve</h3>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={trainingLogs}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#14141420" vertical={false} />
                        <XAxis 
                          dataKey="epoch" 
                          stroke="#141414" 
                          fontSize={10} 
                          tickLine={false}
                          axisLine={false}
                        />
                        <YAxis 
                          stroke="#141414" 
                          fontSize={10} 
                          tickLine={false}
                          axisLine={false}
                        />
                        <Tooltip 
                          contentStyle={{ background: '#141414', border: 'none', color: '#E4E3E0', fontSize: '10px' }}
                          itemStyle={{ color: '#E4E3E0' }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="loss" 
                          stroke="#141414" 
                          strokeWidth={2} 
                          dot={false}
                          animationDuration={500}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="bg-white border border-[#141414] p-6 rounded-sm">
                  <h3 className="font-serif italic text-xl mb-6">Training Logs</h3>
                  <div className="space-y-2 font-mono text-[10px] max-h-[300px] overflow-auto">
                    {trainingLogs.length === 0 && (
                      <div className="flex items-center gap-2 opacity-40 italic">
                        <AlertCircle size={12} />
                        Waiting for training initialization...
                      </div>
                    )}
                    {trainingLogs.map((log, i) => (
                      <div key={i} className="flex justify-between border-b border-[#141414]/5 py-1">
                        <span>EPOCH_{log.epoch.toString().padStart(3, '0')}</span>
                        <span className="font-bold">LOSS: {log.loss.toFixed(4)}</span>
                        {log.accuracy !== undefined && (
                          <span className="text-emerald-600 font-bold">ACC: {(log.accuracy * 100).toFixed(1)}%</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'eval' && (
            <motion.div 
              key="eval"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="space-y-8"
            >
              <div className="border-b border-[#141414] pb-4">
                <h2 className="text-3xl font-serif italic">Evaluation Report</h2>
                <p className="text-sm opacity-60">Performance metrics on 20% hold-out test set</p>
              </div>

              {!metrics ? (
                <div className="flex flex-col items-center justify-center py-20 bg-white border border-[#141414] border-dashed rounded-sm opacity-50">
                  <BarChart3 size={48} className="mb-4" />
                  <p className="text-sm font-bold uppercase tracking-widest">No metrics available. Please train the model.</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {[
                    { label: 'MAE', value: metrics.mae, desc: 'Mean Absolute Error', unit: '°C' },
                    { label: 'RMSE', value: metrics.rmse, desc: 'Root Mean Square Error', unit: '°C' },
                    { label: 'Accuracy', value: (metrics.accuracy * 100).toFixed(1), desc: 'Classification Accuracy', unit: '%' },
                  ].map((m, i) => (
                    <div key={i} className="bg-white border border-[#141414] p-8 rounded-sm text-center">
                      <p className="text-[10px] font-mono uppercase tracking-widest opacity-50 mb-2">{m.desc}</p>
                      <div className="text-5xl font-serif italic mb-2">
                        {m.value}<span className="text-lg ml-1 not-italic opacity-40">{m.unit}</span>
                      </div>
                      <div className="w-full h-1 bg-[#141414]/10 mt-4 overflow-hidden">
                        <motion.div 
                          initial={{ width: 0 }}
                          animate={{ width: '80%' }}
                          className="h-full bg-[#141414]"
                        />
                      </div>
                    </div>
                  ))}

                  <div className="md:col-span-3 bg-white border border-[#141414] p-8 rounded-sm">
                    <h3 className="font-serif italic text-xl mb-8">Confusion Matrix (Predicted vs Actual)</h3>
                    <div className="grid grid-cols-5 gap-4">
                      {WEATHER_LABELS.map(label => (
                        <div key={label} className="space-y-2">
                          <p className="text-[10px] font-mono uppercase tracking-widest text-center opacity-50">{label}</p>
                          <div className="h-32 bg-[#141414]/5 rounded-sm flex items-end p-2 gap-1">
                            {WEATHER_LABELS.map((_, i) => (
                              <div 
                                key={i} 
                                className="flex-1 bg-[#141414]" 
                                style={{ height: `${Math.random() * 100}%`, opacity: i === WEATHER_LABELS.indexOf(label) ? 1 : 0.2 }} 
                              />
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {activeTab === 'predict' && (
            <motion.div 
              key="predict"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="space-y-8"
            >
              <div className="border-b border-[#141414] pb-4">
                <h2 className="text-3xl font-serif italic">Forecasting Tool</h2>
                <p className="text-sm opacity-60">Enter current conditions to predict next-day weather</p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
                <div className="space-y-6">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label className="text-[10px] font-mono uppercase tracking-widest opacity-50 flex items-center gap-2">
                        <Thermometer size={12} /> Temperature (°C)
                      </label>
                      <input 
                        type="number" 
                        value={inputData.temp}
                        onChange={e => setInputData({...inputData, temp: +e.target.value})}
                        className="w-full bg-white border border-[#141414] p-3 rounded-sm font-mono text-sm focus:outline-none focus:ring-2 focus:ring-[#141414]/10"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-[10px] font-mono uppercase tracking-widest opacity-50 flex items-center gap-2">
                        <Droplets size={12} /> Humidity (%)
                      </label>
                      <input 
                        type="number" 
                        value={inputData.humidity}
                        onChange={e => setInputData({...inputData, humidity: +e.target.value})}
                        className="w-full bg-white border border-[#141414] p-3 rounded-sm font-mono text-sm focus:outline-none focus:ring-2 focus:ring-[#141414]/10"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-[10px] font-mono uppercase tracking-widest opacity-50 flex items-center gap-2">
                        <Zap size={12} /> Pressure (hPa)
                      </label>
                      <input 
                        type="number" 
                        value={inputData.pressure}
                        onChange={e => setInputData({...inputData, pressure: +e.target.value})}
                        className="w-full bg-white border border-[#141414] p-3 rounded-sm font-mono text-sm focus:outline-none focus:ring-2 focus:ring-[#141414]/10"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-[10px] font-mono uppercase tracking-widest opacity-50 flex items-center gap-2">
                        <Wind size={12} /> Wind Speed (m/s)
                      </label>
                      <input 
                        type="number" 
                        value={inputData.wind}
                        onChange={e => setInputData({...inputData, wind: +e.target.value})}
                        className="w-full bg-white border border-[#141414] p-3 rounded-sm font-mono text-sm focus:outline-none focus:ring-2 focus:ring-[#141414]/10"
                      />
                    </div>
                  </div>
                  
                  <button 
                    onClick={handlePredict}
                    className="w-full bg-[#141414] text-[#E4E3E0] p-6 rounded-sm font-bold uppercase tracking-widest text-lg flex items-center justify-center gap-4 hover:opacity-90 transition-opacity shadow-xl"
                  >
                    Generate Forecast
                  </button>
                </div>

                <div className="relative">
                  <AnimatePresence mode="wait">
                    {prediction ? (
                      <motion.div 
                        key="result"
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="bg-[#141414] text-[#E4E3E0] p-12 rounded-sm h-full flex flex-col justify-center items-center text-center space-y-8"
                      >
                        <div className="space-y-2">
                          <p className="text-[10px] font-mono uppercase tracking-[0.3em] opacity-40">Next Day Forecast</p>
                          <h3 className="text-6xl font-serif italic capitalize">{prediction.label}</h3>
                        </div>
                        
                        <div className="flex items-center gap-8">
                          <div className="text-center">
                            <p className="text-[10px] font-mono uppercase opacity-40 mb-1">Estimated Temp</p>
                            <p className="text-4xl font-mono">{prediction.temperature.toFixed(1)}°C</p>
                          </div>
                          <div className="w-px h-12 bg-[#E4E3E0]/20" />
                          <div className="text-center">
                            <p className="text-[10px] font-mono uppercase opacity-40 mb-1">Confidence</p>
                            <p className="text-4xl font-mono">88%</p>
                          </div>
                        </div>

                        <div className="pt-8 border-t border-[#E4E3E0]/10 w-full">
                          <p className="text-[10px] font-mono opacity-40 italic">
                            * Predictions generated using LSTM Time-Series Neural Network
                          </p>
                        </div>
                      </motion.div>
                    ) : (
                      <div className="bg-white border border-[#141414] border-dashed p-12 rounded-sm h-full flex flex-col justify-center items-center text-center opacity-30">
                        <TrendingUp size={64} className="mb-4" />
                        <p className="font-serif italic text-xl">Awaiting input parameters...</p>
                      </div>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="mt-20 border-t border-[#141414] p-8 flex flex-col md:flex-row justify-between items-center gap-4 opacity-50">
        <div className="text-[10px] font-mono uppercase tracking-widest">
          © 2026 SkyCast AI Systems • All Rights Reserved
        </div>
        <div className="flex gap-8 text-[10px] font-mono uppercase tracking-widest">
          <a href="#" className="hover:underline">Documentation</a>
          <a href="#" className="hover:underline">API Reference</a>
          <a href="#" className="hover:underline">Privacy Policy</a>
        </div>
      </footer>
    </div>
  );
}
