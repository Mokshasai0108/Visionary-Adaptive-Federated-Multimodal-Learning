import axios from 'axios'
const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const api = axios.create({ baseURL: BASE, timeout: 30000 })
export const trainAPI = { start: (p) => api.post('/train/start', p), status: () => api.get('/train/status'), stop: () => api.post('/train/stop') }
export const modelAPI = {
  status: () => api.get('/model/status'),
  reload: () => api.post('/model/reload'),
  list: () => api.get('/model/list'),
  switch: (modelName) => api.post('/model/switch', null, { params: { model_name: modelName } }),
}
export const metricsAPI = { get: () => api.get('/metrics') }
export const inferenceAPI = { run: (fd) => api.post('/infer', fd, { headers: { 'Content-Type': 'multipart/form-data' } }) }
export const promptsAPI = { get: () => api.get('/prompts') }
export const diagnosticsAPI = { get: () => api.get('/diagnostics') }
export const healthAPI = { check: () => api.get('/health') }
export default api
