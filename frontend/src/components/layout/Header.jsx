import { useState, useEffect } from 'react'
import { Wifi, WifiOff, Clock } from 'lucide-react'
import { healthAPI } from '../../services/api'
import useTrainingStore from '../../store/trainingStore'
import clsx from 'clsx'

export default function Header({ title }) {
  const [connected, setConnected] = useState(false)
  const [time, setTime] = useState(new Date())
  const { status: trainingStatus, currentRound, totalRounds, modelLoaded, modelRound, modelVersion } = useTrainingStore()
  useEffect(() => {
    const check = async () => { try { await healthAPI.check(); setConnected(true) } catch { setConnected(false) } }
    check(); const id = setInterval(check, 10000); return () => clearInterval(id)
  }, [])
  useEffect(() => { const id = setInterval(() => setTime(new Date()), 1000); return () => clearInterval(id) }, [])
  return (
    <header className="h-14 bg-slate-900/80 backdrop-blur border-b border-white/5 flex items-center justify-between px-6 flex-shrink-0">
      <h1 className="font-semibold text-white text-lg">{title}</h1>
      <div className="flex items-center gap-5 text-xs font-mono">
        <span className="text-slate-400 flex items-center gap-1.5"><Clock size={12}/>{time.toLocaleTimeString()}</span>
        {trainingStatus !== 'idle' && (
          <span className={clsx('flex items-center gap-1.5', ['running', 'initializing', 'training'].includes(trainingStatus)?'text-emerald-400':trainingStatus==='completed'?'text-cyan-400':'text-rose-400')}>
            <span className={clsx('w-1.5 h-1.5 rounded-full', ['running', 'initializing', 'training'].includes(trainingStatus)?'animate-pulse bg-emerald-400':'bg-current')}/>
            <span className="uppercase">{trainingStatus}</span>
            {['running', 'initializing', 'training'].includes(trainingStatus)&&<span className="text-slate-500">R{currentRound}/{totalRounds}</span>}
          </span>
        )}
        <span className={clsx('flex items-center gap-1.5', connected?'text-emerald-400':'text-rose-400')}>
          {connected?<Wifi size={13}/>:<WifiOff size={13}/>}
          {connected?'Connected':'Offline'}
        </span>
        <span className={clsx('flex items-center gap-1.5', modelLoaded ? 'text-emerald-400' : 'text-amber-400')}>
          <span className={clsx('w-1.5 h-1.5 rounded-full', modelLoaded ? 'bg-emerald-400' : 'bg-amber-400')} />
          {modelLoaded ? 'Model Loaded' : 'Cold Start'}
          <span className="text-slate-500">R{modelRound} / {modelVersion}</span>
        </span>
      </div>
    </header>
  )
}
