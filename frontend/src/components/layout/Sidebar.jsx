import { NavLink } from "react-router-dom"
import { motion, AnimatePresence } from "framer-motion"
import { LayoutDashboard, Play, Zap, Network, Eye, BarChart3, AlertTriangle, ChevronLeft, BrainCircuit, Replace } from "lucide-react"
import { useStore } from "../../store/useStore"
import clsx from "clsx"

const NAV = [
  { to: "/", icon: LayoutDashboard, label: "Dashboard" },
  { to: "/training", icon: Play, label: "Training" },
  { to: "/inference", icon: Zap, label: "Inference" },
  { to: "/mirror", icon: Replace, label: "Mirror Workspace" },
  { to: "/federated", icon: Network, label: "Federation" },
  { to: "/prompts", icon: Eye, label: "Prompt Inspector" },
  { to: "/results", icon: BarChart3, label: "Results" },
  { to: "/diagnostics", icon: AlertTriangle, label: "Diagnostics" },
]

export default function Sidebar() {
  const { sidebarOpen, toggleSidebar } = useStore()
  return (
    <motion.aside animate={{ width: sidebarOpen ? 240 : 64 }} transition={{ type: "spring", stiffness: 300, damping: 30 }}
      className="relative flex-shrink-0 h-screen border-r border-white/5 bg-slate-900/80 backdrop-blur flex flex-col z-20">
      <div className="flex items-center gap-3 px-4 py-5 border-b border-white/5">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-400 to-emerald-400 flex items-center justify-center flex-shrink-0">
          <BrainCircuit size={16} className="text-slate-900" />
        </div>
        <AnimatePresence>{sidebarOpen && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <p className="font-semibold text-sm text-white leading-none">AFSPL</p>
            <p className="text-[10px] text-cyan-400/60 font-mono mt-0.5">Federated Prompts</p>
          </motion.div>
        )}</AnimatePresence>
      </div>
      <nav className="flex-1 px-2 py-4 space-y-1 overflow-hidden">
        {NAV.map(({ to, icon: Icon, label }) => (
          <NavLink key={to} to={to} end={to === "/"}>
            {({ isActive }) => (
              <div className={clsx("flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all cursor-pointer",
                isActive ? "bg-cyan-400/10 text-cyan-400 border border-cyan-400/20" : "text-slate-400 hover:text-white hover:bg-white/5")}>
                <Icon size={16} className="flex-shrink-0" />
                <AnimatePresence>{sidebarOpen && (
                  <motion.span initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }}
                    className="text-sm whitespace-nowrap">{label}</motion.span>
                )}</AnimatePresence>
              </div>
            )}
          </NavLink>
        ))}
      </nav>
      <button onClick={toggleSidebar} className="m-3 p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white flex items-center justify-center">
        <motion.div animate={{ rotate: sidebarOpen ? 0 : 180 }}><ChevronLeft size={16} /></motion.div>
      </button>
    </motion.aside>
  )
}
