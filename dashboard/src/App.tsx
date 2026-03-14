import React from "react";
import { Link, Route, Routes } from "react-router-dom";
import { ExperimentsList } from "./pages/ExperimentsList";
import { ExperimentDetail } from "./pages/ExperimentDetail";
import { StudyView } from "./pages/StudyView";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || window.location.origin;

const App: React.FC = () => {
  return (
    <div className="app-root">
      <header className="app-header">
        <div className="app-header-inner">
          <h1 className="app-title">
            <Link to="/">AutoForge</Link>
          </h1>
          <nav className="app-nav">
            <Link to="/">Experiments</Link>
          </nav>
        </div>
      </header>
      <main className="app-main">
        <Routes>
          <Route path="/" element={<ExperimentsList />} />
          <Route path="/experiments/:id" element={<ExperimentDetail />} />
          <Route path="/studies/:studyName" element={<StudyView />} />
        </Routes>
      </main>
    </div>
  );
};

export default App;
