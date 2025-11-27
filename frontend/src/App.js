import { BrowserRouter, Routes, Route } from "react-router-dom";
import MainLayout from "./layouts/MainLayout";
import Home from "./pages/Home";
import Dashboard from "./pages/Dashboard";
import Documentation from "./pages/Documentation";
import About from "./pages/About";

function App() {
  return (
    <BrowserRouter>
      <MainLayout>
        <Routes>
          {/* 
          <Route path="/" element={<Home />} />
          */}
          <Route path="/dashboard" element={<Dashboard />} />
          {/* 
          <Route path="/docs" element={<Documentation />} />
          <Route path="/about" element={<About />} />
          */}
        </Routes>
      </MainLayout>
    </BrowserRouter>
  );
}

export default App;