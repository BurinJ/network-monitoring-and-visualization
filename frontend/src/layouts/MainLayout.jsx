import { Link } from "react-router-dom";

export default function MainLayout({ children }) {
  return (
    <div>
      <nav style={{ padding: 20, background: "#111", color: "#fff" }}>
        {/*
        <Link to="/" style={{ marginRight: 20, color: "#fff" }}>Home</Link>
        */}
        <Link to="/dashboard" style={{ marginRight: 20, color: "#fff" }}>Dashboard</Link>
        {/*
        <Link to="/docs" style={{ marginRight: 20, color: "#fff" }}>Docs</Link>
        <Link to="/about" style={{ marginRight: 20, color: "#fff" }}>About</Link>
        */}
      </nav>
      <div style={{ padding: 20 }}>
        {children}
      </div>
    </div>
  );
}