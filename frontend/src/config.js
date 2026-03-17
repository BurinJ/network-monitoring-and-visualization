// Logic:
// 1. If the URL has port 5173, we are in Dev mode (Vite) -> Use localhost:5000
// 2. If the URL has port 5000 (or anything else), we are in Production -> Use relative path '/api'

const hostname = window.location.hostname;
const port = window.location.port;

const isDevelopment = port === '5173';

// If dev, point to local backend. If prod, use relative path (browser fills in the current IP)
export const API_BASE_URL = isDevelopment 
  ? 'http://localhost:5000/api' 
  : '/api';

console.log("%c 🔧 Network Config ", "background: #222; color: #bada55; font-size:14px");
console.log("Running on:", hostname);
console.log("Port:", port);
console.log("Mode:", isDevelopment ? "Development (Vite)" : "Production (Flask)");
console.log("Target API URL:", API_BASE_URL);